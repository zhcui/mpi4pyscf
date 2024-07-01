#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Zhi-Hao Cui <zhcui0408@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#         Huanchen Zhai <hczhai@caltech.edu>
#

"""
MPI-GCCSD with real intergals.

Usage: mpirun -np 2 python gccsd.py
"""

import gc
import os
import time
from functools import reduce
from functools import partial
import h5py
import numpy as np
import scipy.linalg as la

from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf import scf
from pyscf.cc import ccsd as rccsd
from pyscf.cc import gccsd
from pyscf import __config__

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.ccsd import (kernel, _task_location, _sync_,
                               _pack_scf, _rotate_vir_block,
                               distribute_amplitudes_, gather_amplitudes,
                               gather_lambda, restore_from_diis_)
from mpi4pyscf.cc import gccsd_lambda
from mpi4pyscf.cc import gccsd_rdm

comm = mpi.comm
rank = mpi.rank

einsum = lib.einsum
#einsum = partial(np.einsum, optimize=True)
einsum_mv = lib.einsum 
#einsum_mv = partial(np.einsum, optimize=True)

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

def update_amps(mycc, t1, t2, eris):
    """
    Update GCCSD amplitudes.
    """
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1T = t1.T
    t2T = np.asarray(t2.transpose(2, 3, 0, 1), order='C')
    nvir_seg, nvir, nocc = t2T.shape[:3]
    t1 = t2 = None
    ntasks = mpi.pool.size
    
    vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vloc0, vloc1 = vlocs[rank]
    log.debug2('vlocs %s', vlocs)
    assert vloc1 - vloc0 == nvir_seg

    fock = eris.fock
    fvo = fock[nocc:, :nocc]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift

    tauT_tilde = make_tauT(t1T, t2T, fac=0.5, vlocs=vlocs)
    Fvv = cc_Fvv(t1T, t2T, eris, tauT_tilde=tauT_tilde, vlocs=vlocs)
    Foo = cc_Foo(t1T, t2T, eris, tauT_tilde=tauT_tilde, vlocs=vlocs)
    tauT_tilde = None
    Fov = cc_Fov(t1T, eris, vlocs=vlocs)
    
    rk = comm.allreduce(getattr(mycc, "rk", None), op=mpi.MPI.LOR)
    if not rk:
        # Move energy terms to the other side
        Fvv[np.diag_indices(nvir)] -= mo_e_v
        Foo[np.diag_indices(nocc)] -= mo_e_o
    
    # T1 equation
    t1Tnew  = np.dot(Fvv, t1T)
    t1Tnew -= np.dot(t1T, Foo)

    tmp  = einsum_mv('aeim, me -> ai', t2T, Fov)
    tmp -= einsum_mv('fn, anfi -> ai', t1T, eris.xovo)
    tmp  = mpi.allgather(tmp)
    
    tmp2  = einsum('eamn, einm -> ai', t2T, eris.xooo)
    tmp2 += einsum('efim, efam -> ai', t2T, eris.xvvo)
    tmp2 *= 0.5
    tmp2  = mpi.allreduce_inplace(tmp2)
    tmp  += tmp2
    tmp2  = None

    t1Tnew += tmp
    t1Tnew += fvo

    # T2 equation
    Ftmp = Fvv - 0.5 * np.dot(t1T, Fov)
    t2Tnew = einsum('aeij, be -> abij', t2T, Ftmp)
    t2T_tmp = mpi.alltoall_new([t2Tnew[:, p0:p1] for p0, p1 in vlocs],
                               split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = t2T_tmp[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        t2Tnew[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None
    t2T_tmp = None

    Ftmp = Foo + 0.5 * np.dot(Fov, t1T)
    tmp = einsum('abim, mj -> abij', t2T, Ftmp)
    t2Tnew -= tmp
    t2Tnew += tmp.transpose(0, 1, 3, 2)
    tmp = None
    
    t2Tnew += np.asarray(eris.xvoo)
    tauT = make_tauT(t1T, t2T, vlocs=vlocs)
    Woooo = cc_Woooo(t1T, t2T, eris, tauT=tauT, vlocs=vlocs)
    Woooo *= 0.5
    t2Tnew += einsum('abmn, mnij -> abij', tauT, Woooo)
    Woooo = None

    Wvvvv = cc_Wvvvv(t1T, t2T, eris, tauT=tauT, vlocs=vlocs)
    for task_id, tauT_tmp, p0, p1 in _rotate_vir_block(tauT, vlocs=vlocs):
        tmp = einsum('abef, efij -> abij', Wvvvv[:, :, p0:p1], tauT_tmp)
        tmp *= 0.5
        t2Tnew += tmp
        tmp = tauT_tmp = None
    Wvvvv = None
    tauT = None

    tmp = einsum('bmej, ei -> bmij', eris.xovo, t1T) # [b]mij
    tmp = mpi.allgather_new(tmp) # bmij
    tmp = einsum('am, bmij -> abij', t1T[vloc0:vloc1], tmp) # [a]bij

    Wvovo = cc_Wovvo(t1T, t2T, eris, vlocs=vlocs).transpose(2, 0, 1, 3)
    for task_id, w_tmp, p0, p1 in _rotate_vir_block(Wvovo, vlocs=vlocs):
        tmp += einsum('aeim, embj -> abij', t2T[:, p0:p1], w_tmp)
        w_tmp = None
    Wvovo = None

    tmp = tmp - tmp.transpose(0, 1, 3, 2)
    t2Tnew += tmp
    tmpT = mpi.alltoall_new([tmp[:, p0:p1] for p0, p1 in vlocs],
                            split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        t2Tnew[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None
    tmpT = None

    tmp = einsum('ei, abej -> abij', t1T, eris.xvvo)
    t2Tnew += tmp
    t2Tnew -= tmp.transpose(0, 1, 3, 2)

    tmp = einsum('am, bmji -> baij', t1T, eris.xooo)
    t2Tnew += tmp
    tmpT = mpi.alltoall_new([tmp[:, p0:p1] for p0, p1 in vlocs],
                            split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        t2Tnew[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None
    tmpT = None
    
    if not rk:
        if comm.allreduce(getattr(mycc, "dt", None), op=mpi.MPI.LOR):
            # ZHC NOTE imagninary time evolution
            #if getattr(mycc, "ignore_level_shift", True):
            #    mo_e_v = eris.mo_energy[nocc:]
            #    eia = mo_e_o[:, None] - mo_e_v
            #else:
            #    eia = mo_e_o[:, None] - mo_e_v
            eia = mo_e_o[:, None] - mo_e_v
            
            dt = mycc.dt
            t1Tnew *= (-dt)
            t1Tnew += t1T * (1.0 + dt * eia.T)
            
            t2Tnew *= (-dt)
            eia *= dt
            for i in range(vloc0, vloc1):
                ebij = lib.direct_sum('i + jb -> bij', eia[:, i] + 1, eia)
                t2Tnew[i-vloc0] += t2T[i-vloc0] * ebij
        else:
            eia = mo_e_o[:, None] - mo_e_v
            t1Tnew /= eia.T
            for i in range(vloc0, vloc1):
                t2Tnew[i-vloc0] /= lib.direct_sum('i + jb -> bij', eia[:, i], eia)
    if comm.allreduce(getattr(mycc, "frozen_abab", False), op=mpi.MPI.LOR):
        mycc.remove_t2_abab(t2Tnew.transpose(2, 3, 0, 1))
    if comm.allreduce(getattr(mycc, "frozen_aaaa_bbbb", False), op=mpi.MPI.LOR):
        mycc.remove_t2_aaaa_bbbb(t2Tnew.transpose(2, 3, 0, 1))
    if getattr(mycc, "t1_frozen_list", None) or getattr(mycc, "t2_frozen_list", None):
        mycc.remove_amps(t1Tnew.T, t2Tnew.transpose(2, 3, 0, 1), 
                         t1_frozen_list=mycc.t1_frozen_list,
                         t2_frozen_list=mycc.t2_frozen_list)
    if getattr(mycc, "t1_fix_list", None) or getattr(mycc, "t2_fix_list", None):
        mycc.remove_amps(t1Tnew.T, t2Tnew.transpose(2, 3, 0, 1), 
                         t1_frozen_list=mycc.t1_fix_list,
                         t2_frozen_list=mycc.t2_fix_list)

    time0 = log.timer_debug1('update t1 t2', *time0)
    return t1Tnew.T, t2Tnew.transpose(2, 3, 0, 1)

def make_tauT(t1T, t2T, fac=1, vlocs=None):
    """
    Make effective t2T (abij) using t2T and t1T.

    Args:
        t1T: ai
        t2T: [a]bij, a is segmented.
        fac: factor

    Returns:
        tauT: [a]bij, a is segmented.
    """
    nvir_seg, nvir, nocc, _ = t2T.shape
    if vlocs is None:
        ntasks = mpi.pool.size
        vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vloc0, vloc1 = vlocs[rank]
    tauT = np.einsum('ai, bj -> abij', t1T[vloc0:vloc1] * (fac * 0.5), t1T, optimize=True)
    tauT = tauT - tauT.transpose(0, 1, 3, 2)
    ##:tauT = tauT - tauT.transpose(1, 0, 2, 3)
    tauT_tmp = mpi.alltoall_new([tauT[:, p0:p1] for p0, p1 in vlocs], split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = tauT_tmp[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        tauT[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
    tauT += t2T
    return tauT

def cc_Fov(t1T, eris, vlocs=None):
    """
    Fov: me.
    """
    nvir, nocc = t1T.shape
    if vlocs is None:
        ntasks = mpi.pool.size
        vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    fov  = eris.fock[:nocc, nocc:]
    Fme  = einsum_mv('efmn, fn -> em', eris.xvoo, t1T)
    Fme  = mpi.allgather(Fme).T
    Fme += fov
    return Fme

def cc_Fvv(t1T, t2T, eris, tauT_tilde=None, vlocs=None):
    """
    Fvv: ae.
    """
    nvir, nocc = t1T.shape
    if vlocs is None:
        ntasks = mpi.pool.size
        vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    if tauT_tilde is None:
        tauT_tilde = make_tauT(t1T, t2T, fac=0.5, vlocs=vlocs)
    vloc0, vloc1 = vlocs[rank]

    fvo = eris.fock[nocc:, :nocc]
    fvv = eris.fock[nocc+vloc0:nocc+vloc1, nocc:]
    Fea = fvv - 0.5 * np.dot(fvo[vloc0:vloc1], t1T.T)

    Fae = (-0.5) * einsum('femn, famn -> ae', eris.xvoo, tauT_tilde)
    Fae = mpi.allreduce_inplace(Fae)
    tauT_tilde = None
    
    Fea += einsum_mv('efam, fm -> ea', eris.xvvo, t1T)
    Fae += mpi.allgather(Fea).T
    return Fae

def cc_Foo(t1T, t2T, eris, tauT_tilde=None, vlocs=None):
    nvir, nocc = t1T.shape
    if vlocs is None:
        ntasks = mpi.pool.size
        vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    if tauT_tilde is None:
        tauT_tilde = make_tauT(t1T, t2T, fac=0.5, vlocs=vlocs)
    vloc0, vloc1 = vlocs[rank]
    
    Fmi  = 0.5 * einsum('efmn, efin -> mi', eris.xvoo, tauT_tilde)
    tauT_tilde = None
    Fmi += einsum_mv('einm, en -> mi', eris.xooo, t1T[vloc0:vloc1])
    Fmi  = mpi.allreduce_inplace(Fmi)

    fov = eris.fock[:nocc, nocc:]
    foo = eris.fock[:nocc, :nocc]

    Fmi += foo
    Fmi += 0.5 * np.dot(fov, t1T)
    return Fmi

def cc_Woooo(t1T, t2T, eris, tauT=None, vlocs=None):
    nvir = t1T.shape[0]
    if vlocs is None:
        ntasks = mpi.pool.size
        vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vloc0, vloc1 = vlocs[rank]
    if tauT is None:
        tauT = make_tauT(t1T, t2T, vlocs=vlocs)

    Wmnij = einsum('efmn, efij -> mnij', eris.xvoo, tauT)
    Wmnij *= 0.25
    tauT = None
    
    tmp = einsum('eimn, ej -> mnij', eris.xooo, t1T[vloc0:vloc1])
    Wmnij -= tmp
    Wmnij += tmp.transpose(0, 1, 3, 2)
    tmp = None
    Wmnij  = mpi.allreduce_inplace(Wmnij)
    Wmnij += eris.oooo
    return Wmnij

def cc_Wvvvv(t1T, t2T, eris, tauT=None, vlocs=None):
    """
    Wvvvv intermidiates.
    """
    # ZHC TODO make Wvvvv outcore
    nvir_seg, nvir, nocc, _ = t2T.shape
    if vlocs is None:
        ntasks = mpi.pool.size
        vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vloc0, vloc1 = vlocs[rank]
    if tauT is None:
        tauT = make_tauT(t1T, t2T, vlocs=vlocs)

    Wabef = np.empty((vloc1-vloc0, nvir, nvir, nvir))
    eris_vvoo = eris.xvoo

    tauT = tauT * 0.25
    for task_id, eri_tmp, p0, p1 in _rotate_vir_block(eris_vvoo, vlocs=vlocs):
        Wabef[:, :, p0:p1] = einsum('abmn, efmn -> abef', tauT, eri_tmp)
        eri_tmp = None
    eris_vvoo = None
    tauT = None

    Wabef += np.asarray(eris.xvvv)
    tmp = einsum('bm, amef -> abfe', t1T, eris.xovv)
    Wabef += tmp

    tmpT = mpi.alltoall_new([tmp[:, p0:p1] for p0, p1 in vlocs],
                            split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nvir, nvir)
        Wabef[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None

    return Wabef

def cc_Wovvo(t1T, t2T, eris, vlocs=None):
    """
    mb[e]j.
    """
    nvir_seg, nvir, nocc, _ = t2T.shape
    if vlocs is None:
        ntasks = mpi.pool.size
        vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]

    Wmbej = einsum('efmn, fj -> mnej', eris.xvoo, -t1T)
    Wmbej = einsum('mnej, bn -> mbej', Wmbej, t1T)
    
    Wmbej -= einsum('efbm, fj -> mbej', eris.xvvo, t1T)
    Wmbej += einsum('bn, ejnm -> mbej', t1T, eris.xooo)

    for task_id, t2T_tmp, p0, p1 in _rotate_vir_block(t2T, vlocs=vlocs):
        tmp = einsum('fbjn, efmn -> mbej', t2T_tmp, eris.xvoo[:, p0:p1])
        tmp *= (-0.5)
        Wmbej += tmp
        tmp = t2T_tmp = None

    Wmbej -= np.asarray(eris.xovo).transpose(3, 2, 0, 1)
    return Wmbej

@mpi.parallel_call(skip_args=[3], skip_kwargs=['eris'])
def energy(mycc, t1=None, t2=None, eris=None):
    '''CCSD correlation energy'''
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris

    nocc, nvir = t1.shape
    fock = eris.fock
    loc0, loc1 = _task_location(nvir)
    if rank == 0:
        e = np.einsum('ia, ia', fock[:nocc, nocc:], t1, optimize=True)
    else:
        e = 0.0
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir+1))))
    for p0, p1 in lib.prange(0, loc1-loc0, blksize):
        eris_vvoo = eris.xvoo[p0:p1]
        e += 0.25 * np.einsum('ijab, abij', t2[:, :, p0:p1], eris_vvoo, optimize=True)
        e += 0.50 * np.einsum('ia, jb, abij', t1[:, loc0+p0:loc0+p1], t1,
                              eris_vvoo, optimize=True)
    e = comm.allreduce(e)

    if rank == 0 and abs(e.imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in CCSD energy %s', e)
    return e.real

@mpi.parallel_call(skip_args=[1], skip_kwargs=['eris'])
def init_amps(mycc, eris=None):
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris

    time0 = logger.process_clock(), logger.perf_counter()
    mo_e = eris.mo_energy
    nocc = mycc.nocc
    nvir = mo_e.size - nocc
    mo_e_o = mo_e[:nocc]
    mo_e_v = mo_e[nocc:] + mycc.level_shift
    eia = mo_e_o[:, None] - mo_e_v
    t1T = eris.fock[nocc:, :nocc] / eia.T
    loc0, loc1 = _task_location(nvir)

    t2T = np.empty((loc1-loc0, nvir, nocc, nocc))
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir+1))))
    emp2 = 0
    for p0, p1 in lib.prange(0, loc1-loc0, blksize):
        eris_vvoo = eris.xvoo[p0:p1]
        t2T[p0:p1] = (eris_vvoo / lib.direct_sum('ia, jb -> abij', eia[:, loc0+p0:loc0+p1], eia))
        emp2 += np.einsum('abij, abij', t2T[p0:p1], eris_vvoo.conj(), optimize=True).real
        eris_vvoo = None

    mycc.emp2 = comm.allreduce(emp2) * 0.25
    logger.info(mycc, 'Init t2, MP2 energy = %.15g', mycc.emp2)
    logger.timer(mycc, 'init mp2', *time0)
    mycc.t1 = t1T.T
    mycc.t2 = t2T.transpose(2, 3, 0, 1)
    return mycc.emp2, mycc.t1, mycc.t2

def _init_ccsd(ccsd_obj):
    from pyscf import gto
    from mpi4pyscf.tools import mpi
    from mpi4pyscf.cc import gccsd
    if mpi.rank == 0:
        mpi.comm.bcast((ccsd_obj.mol.dumps(), ccsd_obj.pack()))
    else:
        ccsd_obj = gccsd.GCCSD.__new__(gccsd.GCCSD)
        ccsd_obj.t1 = ccsd_obj.t2 = ccsd_obj.l1 = ccsd_obj.l2 = None
        mol, cc_attr = mpi.comm.bcast(None)
        ccsd_obj.mol = gto.mole.loads(mol)
        ccsd_obj.unpack_(cc_attr)
    if True:  # If also to initialize cc._scf object
        if mpi.rank == 0:
            if hasattr(ccsd_obj._scf, '_scf'):
                # ZHC FIXME a hack, newton need special treatment to broadcast
                e_tot = ccsd_obj._scf.e_tot
                ccsd_obj._scf = ccsd_obj._scf._scf
                ccsd_obj._scf.e_tot = e_tot
            mpi.comm.bcast((ccsd_obj._scf.__class__, _pack_scf(ccsd_obj._scf)))
        else:
            mf_cls, mf_attr = mpi.comm.bcast(None)
            ccsd_obj._scf = mf_cls(ccsd_obj.mol)
            ccsd_obj._scf.__dict__.update(mf_attr)

    key = id(ccsd_obj)
    mpi._registry[key] = ccsd_obj
    regs = mpi.comm.gather(key)
    return regs

@mpi.parallel_call
def test_make_tau(mycc, t1T=None, t2T=None, fac=1, ref=None):
    """
    Make effective t2T (abij) using t2T and t1T.

    Args:
        mycc: GCCSD object.
        t1T: ai
        t2T: abij, a is segmented.
        fac: factor

    Returns:
        tauT: abij, a is segmented.
    """
    if t1T is None:
        t1T = mycc.t1.T
    if t2T is None:
        t2T = mycc.t2.transpose(2, 3, 0, 1)
    tauT = make_tauT(t1T, t2T, fac=fac)
    tau = mpi.gather_new(tauT).transpose(2, 3, 0, 1)
    if rank == 0:
        tau_diff = la.norm(tau - ref)
        assert tau_diff < 1e-10

@mpi.parallel_call
def test_cc_Fov(mycc, t1T=None, ref=None):
    """
    test Fov.
    """
    if t1T is None:
        t1T = mycc.t1.T
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris
    Fov = cc_Fov(t1T, eris)
    assert la.norm(Fov - ref) < 1e-10

@mpi.parallel_call
def test_cc_Fvv(mycc, t1T=None, t2T=None, ref=None):
    """
    test Fvv.
    """
    if t1T is None:
        t1T = mycc.t1.T
    if t2T is None:
        t2T = mycc.t2.transpose(2, 3, 0, 1)
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris
    Fvv = cc_Fvv(t1T, t2T, eris)
    assert la.norm(Fvv - ref) < 1e-10

@mpi.parallel_call
def test_cc_Foo(mycc, t1T=None, t2T=None, ref=None):
    if t1T is None:
        t1T = mycc.t1.T
    if t2T is None:
        t2T = mycc.t2.transpose(2, 3, 0, 1)
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris
    Foo = cc_Foo(t1T, t2T, eris)
    assert la.norm(Foo - ref) < 1e-10

@mpi.parallel_call
def test_cc_Woooo(mycc, t1T=None, t2T=None, ref=None):
    if t1T is None:
        t1T = mycc.t1.T
    if t2T is None:
        t2T = mycc.t2.transpose(2, 3, 0, 1)
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris
    Woooo = cc_Woooo(t1T, t2T, eris)
    assert la.norm(Woooo - ref) < 1e-10

@mpi.parallel_call
def test_cc_Wvvvv(mycc, t1T=None, t2T=None, ref=None):
    if t1T is None:
        t1T = mycc.t1.T
    if t2T is None:
        t2T = mycc.t2.transpose(2, 3, 0, 1)
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris
    Wvvvv = mpi.gather_new(cc_Wvvvv(t1T, t2T, eris))
    if rank == 0:
        assert la.norm(Wvvvv - ref) < 1e-10

@mpi.parallel_call
def test_cc_Wovvo(mycc, t1T=None, t2T=None, ref=None):
    if t1T is None:
        t1T = mycc.t1.T
    if t2T is None:
        t2T = mycc.t2.transpose(2, 3, 0, 1)
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris
    Wovvo = cc_Wovvo(t1T, t2T, eris)
    Wovvo = mpi.gather_new(Wovvo.transpose(2, 0, 1, 3)).transpose(1, 2, 0, 3)
    if rank == 0:
        assert la.norm(Wovvo - ref) < 1e-10

@mpi.parallel_call
def test_update_amps(mycc, ref):
    t1 = mycc.t1
    t2 = mycc.t2
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris

    t1new, t2new = update_amps(mycc, t1, t2, eris)
    if rank == 0:
        print ("diff: ", la.norm(t1new - ref[0]))
        assert la.norm(t1new - ref[0]) < 1e-10
    
    if t2new is not None:
        t1new, t2new = mycc.gather_amplitudes(t1new, t2new)
        if rank == 0:
            assert la.norm(t2new - ref[1]) < 1e-10

@mpi.parallel_call
def test_lambda_imds(mycc, ref):
    t1 = mycc.t1
    t2 = mycc.t2
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris
    
    imds = gccsd_lambda.make_intermediates(mycc, t1, t2, eris)
    imds.v1 = np.asarray(imds.v1)
    imds.v2 = np.asarray(imds.v2)
    imds.w3 = np.asarray(imds.w3)
    imds.woooo = np.asarray(imds.woooo)
    imds.wovvo = mpi.gather_new(np.asarray(imds.wovvo).transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3)
    imds.wovoo = mpi.gather_new(np.asarray(imds.wovoo).transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3)
    imds.wvvvo = mpi.gather_new(np.asarray(imds.wvvvo))
    return imds

@mpi.parallel_call
def test_update_lambda(mycc, ref):
    l1 = t1 = mycc.t1
    l2 = t2 = mycc.t2
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris
    
    imds = gccsd_lambda.make_intermediates(mycc, t1, t2, eris)
    l1new, l2new = gccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    
    if rank == 0:
        print ("l1 diff: ", la.norm(l1new - ref[0]))
        #assert la.norm(l1new - ref[0]) < 1e-10
    
    if l2new is not None:
        l1new, l2new = mycc.gather_amplitudes(l1new, l2new)
        if rank == 0:
            print ("l2 diff: ", la.norm(l2new - ref[1]))
            #assert la.norm(l2new - ref[1]) < 1e-10

def get_vtril(nvir, vloc, p0=None, p1=None):
    idx_tril = np.tril_indices(nvir, k=-1)
    start = np.searchsorted(idx_tril[0], vloc[0], side='left')
    end   = np.searchsorted(idx_tril[0], vloc[1] - 1, side='right')
    vtril = [idx_tril[0][start:end] - vloc[0], idx_tril[1][start:end]]
    if (p0 is not None) and (p1 is not None):
        vtril_new = [[], []]
        for (ix, iy) in zip(*vtril):
            if iy >= p0 and iy < p1:
                vtril_new[0].append(ix)
                vtril_new[1].append(iy)
        vtril_new[0] = np.asarray(vtril_new[0])
        vtril_new[1] = np.asarray(vtril_new[1])
        vtril = vtril_new
    vtril = tuple(vtril)
    return vtril

def amplitudes_to_vector(t1, t2, out=None):
    """
    amps to vector, with the same bahavior as pyscf gccsd.
    """
    t2T = np.asarray(t2.transpose(2, 3, 0, 1), order='C')
    nvir_seg, nvir, nocc = t2T.shape[:3]
    
    ntasks = mpi.pool.size
    vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vtril = get_vtril(nvir, vlocs[rank])
    otril = np.tril_indices(nocc, k=-1)
    nvir2 = len(vtril[0])
    nocc2 = nocc * (nocc - 1) // 2
    size = nvir2 * nocc2
    nov = nocc * nvir
    if rank == 0:
        size += nov
    
    if rank == 0:
        t1T = t1.T
        vector = np.ndarray(size, t1.dtype, buffer=out)
        vector[:nov] = t1T.ravel()
        lib.take_2d(t2T.reshape(-1, nocc**2), vtril[0]*nvir+vtril[1],
                    otril[0]*nocc+otril[1], out=vector[nov:])
    else:
        vector = np.ndarray(size, t1.dtype, buffer=out)
        lib.take_2d(t2T.reshape(-1, nocc**2), vtril[0]*nvir+vtril[1],
                    otril[0]*nocc+otril[1], out=vector) 
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    """
    vector to amps, with the same bahavior as pyscf gccsd.
    """
    nvir = nmo - nocc
    nov = nocc * nvir
    nocc2 = nocc * (nocc - 1) // 2
    otril = np.tril_indices(nocc, k=-1)
    
    vlocs = [_task_location(nvir, task_id) for task_id in range(mpi.pool.size)]
    vloc0, vloc1 = vlocs[rank]
    nvir_seg = vloc1 - vloc0
    vtril = get_vtril(nvir, vlocs[rank])
    nvir2 = len(vtril[0])

    if rank == 0:
        t1T = vector[:nov].copy().reshape(nvir, nocc)
        mpi.bcast(t1T)
        t2tril = vector[nov:].reshape(nvir2, nocc2)
    else:
        t1T = mpi.bcast(None)
        t2tril = vector.reshape(nvir2, nocc2)
    t2T = np.zeros((nvir_seg * nvir, nocc**2), dtype=t2tril.dtype)
    lib.takebak_2d(t2T, t2tril, vtril[0]*nvir+vtril[1], otril[0]*nocc+otril[1])
    # anti-symmetry when exchanging two particle indices
    lib.takebak_2d(t2T, -t2tril, vtril[0]*nvir+vtril[1], otril[1]*nocc+otril[0])
    t2T = t2T.reshape(nvir_seg, nvir, nocc, nocc)
    
    t2tmp = mpi.alltoall_new([t2T[:, p0:p1] for p0,p1 in vlocs], split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        if task_id < rank:
            # do not need this part since it is already filled.
            continue
        elif task_id == rank:
            # fill the trlu by -tril.
            v_idx = get_vtril(nvir, vlocs[task_id], p0=p0, p1=p1)
            tmp = t2tmp[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
            t2T[v_idx[1]-p0, v_idx[0]+p0] = tmp[v_idx[0], v_idx[1]-p0].transpose(0, 2, 1)
        else:
            tmp = t2tmp[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
            t2T[:, p0:p1] = tmp.transpose(1, 0, 3, 2)

    return t1T.T, t2T.transpose(2, 3, 0, 1)

@mpi.parallel_call
def save_amps(mycc, fname="fcc"):
    """
    Save amplitudes to a file.
    """
    _sync_(mycc)
    if fname.endswith(".h5"):
        fname = fname[:-3]
    filename = fname + '__rank' + str(mpi.rank) + ".h5"
    fcc = lib.H5TmpFile(filename, 'w')
    fcc['mo_coeff'] = mycc.mo_coeff
    if (getattr(mycc, "t1", None) is not None) and \
       (getattr(mycc, "t2", None) is not None):
        tvec = mycc.amplitudes_to_vector(mycc.t1, mycc.t2)
        fcc['tvec'] = tvec
    tvec = None
    if (getattr(mycc, "l1", None) is not None) and \
       (getattr(mycc, "l2", None) is not None):
        lvec = mycc.amplitudes_to_vector(mycc.l1, mycc.l2)
        fcc['lvec'] = lvec
    lvec = None

@mpi.parallel_call
def load_amps(mycc, fname="fcc", only_t=False):
    """
    Load amplitudes from a file.
    """
    _sync_(mycc)
    if fname.endswith(".h5"):
        fname = fname[:-3]
    filename = fname + '__rank' + str(mpi.rank) + ".h5"
    fcc = h5py.File(filename, 'r')
    keys = fcc.keys()
    mo_coeff = np.asarray(fcc["mo_coeff"])
    if 'tvec' in keys:
        tvec = np.asarray(fcc["tvec"])
        t1, t2 = mycc.vector_to_amplitudes(tvec)
        tvec = None
    else:
        t1 = t2 = None
    
    if only_t:
        l1 = l2 = None
    else:
        if 'lvec' in keys:
            lvec = np.asarray(fcc["lvec"])
            l1, l2 = mycc.vector_to_amplitudes(lvec)
            lvec = None
        else:
            l1 = l2 = None
    fcc.close()
    return t1, t2, l1, l2, mo_coeff

@mpi.parallel_call
def restore_from_h5(mycc, fname="fcc", umat=None, only_t=False):
    """
    Restore t1, t2, l1, l2 from file.
    
    Args:
        mycc: CC object.
        fname: prefix for the filename.
        umat: (nmo, nmo), rotation matrix to rotate amps.
        only_t: if yes, will only load t1, t2; set l1, l2 to None.

    Return:
        mycc: CC object, with t1, t2, l1, l2 updated.
    """
    _sync_(mycc)
    if fname.endswith(".h5"):
        fname = fname[:-3]
    filename = fname + '__rank' + str(rank) + ".h5"
    logger.info(mycc, "restore amps from %s (rank 0-%s) ...",
                filename, mpi.pool.size)
    if all(comm.allgather(os.path.isfile(filename))):
        t1, t2, l1, l2, mo_coeff = mycc.load_amps(fname=fname, only_t=only_t)
        if umat is not None:
            logger.info(mycc, "rotate amps ...")
            nocc, nvir = t1.shape
            ntasks = mpi.pool.size
            vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
            
            t1 = transform_t1_to_bo(t1, umat)
            t2 = transform_t2_to_bo(t2, umat, vlocs=vlocs)
            
            if l1 is not None:
                l1 = transform_l1_to_bo(l1, umat)
            if l2 is not None:
                l2 = transform_l2_to_bo(l2, umat, vlocs=vlocs)
            
        mycc.t1 = t1
        mycc.t2 = t2
        mycc.l1 = l1
        mycc.l2 = l2
    else:
        raise ValueError("restore_from_h5 failed, (part of) files %s not exist."%filename)
    return mycc

def transform_t1_to_bo(t1, u):
    """
    transform t1.
    """
    t1T = t1.T
    nvir, nocc = t1T.shape
    u_oo = u[:nocc, :nocc]
    u_vv = u[nocc:, nocc:]
    t1T = reduce(np.dot, (u_vv.conj().T, t1T, u_oo))
    return t1T.T

transform_l1_to_bo = transform_t1_to_bo

def transform_t2_to_bo(t2, u, vlocs=None):
    """
    transform t2.
    """
    t2T = t2.transpose(2, 3, 0, 1)
    nvir_seg, nvir, nocc = t2T.shape[:3]
    if vlocs is None:
        ntasks = mpi.pool.size
        vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vloc0, vloc1 = vlocs[rank]
    assert vloc1 - vloc0 == nvir_seg
    
    u_oo = u[:nocc, :nocc]
    u_vv = u[nocc:, nocc:] 
            
    t2Tnew = 0.0
    for task_id, t2T_tmp, p0, p1 in _rotate_vir_block(t2T, vlocs=vlocs):
        t2Tnew += np.einsum('aA, abij -> Abij', u_vv[p0:p1, vloc0:vloc1],
                            t2T_tmp, optimize=True)
        t2T_tmp = None
    t2 = t2T = None

    t2Tnew = np.einsum("Abij, bB, iI, jJ -> ABIJ", t2Tnew, u_vv,
                       u_oo, u_oo, optimize=True)
    t2 = t2Tnew.transpose(2, 3, 0, 1)
    return t2

transform_l2_to_bo = transform_t2_to_bo

@mpi.parallel_call
def _release_regs(mycc, remove_h2=False):
    comm.Barrier()
    pairs = list(mpi._registry.items())
    for key, val in pairs:
        if isinstance(val, (GCCSD, GGCCSD)):
            if remove_h2:
                mpi._registry[key]._scf = None
            else:
                del mpi._registry[key]
    gc.collect()
    comm.Barrier()
    pairs = list(mpi._registry.items())
    if not remove_h2:
        mycc._reg_procs = []

class GCCSD(gccsd.GCCSD):
    """
    MPI GCCSD.
    """
    conv_tol = getattr(__config__, 'cc_gccsd_GCCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_gccsd_GCCSD_conv_tol_normt', 1e-6)

    # ************************************************************************
    # Initialization
    # ************************************************************************

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, 
                 remove_h2=False):
        assert isinstance(mf, scf.ghf.GHF)
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.remove_h2 = remove_h2
        regs = mpi.pool.apply(_init_ccsd, (self,), (None,))
        self._reg_procs = regs

    def pack(self):
        return {'verbose'    : self.verbose,
                'max_memory' : self.max_memory,
                'frozen'     : self.frozen,
                'mo_coeff'   : self.mo_coeff,
                'mo_occ'     : self.mo_occ,
                '_nocc'      : self._nocc,
                '_nmo'       : self._nmo,
                'diis_file'  : self.diis_file,
                'diis_start_cycle' : self.diis_start_cycle,
                'level_shift': self.level_shift,
                'direct'     : self.direct,
                'diis_space' : self.diis_space}

    def unpack_(self, ccdic):
        self.__dict__.update(ccdic)
        return self

    def dump_flags(self, verbose=None):
        if rank == 0:
            gccsd.GCCSD.dump_flags(self, verbose)
            logger.info(self, 'level_shift = %.9g', self.level_shift)
            logger.info(self, 'nproc       = %4d', mpi.pool.size)
        return self

    def sanity_check(self):
        if rank == 0:
            gccsd.GCCSD.sanity_check(self)
        return self

    # ************************************************************************
    # core functions
    # ************************************************************************

    init_amps = init_amps
    energy = energy
    update_amps = update_amps

    def ao2mo(self, mo_coeff=None):
        _make_eris_incore(self, mo_coeff)
        return 'Done'
    
    def ao2mo_ghf(self, mo_coeff=None):
        _make_eris_incore_ghf(self, mo_coeff)
        return 'Done'

    def ccsd(self, t1=None, t2=None, eris=None):
        assert self.mo_coeff is not None
        assert self.mo_occ is not None
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        
        if self.e_hf is None:
            self.e_hf = self._scf.e_tot

        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        if rank == 0:
            self._finalize()
        return self.e_corr, self.t1, self.t2

    def kernel(self, t1=None, t2=None, eris=None):
        return self.ccsd(t1, t2, eris)
    
    def _finalize(self):
        """
        Hook for dumping results and clearing up the object.
        """
        rccsd.CCSD._finalize(self)
        # ZHC NOTE unregister the ccsd_obj
        # self._release_regs()
        return self

    _release_regs = _release_regs

    def run_diis(self, t1, t2, istep, normt, de, adiis):
        if (adiis and
            istep >= self.diis_start_cycle and
            abs(de) < self.diis_start_energy_diff):
            vec = self.amplitudes_to_vector(t1, t2)
            t1, t2 = self.vector_to_amplitudes(adiis.update(vec))
            logger.debug1(self, 'DIIS for step %d', istep)
        return t1, t2

    # ************************************************************************
    # Conversion of amplitudes
    # ************************************************************************

    distribute_amplitudes_ = distribute_amplitudes_
    gather_amplitudes = gather_amplitudes
    gather_lambda = gather_lambda

    def amplitudes_to_vector(self, t1, t2, out=None):
        return amplitudes_to_vector(t1, t2, out)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vec, nmo, nocc)

    restore_from_diis_ = restore_from_diis_
    
    restore_from_h5 = restore_from_h5
    
    save_amps = save_amps
    
    load_amps = load_amps
    
    def vector_size(self, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        ntasks = mpi.pool.size
        vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
        vtril = get_vtril(nvir, vlocs[rank])
        nvir2 = len(vtril[0])
        nocc2 = nocc * (nocc - 1) // 2
        size = nvir2 * nocc2
        if rank == 0:
            size += nocc * nvir
        return size

    # ************************************************************************
    # Lambda, rdm, ip ea
    # ************************************************************************

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None, approx_l=False):
        self.converged_lambda, self.l1, self.l2 = \
                gccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                    max_cycle=self.max_cycle,
                                    tol=self.conv_tol_normt,
                                    verbose=self.verbose, approx_l=approx_l)
        return self.l1, self.l2

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''Un-relaxed 1-particle density matrix in MO space'''
        return gccsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr)
    
    def make_rdm1_ref(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''Un-relaxed 1-particle density matrix in MO space'''
        return gccsd_rdm.make_rdm1_ref(self, t1, t2, l1, l2, ao_repr=ao_repr)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        return gccsd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr)
    
    #def ccsd_t(self, t1=None, t2=None, eris=None):
    #    from pyscf.cc import gccsd_t
    #    if t1 is None: t1 = self.t1
    #    if t2 is None: t2 = self.t2
    #    if eris is None: eris = self.ao2mo(self.mo_coeff)
    #    return gccsd_t.kernel(self, eris, t1, t2, self.verbose)

    #def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
    #           partition=None, eris=None):
    #    from pyscf.cc import eom_gccsd
    #    return eom_gccsd.EOMIP(self).kernel(nroots, left, koopmans, guess,
    #                                        partition, eris)

    #def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
    #           partition=None, eris=None):
    #    from pyscf.cc import eom_gccsd
    #    return eom_gccsd.EOMEA(self).kernel(nroots, left, koopmans, guess,
    #                                        partition, eris)

    #def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
    #    from pyscf.cc import eom_gccsd
    #    return eom_gccsd.EOMEE(self).kernel(nroots, koopmans, guess, eris)

    #def eomip_method(self):
    #    from pyscf.cc import eom_gccsd
    #    return eom_gccsd.EOMIP(self)

    #def eomea_method(self):
    #    from pyscf.cc import eom_gccsd
    #    return eom_gccsd.EOMEA(self)

    #def eomee_method(self):
    #    from pyscf.cc import eom_gccsd
    #    return eom_gccsd.EOMEE(self)
    
    # ************************************************************************
    # test functions
    # ************************************************************************
    test_make_tau = test_make_tau    
    test_cc_Foo = test_cc_Foo
    test_cc_Fov = test_cc_Fov
    test_cc_Fvv = test_cc_Fvv    
    test_cc_Woooo = test_cc_Woooo
    test_cc_Wvvvv = test_cc_Wvvvv    
    test_cc_Wovvo = test_cc_Wovvo    
    test_update_amps = test_update_amps
    test_update_lambda = test_update_lambda
    test_lambda_imds = test_lambda_imds

CCSD = GCCSD

def _init_ggccsd(ccsd_obj):
    from pyscf import gto
    from mpi4pyscf.tools import mpi
    from mpi4pyscf.cc import gccsd
    if mpi.rank == 0:
        mpi.comm.bcast((ccsd_obj.mol.dumps(), ccsd_obj.pack()))
    else:
        ccsd_obj = gccsd.GGCCSD.__new__(gccsd.GGCCSD)
        ccsd_obj.t1 = ccsd_obj.t2 = ccsd_obj.l1 = ccsd_obj.l2 = None
        mol, cc_attr = mpi.comm.bcast(None)
        ccsd_obj.mol = gto.mole.loads(mol)
        ccsd_obj.unpack_(cc_attr)
    if False:  # If also to initialize cc._scf object
        if mpi.rank == 0:
            if hasattr(ccsd_obj._scf, '_scf'):
                # ZHC FIXME a hack, newton need special treatment to broadcast
                e_tot = ccsd_obj._scf.e_tot
                ccsd_obj._scf = ccsd_obj._scf._scf
                ccsd_obj._scf.e_tot = e_tot
            mpi.comm.bcast((ccsd_obj._scf.__class__, _pack_scf(ccsd_obj._scf)))
        else:
            mf_cls, mf_attr = mpi.comm.bcast(None)
            ccsd_obj._scf = mf_cls(ccsd_obj.mol)
            ccsd_obj._scf.__dict__.update(mf_attr)

    key = id(ccsd_obj)
    mpi._registry[key] = ccsd_obj
    regs = mpi.comm.gather(key)
    return regs

class GGCCSD(GCCSD):
    """
    MPI GGCCSD.
    """
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 remove_h2=False, save_mem=False, dt=None, 
                 ignore_level_shift=True, rk=False, rk_order=4):
        assert isinstance(mf, scf.ghf.GHF)
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.remove_h2 = remove_h2
        self.save_mem = save_mem
        self.dt = dt
        self.ignore_level_shift = ignore_level_shift
        self.rk = rk
        self.rk_order = rk_order
        self._keys = self._keys.union(["remove_h2", "save_mem", "dt", "ignore_level_shift",
                                       "rk", "rk_order"])

        regs = mpi.pool.apply(_init_ggccsd, (self,), (None,))
        self._reg_procs = regs

    def ao2mo(self, mo_coeff=None):
        _make_eris_incore_ghf(self, mo_coeff)
        return 'Done'
    
    def pack(self):
        return {'verbose'    : self.verbose,
                'max_memory' : self.max_memory,
                'frozen'     : self.frozen,
                'mo_coeff'   : self.mo_coeff,
                'mo_occ'     : self.mo_occ,
                '_nocc'      : self._nocc,
                '_nmo'       : self._nmo,
                'diis_file'  : self.diis_file,
                'level_shift': self.level_shift,
                'direct'     : self.direct,
                'diis_space' : self.diis_space,
                'diis_start_cycle' : self.diis_start_cycle,
                'remove_h2'  : self.remove_h2,
                'save_mem'   : self.save_mem,
                'dt'         : self.dt,
                'ignore_level_shift': self.ignore_level_shift,
                'rk'         : self.rk,
                'rk_order'   : self.rk_order
                }

# ************************************************************************
# ITE routines
# ************************************************************************

def update_amps_rk(mycc, t1, t2, eris, fupdate=None):
    """
    Update amplitudes using RK4.
    """
    if fupdate is None:
        fupdate = update_amps
    
    h = mycc.dt
    dt11, dt21 = fupdate(mycc, t1, t2, eris)
    dt1new = dt11
    dt2new = dt21
    if mycc.rk_order != 1:
        t1_, t2_ = t1 + dt11 * (h*0.5), t2 + dt21 * (h*0.5)
        dt12, dt22 = fupdate(mycc, t1_, t2_, eris)
        t1_, t2_ = t1 + dt12 * (h*0.5), t2 + dt22 * (h*0.5)
        dt1new += (2.0 * dt12)
        dt2new += (2.0 * dt22)
        dt12 = dt22 = None
        
        dt13, dt23 = fupdate(mycc, t1_, t2_, eris)
        t1_, t2_ = t1 + dt13 * h, t2 + dt23 * h
        dt1new += (2.0 * dt13)
        dt2new += (2.0 * dt23)
        dt13 = dt23 = None
        
        dt14, dt24 = fupdate(mycc, t1_, t2_, eris)
        t1_ = t2_ = None
        dt1new += dt14
        dt2new += dt24
        dt14 = dt24 = None
        
        dt1new *= (-h / 6.0)
        dt2new *= (-h / 6.0)
    else:
        dt1new *= (-h)
        dt2new *= (-h)

    dt1new += t1
    dt2new += t2

    return dt1new, dt2new

def _init_ggccsd_ite_rk(ccsd_obj):
    from pyscf import gto
    from mpi4pyscf.tools import mpi
    from mpi4pyscf.cc import gccsd
    if mpi.rank == 0:
        mpi.comm.bcast((ccsd_obj.mol.dumps(), ccsd_obj.pack()))
    else:
        ccsd_obj = gccsd.GGCCSDITE_RK.__new__(gccsd.GGCCSDITE_RK)
        ccsd_obj.t1 = ccsd_obj.t2 = ccsd_obj.l1 = ccsd_obj.l2 = None
        mol, cc_attr = mpi.comm.bcast(None)
        ccsd_obj.mol = gto.mole.loads(mol)
        ccsd_obj.unpack_(cc_attr)

    if False:  # If also to initialize cc._scf object
        if mpi.rank == 0:
            if hasattr(ccsd_obj._scf, '_scf'):
                # ZHC FIXME a hack, newton need special treatment to broadcast
                e_tot = ccsd_obj._scf.e_tot
                ccsd_obj._scf = ccsd_obj._scf._scf
                ccsd_obj._scf.e_tot = e_tot
            mpi.comm.bcast((ccsd_obj._scf.__class__, _pack_scf(ccsd_obj._scf)))
        else:
            mf_cls, mf_attr = mpi.comm.bcast(None)
            ccsd_obj._scf = mf_cls(ccsd_obj.mol)
            ccsd_obj._scf.__dict__.update(mf_attr)

    key = id(ccsd_obj)
    mpi._registry[key] = ccsd_obj
    regs = mpi.comm.gather(key)
    return regs

class GGCCSDITE_RK(GGCCSD):
    """
    MPI GGCCSD with ITE using RK.
    """
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 remove_h2=False, save_mem=False, dt=0.1, 
                 ignore_level_shift=True, rk=True, rk_order=4,
                 diis_start_cycle=999999):
        assert isinstance(mf, scf.ghf.GHF)
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.remove_h2 = remove_h2
        self.save_mem = save_mem
        self.dt = dt
        self.ignore_level_shift = ignore_level_shift
        self.rk = rk
        self.rk_order = rk_order
        self.diis_start_cycle = diis_start_cycle
        self._keys = self._keys.union(["remove_h2", "save_mem", "dt", "ignore_level_shift",
                                       "rk", "rk_order"])

        regs = mpi.pool.apply(_init_ggccsd_ite_rk, (self,), (None,))
        self._reg_procs = regs
    
    def dump_flags(self, verbose=None):
        if rank == 0:
            GGCCSD.dump_flags(self, verbose)
            logger.info(self, 'ITE dt = %.5g', self.dt)
            logger.info(self, 'rk order = %s', self.rk_order)
        return self

    update_amps = update_amps_rk

# ************************************************************************
# ao2mo routines
# ************************************************************************

@mpi.parallel_call
def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    """
    Make physist eri with incore ao2mo.
    """
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    _sync_(mycc)
    eris = gccsd._PhysicistsERIs()
    
    if rank == 0:
        eris._common_init_(mycc, mo_coeff)
        comm.bcast((eris.mo_coeff, eris.fock, eris.nocc, eris.mo_energy))
    else:
        eris.mol = mycc.mol
        eris.mo_coeff, eris.fock, eris.nocc, eris.mo_energy = comm.bcast(None)
    
    # if workers does not have _eri, bcast from root
    if comm.allreduce(mycc._scf._eri is None, op=mpi.MPI.LOR):
        if rank == 0:
            mpi.bcast(mycc._scf._eri)
        else:
            mycc._scf._eri = mpi.bcast(None)
    cput1 = log.timer('CCSD ao2mo initialization:     ', *cput0)

    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc
    vlocs = [_task_location(nvir, task_id) for task_id in range(mpi.pool.size)]
    vloc0, vloc1 = vlocs[rank]
    vseg = vloc1 - vloc0
    
    plocs = [_task_location(nmo, task_id) for task_id in range(mpi.pool.size)]
    ploc0, ploc1 = plocs[rank]
    pseg = ploc1 - ploc0
    
    mo_a = eris.mo_coeff[:nao//2]
    mo_b = eris.mo_coeff[nao//2:]
    mo_seg_a = mo_a[:, ploc0:ploc1]
    mo_seg_b = mo_b[:, ploc0:ploc1]
    
    fname = "gccsd_eri_tmp_%s.h5"%rank
    f = h5py.File(fname, 'w')
    eri_phys = f.create_dataset('eri_phys', (pseg, nmo, nmo, nmo), 'f8', 
                                chunks=(pseg, 1, nmo, nmo))
    
    eri_a = ao2mo.incore.half_e1(mycc._scf._eri, (mo_seg_a, mo_a), compact=False)
    eri_b = ao2mo.incore.half_e1(mycc._scf._eri, (mo_seg_b, mo_b), compact=False)
    cput1 = log.timer('CCSD ao2mo half_e1:            ', *cput1)

    unit = pseg * nmo * nmo * 2
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blksize = min(nmo, max(BLKMIN, int((max_memory*0.9e6/8)/unit)))

    for p0, p1 in lib.prange(0, nmo, blksize):
        klmosym_a, nkl_pair_a, mokl_a, klshape_a = \
                ao2mo.incore._conc_mos(mo_a[:, p0:p1], mo_a, compact=False)
        klmosym_b, nkl_pair_b, mokl_b, klshape_b = \
                ao2mo.incore._conc_mos(mo_b[:, p0:p1], mo_b, compact=False)
        
        eri  = _ao2mo.nr_e2(eri_a, mokl_a, klshape_a, aosym='s4', mosym=klmosym_a)
        eri += _ao2mo.nr_e2(eri_a, mokl_b, klshape_b, aosym='s4', mosym=klmosym_b)
        eri += _ao2mo.nr_e2(eri_b, mokl_a, klshape_a, aosym='s4', mosym=klmosym_a)
        eri += _ao2mo.nr_e2(eri_b, mokl_b, klshape_b, aosym='s4', mosym=klmosym_b)
        
        eri = eri.reshape(pseg, nmo, p1-p0, nmo)
        eri_phys[:, p0:p1] = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
        eri = None
    eri_a = None
    eri_b = None
    
    f.close()
    comm.Barrier()
    cput1 = log.timer('CCSD ao2mo nr_e2:              ', *cput1)

    o_idx = -1
    v_idx = mpi.pool.size
    for r, (p0, p1) in enumerate(plocs):
        if p0 <= nocc - 1 < p1:
            o_idx = r
        if p0 <= nocc < p1:
            v_idx = r
            break
    o_files = np.arange(mpi.pool.size)[:(o_idx+1)]
    v_files = np.arange(mpi.pool.size)[v_idx:]

    eris.oooo = np.empty((nocc, nocc, nocc, nocc))
    eris.xooo = np.empty((vseg, nocc, nocc, nocc))
    eris.xovo = np.empty((vseg, nocc, nvir, nocc))
    eris.xovv = np.empty((vseg, nocc, nvir, nvir))
    eris.xvvo = np.empty((vseg, nvir, nvir, nocc))
    eris.xvoo = np.empty((vseg, nvir, nocc, nocc))
    eris.xvvv = np.empty((vseg, nvir, nvir, nvir))
    for r in range(mpi.pool.size):
        f = lib.H5TmpFile(filename="gccsd_eri_tmp_%s.h5"%r, mode='r')
        eri_phys = f["eri_phys"]
        if r in o_files:
            p0, p1 = plocs[r]
            p1 = min(p1, nocc)
            pseg = p1 - p0
            if pseg > 0:
                eris.oooo[p0:p1] = eri_phys[:pseg, :nocc, :nocc, :nocc]
        
        if r in v_files:
            p00, p10 = plocs[r]
            p0 = max(p00, nocc+vloc0)
            p1 = min(p10, nocc+vloc1)
            pseg = p1 - p0
            if pseg > 0:
                eris.xooo[p0-(nocc+vloc0):p1-(nocc+vloc0)] = eri_phys[p0-p00:p1-p00, :nocc, :nocc, :nocc]
                eris.xovo[p0-(nocc+vloc0):p1-(nocc+vloc0)] = eri_phys[p0-p00:p1-p00, :nocc, nocc:, :nocc]
                eris.xvoo[p0-(nocc+vloc0):p1-(nocc+vloc0)] = eri_phys[p0-p00:p1-p00, nocc:, :nocc, :nocc]
                eris.xvvo[p0-(nocc+vloc0):p1-(nocc+vloc0)] = eri_phys[p0-p00:p1-p00, nocc:, nocc:, :nocc]
                eris.xovv[p0-(nocc+vloc0):p1-(nocc+vloc0)] = eri_phys[p0-p00:p1-p00, :nocc, nocc:, nocc:]
                eris.xvvv[p0-(nocc+vloc0):p1-(nocc+vloc0)] = eri_phys[p0-p00:p1-p00, nocc:, nocc:, nocc:]
    cput1 = log.timer('CCSD ao2mo load:               ', *cput1)

    f.close() 
    comm.Barrier()
    os.remove("gccsd_eri_tmp_%s.h5"%rank)
    mycc._eris = eris
    log.timer('CCSD integral transformation   ', *cput0)
    return eris

@mpi.parallel_call
def _make_eris_incore_ghf(mycc, mo_coeff=None, ao2mofn=None):
    """
    Make physist eri with incore ao2mo, for GGHF.
    """
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    _sync_(mycc)
    eris = gccsd._PhysicistsERIs()
    
    if rank == 0:
        eris._common_init_(mycc, mo_coeff)
        comm.bcast((eris.mo_coeff, eris.fock, eris.nocc, eris.mo_energy))
    else:
        eris.mol = mycc.mol
        eris.mo_coeff, eris.fock, eris.nocc, eris.mo_energy = comm.bcast(None)
    
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape

    nvir = nmo - nocc
    vlocs = [_task_location(nvir, task_id) for task_id in range(mpi.pool.size)]
    vloc0, vloc1 = vlocs[rank]
    vseg = vloc1 - vloc0
    
    if rank == 0:
        if callable(ao2mofn):
            raise NotImplementedError
        else:
            assert eris.mo_coeff.dtype == np.double
            eri = mycc._scf._eri
            if (nao == nmo) and (la.norm(eris.mo_coeff - np.eye(nmo)) < 1e-12):
                # ZHC NOTE special treatment for OO-CCD,
                # where the ao2mo is not needed for identity mo_coeff.
                from libdmet.utils import take_eri as fn
                o = np.arange(0, nocc)
                v = np.arange(nocc, nmo)
                if eri.size == nmo**4:
                    eri = ao2mo.restore(8, eri, nmo)
            else:
                if mycc.save_mem:
                    # ZHC NOTE the following is slower, although may save some memory.
                    def fn(x, mo0, mo1, mo2, mo3):
                        return ao2mo.general(x, (mo0, mo1, mo2, mo3),
                                             compact=False).reshape(mo0.shape[-1], mo1.shape[-1],
                                                                    mo2.shape[-1], mo3.shape[-1])
                    o = eris.mo_coeff[:, :nocc]
                    v = eris.mo_coeff[:, nocc:]
                    if eri.size == nao**4:
                        eri = ao2mo.restore(8, eri, nao)
                else:
                    from libdmet.utils import take_eri as fn
                    o = np.arange(0, nocc)
                    v = np.arange(nocc, nmo)
                    if mycc.remove_h2:
                        mycc._scf._eri = None
                        _release_regs(mycc, remove_h2=True)
                    eri = ao2mo.kernel(eri, eris.mo_coeff)
                    if eri.size == nmo**4:
                        eri = ao2mo.restore(8, eri, nmo)

    comm.Barrier()
    cput2 = log.timer('CCSD ao2mo initialization:     ', *cput0)
    
    # chunck and scatter:
    
    # 1. oooo
    if rank == 0:
        tmp = fn(eri, o, o, o, o)
        eris.oooo = tmp.transpose(0, 2, 1, 3) - tmp.transpose(0, 2, 3, 1)
        tmp = None
        mpi.bcast(eris.oooo)
    else:
        eris.oooo = mpi.bcast(None)
    cput3 = log.timer('CCSD bcast   oooo:              ', *cput2)
    
    # 2. xooo
    if rank == 0:
        tmp = fn(eri, v, o, o, o)
        eri_sliced = [tmp[p0:p1] for (p0, p1) in vlocs]
    else:
        tmp = None
        eri_sliced = None
    tmp = mpi.scatter_new(eri_sliced, root=0, data=tmp)
    eri_sliced = None
    eris.xooo = tmp.transpose(0, 2, 1, 3) - tmp.transpose(0, 2, 3, 1)
    tmp = None
    cput4 = log.timer('CCSD scatter xooo:              ', *cput3)
    
    # 3. xovo
    if rank == 0:
        tmp_vvoo = fn(eri, v, v, o, o)
        tmp_voov = fn(eri, v, o, o, v)
        # ZHC NOTE need to keep tmp_voov for xvoo
        eri_1 = [tmp_vvoo[p0:p1] for (p0, p1) in vlocs]
        eri_2 = [tmp_voov[p0:p1] for (p0, p1) in vlocs]
    else:
        tmp_vvoo = None
        tmp_voov = None
        eri_1 = None
        eri_2 = None

    tmp_1 = mpi.scatter_new(eri_1, root=0, data=tmp_vvoo)
    eri_1 = None
    tmp_vvoo = None
    
    tmp_2 = mpi.scatter_new(eri_2, root=0, data=tmp_voov)
    eri_2 = None
    tmp_voov = None
    
    eris.xovo = tmp_1.transpose(0, 2, 1, 3) - tmp_2.transpose(0, 2, 3, 1)
    tmp_1 = None
    cput5 = log.timer('CCSD scatter xovo:              ', *cput4)
    
    # 4. xvoo
    eris.xvoo = tmp_2.transpose(0, 3, 1, 2) - tmp_2.transpose(0, 3, 2, 1)
    tmp_2 = None
    cput6 = log.timer('CCSD scatter xvoo:              ', *cput5)
    
    # 5. 6. xovv, xvvo
    if rank == 0:
        tmp = fn(eri, v, v, o, v)
        eri_sliced = [tmp[p0:p1] for (p0, p1) in vlocs]
    else:
        tmp = None
        eri_sliced = None
    tmp_1 = mpi.scatter_new(eri_sliced, root=0, data=tmp)
    eri_sliced = None
    eris.xovv = tmp_1.transpose(0, 2, 1, 3) - tmp_1.transpose(0, 2, 3, 1)

    if rank == 0:
        tmp_2 = np.asarray(tmp.transpose(3, 2, 1, 0), order='C') # vovv
        tmp = None
        eri_sliced = [tmp_2[p0:p1] for (p0, p1) in vlocs]
    else:
        tmp_2 = None
        tmp = None
        eri_sliced = None
    tmp_2 = mpi.scatter_new(eri_sliced, root=0, data=tmp_2)
    eri_sliced = None
    
    eris.xvvo = tmp_1.transpose(0, 3, 1, 2) - tmp_2.transpose(0, 2, 3, 1)
    tmp_1 = None
    tmp_2 = None
    cput7 = log.timer('CCSD scatter xovv, xvvo:        ', *cput6)

    # 7. xvvv
    if rank == 0:
        tmp = fn(eri, v, v, v, v)
        if mycc.remove_h2:
            eri = None
            if mycc._scf is not None:
                mycc._scf._eri = None
        eri_sliced = [tmp[p0:p1] for (p0, p1) in vlocs]
    else:
        tmp = None
        eri_sliced = None
    tmp = mpi.scatter_new(eri_sliced, root=0, data=tmp)
    eri_sliced = None
    eris.xvvv = tmp.transpose(0, 2, 1, 3) - tmp.transpose(0, 2, 3, 1)
    tmp = None
    eri = None
    cput8 = log.timer('CCSD scatter xvvv:              ', *cput7)
    
    mycc._eris = eris
    log.timer('CCSD integral transformation   ', *cput0)
    return eris

if __name__ == '__main__':
    from pyscf import gto
    import time
    
    np.random.seed(1)
    np.set_printoptions(4, linewidth=1000, suppress=True)
    
    # genrate reference values
    mol = gto.M(
        atom = 'H 0 0 0; F 0 0 1.1',
        basis = '321g',
        spin = 2)
    mol.verbose = 5
    mf = mol.GHF()
    hcoreX = mf.get_hcore()
    # a small random potential to break the Sz symmetry:
    pot = (np.random.random(hcoreX.shape) - 0.5) * 3e-2
    pot = pot + pot.T
    hcoreX += pot
    mf.get_hcore = lambda *args: hcoreX
    mf.kernel()

    mycc_ref = gccsd.GCCSD(mf)
    mycc_ref.conv_tol = 1e-10
    mycc_ref.conv_tol_normt = 1e-6
    eris_ref = mycc_ref.ao2mo()

    # converged Ecc
    ecc_ref, t1_cc_ref, t2_cc_ref = mycc_ref.kernel()

    # test class
    mycc = GCCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-6

    print ("test CC converged energy and t1, t2")
    
    e_cc, t1_cc, t2_cc = mycc.kernel()
    print (abs(e_cc - ecc_ref))
    assert abs(e_cc - ecc_ref) < 1e-8
    
    t1_cc, t2_cc = mycc.gather_amplitudes()
    t1_diff = la.norm(t1_cc - t1_cc_ref)
    print (t1_diff)
    assert t1_diff < 1e-7

    if rank == 0:
        t2_diff = la.norm(t2_cc - t2_cc_ref)
        print (t2_diff)
        assert t2_diff < 1e-7

