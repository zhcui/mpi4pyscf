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
#

"""
MPI-GCCSD with real intergals.

Usage: mpirun -np 2 python gccsd.py
"""

import os
import time
import h5py
import numpy as np
import scipy.linalg as la

from pyscf import lib
from pyscf import ao2mo
from pyscf import scf
from pyscf.cc import gccsd
from pyscf import __config__

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.ccsd import (kernel, _task_location, _sync_,
                               _pack_scf, _diff_norm, _rotate_vir_block,
                               amplitudes_to_vector, vector_to_amplitudes,
                               distribute_amplitudes_, gather_amplitudes,
                               restore_from_diis_)
from mpi4pyscf.cc import gccsd_lambda
from mpi4pyscf.cc import gccsd_rdm

comm = mpi.comm
rank = mpi.rank

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

@profile
def update_amps(mycc, t1, t2, eris):
    """
    Update GCCSD amplitudes.
    """
    time1 = time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    cpu1 = time0

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

    # Move energy terms to the other side
    Fvv[np.diag_indices(nvir)] -= mo_e_v
    Foo[np.diag_indices(nocc)] -= mo_e_o

    # T1 equation
    t1Tnew  = np.dot(Fvv, t1T)
    t1Tnew -= np.dot(t1T, Foo)

    tmp  = lib.einsum('aeim, me -> ai', t2T, Fov)
    tmp -= lib.einsum('fn, naif -> ai', t1T, eris["oxov"])
    tmp  = mpi.allgather(tmp)

    tmp2  = lib.einsum('eamn, iemn -> ai', t2T, eris["oxoo"])
    tmp2 += lib.einsum('efim, efam -> ai', t2T, eris["xvvo"])
    tmp2 *= 0.5
    tmp2  = mpi.allreduce(tmp2)
    tmp += tmp2
    tmp2  = None

    t1Tnew += tmp
    t1Tnew += fvo

    # T2 equation
    Ftmp = Fvv - 0.5 * np.dot(t1T, Fov)
    t2Tnew = lib.einsum('aeij, be -> abij', t2T, Ftmp)
    t2T_tmp = mpi.alltoall([t2Tnew[:, p0:p1] for p0, p1 in vlocs],
                           split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = t2T_tmp[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        t2Tnew[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None
    t2T_tmp = None

    Ftmp = Foo + 0.5 * np.dot(Fov, t1T)
    tmp = lib.einsum('abim, mj -> abij', t2T, Ftmp)
    t2Tnew -= tmp
    t2Tnew += tmp.transpose(0, 1, 3, 2)
    tmp = None
    
    t2Tnew += np.asarray(eris["xvoo"])
    tauT = make_tauT(t1T, t2T, vlocs=vlocs)
    Woooo = cc_Woooo(t1T, t2T, eris, tauT=tauT, vlocs=vlocs)
    t2Tnew += lib.einsum('abmn, mnij -> abij', tauT, Woooo * 0.5)
    Woooo = None

    Wvvvv = cc_Wvvvv(t1T, t2T, eris, tauT=tauT, vlocs=vlocs, full_vvvv=True)
    for task_id, tauT_tmp, p0, p1 in _rotate_vir_block(tauT, vlocs=vlocs):
        t2Tnew += 0.5 * lib.einsum('abef, efij -> abij', Wvvvv[:, :, p0:p1], tauT_tmp)
        tauT_tmp = None
    Wvvvv = None
    tauT = None

    #tmp = lib.einsum('mbje, ei -> bmij', eris["oxov"], t1T) # [b]mij
    tmp = lib.einsum('bmje, ei -> bmij', eris["xoov"], -t1T) # [b]mij
    tmp = mpi.allgather(tmp) # bmij
    tmp = lib.einsum('am, bmij -> abij', t1T[vloc0:vloc1], tmp) # [a]bij

    Wvovo = cc_Wovvo(t1T, t2T, eris, vlocs=vlocs).transpose(2, 0, 1, 3)
    for task_id, w_tmp, p0, p1 in _rotate_vir_block(Wvovo, vlocs=vlocs):
        tmp += lib.einsum('aeim, embj -> abij', t2T[:, p0:p1], w_tmp)
        w_tmp = None
    Wvovo = None

    tmp = tmp - tmp.transpose(0, 1, 3, 2)
    t2Tnew += tmp
    tmpT = mpi.alltoall([tmp[:, p0:p1] for p0, p1 in vlocs],
                        split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        t2Tnew[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None
    tmpT = None

    #eris_vvvo = np.array(eris.vvvo)
    #for task_id, eri_tmp, p0, p1 in _rotate_vir_block(eris_vvvo, vlocs=vlocs):
    #    tmp = lib.einsum('ei, baej -> abij', -t1T[p0:p1], eri_tmp[:, :, vloc0:vloc1])
    #    t2Tnew += tmp
    #    t2Tnew -= tmp.transpose(0, 1, 3, 2)
    #    tmp = None
    #eris_vovv = None
    tmp = lib.einsum('ei, baej -> abij', -t1T, eris["vxvo"])
    t2Tnew += tmp
    t2Tnew -= tmp.transpose(0, 1, 3, 2)

    tmp = lib.einsum('am, mbij -> baij', t1T, eris["oxoo"].conj())
    t2Tnew += tmp
    tmpT = mpi.alltoall([tmp[:, p0:p1] for p0, p1 in vlocs],
                        split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        t2Tnew[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None
    tmpT = None

    eia = mo_e_o[:, None] - mo_e_v
    t1Tnew /= eia.T
    for i in range(vloc0, vloc1):
        t2Tnew[i-vloc0] /= lib.direct_sum('i + jb -> bij', eia[:, i], eia)

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
    tauT = np.einsum('ai, bj -> abij', t1T[vloc0:vloc1] * (fac * 0.5), t1T)
    tauT = tauT - tauT.transpose(0, 1, 3, 2)
    ##:tauT = tauT - tauT.transpose(1, 0, 2, 3)
    tauT_tmp = mpi.alltoall([tauT[:, p0:p1] for p0, p1 in vlocs], split_recvbuf=True)
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
    Fme  = lib.einsum('efmn, fn -> em', eris["xvoo"], t1T)
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

    #for task_id, tauT_tilde, p0, p1 in _rotate_vir_block(tauT_tilde, vlocs=vlocs):
    #    Fea += 0.5 * lib.einsum('efmn, famn -> ea', eris["xvoo"][:, p0:p1], tauT_tilde)
    #tauT_tilde = None
    Fae = (-0.5) * lib.einsum('femn, famn -> ae', eris["xvoo"], tauT_tilde)
    Fae = mpi.allreduce(Fae)
    tauT_tilde = None
    
    Fea += lib.einsum('efam, fm -> ea', eris["xvvo"], t1T)
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
    
    Fmi  = 0.5 * lib.einsum('mnef, efin -> mi', eris["ooxv"], tauT_tilde)
    tauT_tilde = None
    Fmi += lib.einsum('mnie, en -> mi', eris["ooox"], t1T[vloc0:vloc1])
    Fmi  = mpi.allreduce(Fmi)

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

    Wmnij = lib.einsum('mnef, efij -> mnij', eris["ooxv"], tauT) * 0.25
    tauT = None
    tmp = lib.einsum('mnie, ej -> mnij', eris["ooox"], t1T[vloc0:vloc1])
    Wmnij += tmp
    Wmnij -= tmp.transpose(0, 1, 3, 2)
    tmp = None
    Wmnij  = mpi.allreduce(Wmnij)
    Wmnij += eris["oooo"]
    return Wmnij

def cc_Wvvvv(t1T, t2T, eris, tauT=None, vlocs=None, full_vvvv=False):
    """
    Wvvvv intermidiates.
    Returns:
        Wvvvv: if full_vvvv, (nvir_seg, nvir, nvir, nvir)
               else: (nvir_seg, nvir, nvir_seg, nvir)
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
    # ZHC FIXME here, I don't understand why this is needed.
    eris_vvoo = np.empty((vloc1-vloc0, nvir, nocc, nocc))
    eris_vvoo[:] = eris["xvoo"]

    tauT = tauT * 0.25
    for task_id, eri_tmp, p0, p1 in _rotate_vir_block(eris_vvoo, vlocs=vlocs):
        Wabef[:, :, p0:p1] = lib.einsum('abmn, efmn -> abef', tauT, eri_tmp)
        eri_tmp = None
    eris_vvoo = None
    tauT = None

    Wabef += np.asarray(eris["xvvv"])
    tmp = lib.einsum('bm, mafe -> abfe', t1T, eris["oxvv"])
    Wabef += tmp

    tmpT = mpi.alltoall([tmp[:, p0:p1] for p0, p1 in vlocs],
                        split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nvir, nvir)
        Wabef[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None

    if not full_vvvv:
        Wabef = Wabef[:, :, vloc0:vloc1]
    return Wabef

def cc_Wovvo(t1T, t2T, eris, vlocs=None):
    """
    mb[e]j.
    """
    nvir_seg, nvir, nocc, _ = t2T.shape
    if vlocs is None:
        ntasks = mpi.pool.size
        vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]

    Wmbej = lib.einsum('mnef, fj -> mnej', eris["ooxv"], -t1T)
    Wmbej = lib.einsum('mnej, bn -> mbej', Wmbej, t1T)
    
    Wmbej += lib.einsum('mbef, fj -> mbej', eris["ovxv"], t1T)

    #tmp = lib.einsum('efbm, fj -> bemj', eris["vvvo"], -t1T)
    #tmpT = mpi.alltoall([tmp[:, p0:p1] for p0, p1 in vlocs],
    #                    split_recvbuf=True)
    #for task_id, (p0, p1) in enumerate(vlocs):
    #    tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
    #    Wmbej[:, p0:p1] += tmp.transpose(2, 0, 1, 3)
    #    tmp = None
    #tmpT = None
    Wmbej += lib.einsum('bn, jemn -> mbej', t1T, eris["oxoo"])

    for task_id, t2T_tmp, p0, p1 in _rotate_vir_block(t2T, vlocs=vlocs):
        Wmbej -= 0.5 * lib.einsum('fbjn, efmn -> mbej', t2T_tmp, eris["xvoo"][:, p0:p1])
        t2T_tmp = None

    Wmbej += np.asarray(eris["ovxo"])
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
        e = np.einsum('ia, ia', fock[:nocc, nocc:], t1)
    else:
        e = 0.0
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir+1))))
    for p0, p1 in lib.prange(0, loc1-loc0, blksize):
        eris_vvoo = eris["xvoo"][p0:p1]
        e += 0.25 * np.einsum('ijab, abij', t2[:, :, p0:p1], eris_vvoo)
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

    time0 = time.clock(), time.time()
    mo_e = eris.mo_energy
    nocc = mycc.nocc
    nvir = mo_e.size - nocc
    eia = mo_e[:nocc, None] - mo_e[None, nocc:]
    t1T = eris.fock[nocc:, :nocc] / eia.T
    loc0, loc1 = _task_location(nvir)

    t2T = np.empty((loc1-loc0, nvir, nocc, nocc))
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir+1))))
    emp2 = 0
    for p0, p1 in lib.prange(0, loc1-loc0, blksize):
        eris_vvoo = eris["xvoo"][p0:p1]
        t2T[p0:p1] = (eris_vvoo / lib.direct_sum('ia, jb -> abij', eia[:, loc0+p0:loc0+p1], eia))
        emp2 += 0.25 * np.einsum('abij, abij', t2T[p0:p1], eris_vvoo.conj()).real
        eris_vvoo = None

    mycc.emp2 = comm.allreduce(emp2)
    logger.info(mycc, 'Init t2, MP2 energy = %.15g', mycc.emp2)
    logger.timer(mycc, 'init mp2', *time0)
    return mycc.emp2, t1T.T, t2T.transpose(2, 3, 0, 1)

def _init_ccsd(ccsd_obj):
    from pyscf import gto
    from mpi4pyscf.tools import mpi
    from mpi4pyscf.cc import gccsd
    if mpi.rank == 0:
        mpi.comm.bcast((ccsd_obj.mol.dumps(), ccsd_obj.pack()))
    else:
        ccsd_obj = gccsd.GCCSD.__new__(gccsd.GCCSD)
        ccsd_obj.t1 = ccsd_obj.t2 = None
        mol, cc_attr = mpi.comm.bcast(None)
        ccsd_obj.mol = gto.mole.loads(mol)
        ccsd_obj.unpack_(cc_attr)
    if False:  # If also to initialize cc._scf object
        if mpi.rank == 0:
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
    tau = mpi.gather(tauT).transpose(2, 3, 0, 1)
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
    Wvvvv = mpi.gather(cc_Wvvvv(t1T, t2T, eris, full_vvvv=True))
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
    Wovvo = mpi.gather(Wovvo.transpose(2, 0, 1, 3)).transpose(1, 2, 0, 3)
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
    imds.wovvo = mpi.gather(np.asarray(imds.wovvo).transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3)
    imds.wovoo = mpi.gather(np.asarray(imds.wovoo).transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3)
    imds.wvvvo = mpi.gather(np.asarray(imds.wvvvo))
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

class GCCSD(gccsd.GCCSD):
    """
    MPI GCCSD.
    """

    conv_tol = getattr(__config__, 'cc_gccsd_GCCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_gccsd_GCCSD_conv_tol_normt', 1e-6)

    # ************************************************************************
    # Initialization
    # ************************************************************************

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        assert isinstance(mf, scf.ghf.GHF)
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        regs = mpi.pool.apply(_init_ccsd, (self,), (None,))
        self._reg_procs = regs

    def pack(self):
        return {'verbose'   : self.verbose,
                'max_memory': self.max_memory,
                'frozen'    : self.frozen,
                'mo_coeff'  : self.mo_coeff,
                'mo_occ'    : self.mo_occ,
                '_nocc'     : self._nocc,
                '_nmo'      : self._nmo,
                'diis_file' : self.diis_file,
                'level_shift': self.level_shift,
                'direct'    : self.direct}

    def unpack_(self, ccdic):
        self.__dict__.update(ccdic)
        return self

    def dump_flags(self, verbose=None):
        if rank == 0:
            gccsd.GCCSD.dump_flags(self, verbose)
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

    def ccsd(self, t1=None, t2=None, eris=None):
        assert self.mo_coeff is not None
        assert self.mo_occ is not None
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        if rank == 0:
            self._finalize()
        return self.e_corr, self.t1, self.t2

    def kernel(self, t1=None, t2=None, eris=None):
        return self.ccsd(t1, t2, eris)

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

    def amplitudes_to_vector(self, t1, t2, out=None):
        return amplitudes_to_vector(t1, t2, out)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vec, nmo, nocc)

    restore_from_diis_ = restore_from_diis_

    def vector_size(self, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        return nocc * nvir + nocc*(nocc-1)//2*nvir*(nvir-1)//2

    # ************************************************************************
    # Lambda, rdm, ip ea
    # ************************************************************************

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        self.converged_lambda, self.l1, self.l2 = \
                gccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                    max_cycle=self.max_cycle,
                                    tol=self.conv_tol_normt,
                                    verbose=self.verbose)
        return self.l1, self.l2

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''Un-relaxed 1-particle density matrix in MO space'''
        return gccsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr)

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

# ************************************************************************
# ao2mo routines
# ************************************************************************

def tril_idx(i, j):
    """
    For a pair / list of tril matrix indices i, j,
    find the corresponding compound indices ij in the tril array.
    
    Args:
        i, j

    Returns:
        ij: compound indices.
    """
    ij  = np.maximum(i, j)
    ij *= (ij + 1)
    ij //= 2
    ij += np.minimum(i, j)
    return ij

def tril_take_idx(idx_list1, idx_list2=None, compact=False):
    """
    Take a submatrix from tril array, 

    If one list is provide:
    return the corresponding compound indices in the tril array.
        e.g. idx_list = [1, 3]
              X     X
          00 01 02 03
        X 10 11 12 13
          20 21 22 23
        X 30 31 32 33
              X     X
          0   *  *  *
        X 1   2  *  *
          3   4  5  *
        X 6   7  8  9
        will return 2, 7, 9 (if compact), else 2, 7, 7, 9. 
        i.e. the indices of [(1, 1), (3, 1), (3, 3)].

    If two lists are provide:
    will return a set of indices for generating a 2D matrix.
        e.g. idx_list1 = [1, 3], idx_list2 = [1, 2]
              X  X   
          00 01 02 03
        X 10 11 12 13
          20 21 22 23
        X 30 31 32 33
              X  X   
          0   *  *  *
        X 1   2  *  *
          3   4  5  *
        X 6   7  8  9
        will return 2, 4, 7, 8,
        i.e. the indices of [(1, 1), (1, 2), (3, 1), (3, 2)].
    """
    if idx_list2 is None:
        idx_list2 = idx_list1
    if compact:
        l = len(idx_list1)
        x = np.tri(l, l, dtype=bool).ravel()
        idx = tril_idx(*lib.cartesian_prod((idx_list1, idx_list1))[x].T)
    else:
        idx = tril_idx(*lib.cartesian_prod((idx_list1, idx_list2)).T)
    return idx

def take_eri(eri, list1, list2, list3, list4, compact=False):
    """
    Take sub block of ERI.
    
    Args:
        eri: 1-fold symmetrized ERI, (nao, nao, nao, nao)
          or 4-fold symmetrized ERI, (nao_pair, nao_pair)
          or 8-fold symmetrized ERI, (nao_pair_pair,) 
        list1, list2, list3, list4: list of indices, can be negative.
        compact: only return the compact form of eri, only valid when lists
                 obey the permutation symmetry (only list1 is used)

    Returns:
        res: (len(list1), len(list2), len(list3), len(list4)) if not compact
             else: compact shape depend only on list1 and list3.
    """
    if eri.ndim == 2: # 4-fold
        nao_a = int(np.sqrt(eri.shape[-2] * 2))
        nao_b = int(np.sqrt(eri.shape[-1] * 2))
        list1 = np.asarray(list1) % nao_a
        list2 = np.asarray(list2) % nao_a
        list3 = np.asarray(list3) % nao_b
        list4 = np.asarray(list4) % nao_b
        idx1 = tril_take_idx(list1, list2, compact=compact)
        idx2 = tril_take_idx(list3, list4, compact=compact)
        if compact:
            res = eri[np.ix_(idx1, idx2)]
        else:
            res = eri[np.ix_(idx1, idx2)].reshape(len(list1), len(list2), \
                    len(list3), len(list4))
    elif eri.ndim == 1: # 8-fold
        nao = int(np.sqrt(int(np.sqrt(eri.shape[-1] * 2)) * 2))
        list1 = np.asarray(list1) % nao
        list2 = np.asarray(list2) % nao
        list3 = np.asarray(list3) % nao
        list4 = np.asarray(list4) % nao
        idx1 = tril_take_idx(list1, list2, compact=compact)
        idx2 = tril_take_idx(list3, list4, compact=compact)
        if compact:
            res = eri[tril_take_idx(idx1, idx2, compact=compact)]
        else:
            res = eri[tril_take_idx(idx1, idx2)].reshape(len(list1), len(list2), \
                    len(list3), len(list4))
    else: # 1-fold
        res = eri[np.ix_(list1, list2, list3, list4)]
    return res

@mpi.parallel_call
def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    """
    Make physist eri with incore ao2mo.
    """
    cput0 = (time.clock(), time.time())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    _sync_(mycc)
    eris = gccsd._PhysicistsERIs()
    
    print ("CHECK")
    print ("rank", rank)

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

    fname = "./gccsd_eri_tmp.h5"
    if rank == 0:
        # ZHC TODO use MPI to do ao2mo and build eri_phys.
        f = h5py.File(fname, 'w')
        mo_a = eris.mo_coeff[:nao//2]
        mo_b = eris.mo_coeff[nao//2:]
        eri  = ao2mo.kernel(mycc._scf._eri, (mo_a, mo_a, mo_b, mo_b))
        lib.hermi_sum(eri, hermi=lib.SYMMETRIC, inplace=True)
        eri += ao2mo.kernel(mycc._scf._eri, mo_a)
        eri += ao2mo.kernel(mycc._scf._eri, mo_b)

        eri_phys = f.create_dataset('eri_phys', (nmo, nmo, nmo, nmo), 'f8')
        unit = nmo**3
        mem_now = lib.current_memory()[0]
        max_memory = max(0, mycc.max_memory - mem_now)
        blksize = min(nmo, max(BLKMIN, int((max_memory*0.9e6/8)/unit)))

        for p0, p1 in lib.prange(0, nmo, blksize):
            eri_slice = take_eri(eri, np.arange(p0, p1), np.arange(nmo),
                                 np.arange(nmo), np.arange(nmo))
            eri_phys[p0:p1]  = eri_slice.transpose(0, 2, 1, 3)
            eri_phys[p0:p1] -= eri_slice.transpose(0, 2, 3, 1)
            eri_slice = None
        eri = None
        f.close()

    comm.Barrier()
    #f = h5py.File(fname, 'r')
    f = lib.H5TmpFile(filename=fname, mode='r')
    eri_phys = f["eri_phys"]
    eris.feri1 = f

    #eris.feri1 = lib.H5TmpFile()
    #eris.oooo = eris.feri1.create_dataset('oooo', (nocc, nocc, nocc, nocc), 'f8')
    ## ooov
    #eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc, vseg, nocc, nocc), 'f8', chunks=(nocc, 1, nocc, nocc))
    #eris.oovv = eris.feri1.create_dataset('oovv', (nocc, nocc, vseg, nvir), 'f8', chunks=(nocc, nocc, 1, nvir))
    #eris.ovov = eris.feri1.create_dataset('ovov', (nocc, vseg, nocc, nvir), 'f8', chunks=(nocc, 1, nocc, nvir))
    #eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc, vseg, nvir, nocc), 'f8', chunks=(nocc, 1, nvir, nocc))
    ## ovvv
    #eris.vvvo = eris.feri1.create_dataset('vvvo', (vseg, nvir, nvir, nocc), 'f8', chunks=(1, nvir, 1, nocc))
    #eris.vvvv = eris.feri1.create_dataset('vvvv', (vseg, nvir, nvir, nvir), 'f8', chunks=(1, nvir, 1, nvir))
    #
    #eris.oooo[:] = eri_phys[:nocc, :nocc, :nocc, :nocc]
    #eris.ovoo[:] = eri_phys[:nocc, nocc+vloc0:nocc+vloc1, :nocc, :nocc]
    #eris.oovv[:] = eri_phys[:nocc, :nocc, nocc+vloc0:nocc+vloc1, nocc:]
    #eris.ovov[:] = eri_phys[:nocc, nocc+vloc0:nocc+vloc1, :nocc, nocc:]
    #eris.ovvo[:] = eri_phys[:nocc, nocc+vloc0:nocc+vloc1, nocc:, :nocc]
    #eris.vvvo[:] = eri_phys[nocc+vloc0:nocc+vloc1, nocc:, nocc:, :nocc]
    #eris.vvvv[:] = eri_phys[nocc+vloc0:nocc+vloc1, nocc:, nocc:, nocc:]
    #
    #eri_phys = None
    #f.close()

    def __getitem__(self, string):
        idx = []
        for s in string:
            if s == 'o':
                idx_s = slice(0, nocc)
            elif s == 'v':
                idx_s = slice(nocc, nmo)
            elif s == 'x':
                idx_s = slice(nocc+vloc0, nocc+vloc1)
            else:
                raise ValueError
            idx.append(idx_s)
        return eri_phys[idx[0], idx[1], idx[2], idx[3]]

    gccsd._PhysicistsERIs.__getitem__ = __getitem__
    # ZHC NOTE another way?
    #gccsd._PhysicistsERIs.oooo = property(lambda self: eri_phys[:nocc, :nocc, :nocc, :nocc])
    
    #comm.Barrier()
    #if rank == 0:
    #    os.remove(fname)

    log.timer('CCSD integral transformation', *cput0)
    mycc._eris = eris
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

