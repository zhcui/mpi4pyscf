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
MPI-GCCD with real intergals.

Usage: mpirun -np 2 python gccd.py
"""

import os
import time
from functools import reduce
import h5py
import numpy as np
import scipy.linalg as la

from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
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

from mpi4pyscf.cc import gccsd as mpigccsd
from mpi4pyscf.cc import gccd_lambda as mpigccd_lambda
from mpi4pyscf.cc import gccd_rdm as mpigccd_rdm

comm = mpi.comm
rank = mpi.rank

einsum = lib.einsum

BLKMIN = getattr(__config__, 'cc_ccd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccd_memorymin', 2000)

def update_amps(mycc, t1, t2, eris):
    """
    Update GCCD amplitudes.
    """
    time1 = time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    cpu1 = time0

    t1T = t1.T
    t2T = np.asarray(t2.transpose(2, 3, 0, 1), order='C')
    nvir_seg, nvir, nocc = t2T.shape[:3]
    t2 = None
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
    t1Tnew = np.zeros_like(t1T)
    #t1Tnew  = np.dot(Fvv, t1T)
    #t1Tnew -= np.dot(t1T, Foo)

    tmp  = np.einsum('aeim, me -> ai', t2T, Fov, optimize=True)
    #tmp -= np.einsum('fn, naif -> ai', t1T, eris.oxov, optimize=True)
    tmp  = mpi.allgather(tmp)

    tmp2  = einsum('eamn, mnie -> ai', t2T, eris.ooox)
    tmp2 += einsum('efim, mafe -> ai', t2T, eris.ovvx)
    tmp2 *= 0.5
    tmp2  = mpi.allreduce(tmp2)
    tmp  += tmp2
    tmp2  = None

    #t1Tnew += tmp
    #t1Tnew += fvo

    # T2 equation
    Ftmp = Fvv #- 0.5 * np.dot(t1T, Fov)
    t2Tnew = einsum('aeij, be -> abij', t2T, Ftmp)
    t2T_tmp = mpi.alltoall([t2Tnew[:, p0:p1] for p0, p1 in vlocs],
                           split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = t2T_tmp[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        t2Tnew[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None
    t2T_tmp = None

    Ftmp = Foo #+ 0.5 * np.dot(Fov, t1T)
    tmp = einsum('abim, mj -> abij', t2T, Ftmp)
    t2Tnew -= tmp
    t2Tnew += tmp.transpose(0, 1, 3, 2)
    tmp = None
    
    t2Tnew += np.asarray(eris.xvoo)
    tauT = make_tauT(t1T, t2T, vlocs=vlocs)
    Woooo = cc_Woooo(t1T, t2T, eris, tauT=tauT, vlocs=vlocs)
    t2Tnew += einsum('abmn, mnij -> abij', tauT, Woooo * 0.5)
    Woooo = None

    Wvvvv = cc_Wvvvv(t1T, t2T, eris, tauT=tauT, vlocs=vlocs)
    for task_id, tauT_tmp, p0, p1 in _rotate_vir_block(tauT, vlocs=vlocs):
        t2Tnew += 0.5 * einsum('abef, efij -> abij', Wvvvv[:, :, p0:p1], tauT_tmp)
        tauT_tmp = None
    Wvvvv = None
    tauT = None

    #tmp = einsum('mbje, ei -> bmij', eris.oxov, t1T) # [b]mij
    #tmp = mpi.allgather(tmp) # bmij
    #tmp = einsum('am, bmij -> abij', t1T[vloc0:vloc1], tmp) # [a]bij
    tmp = 0.0

    Wvovo = cc_Wovvo(t1T, t2T, eris, vlocs=vlocs).transpose(2, 0, 1, 3)
    for task_id, w_tmp, p0, p1 in _rotate_vir_block(Wvovo, vlocs=vlocs):
        tmp += einsum('aeim, embj -> abij', t2T[:, p0:p1], w_tmp)
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

    #tmp = einsum('ei, jeba -> abij', t1T, eris.ovvx)
    #t2Tnew += tmp
    #t2Tnew -= tmp.transpose(0, 1, 3, 2)

    #tmp = einsum('am, ijmb -> baij', t1T, eris.ooox.conj())
    #t2Tnew += tmp
    #tmpT = mpi.alltoall([tmp[:, p0:p1] for p0, p1 in vlocs],
    #                    split_recvbuf=True)
    #for task_id, (p0, p1) in enumerate(vlocs):
    #    tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
    #    t2Tnew[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
    #    tmp = None
    #tmpT = None

    eia = mo_e_o[:, None] - mo_e_v
    #t1Tnew /= eia.T
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
    return t2T

def cc_Fov(t1T, eris, vlocs=None):
    """
    Fov: me.
    """
    nvir, nocc = t1T.shape
    #if vlocs is None:
    #    ntasks = mpi.pool.size
    #    vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    fov  = eris.fock[:nocc, nocc:]
    #Fme  = np.einsum('efmn, fn -> em', eris.xvoo, t1T, optimize=True)
    #Fme  = mpi.allgather(Fme).T
    #Fme += fov
    #return Fme
    return fov

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

    #fvo = eris.fock[nocc:, :nocc]
    fvv = eris.fock[nocc+vloc0:nocc+vloc1, nocc:]
    Fea = fvv #- 0.5 * np.dot(fvo[vloc0:vloc1], t1T.T)

    Fae = (-0.5) * einsum('femn, famn -> ae', eris.xvoo, tauT_tilde)
    Fae = mpi.allreduce(Fae)
    tauT_tilde = None
    
    #Fea += np.einsum('mafe, fm -> ea', eris.ovvx, t1T, optimize=True)
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
    #Fmi += np.einsum('mnie, en -> mi', eris.ooox, t1T[vloc0:vloc1], optimize=True)
    Fmi  = mpi.allreduce(Fmi)

    fov = eris.fock[:nocc, nocc:]
    foo = eris.fock[:nocc, :nocc]

    Fmi += foo
    #Fmi += 0.5 * np.dot(fov, t1T)
    return Fmi

def cc_Woooo(t1T, t2T, eris, tauT=None, vlocs=None):
    nvir = t1T.shape[0]
    if vlocs is None:
        ntasks = mpi.pool.size
        vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vloc0, vloc1 = vlocs[rank]
    if tauT is None:
        tauT = make_tauT(t1T, t2T, vlocs=vlocs)

    Wmnij = einsum('efmn, efij -> mnij', eris.xvoo, tauT) * 0.25
    tauT = None
    #tmp = einsum('mnie, ej -> mnij', eris.ooox, t1T[vloc0:vloc1])
    #Wmnij += tmp
    #Wmnij -= tmp.transpose(0, 1, 3, 2)
    #tmp = None
    Wmnij  = mpi.allreduce(Wmnij)
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
    #tmp = einsum('bm, mafe -> abfe', t1T, eris.oxvv)
    #Wabef += tmp

    #tmpT = mpi.alltoall([tmp[:, p0:p1] for p0, p1 in vlocs],
    #                    split_recvbuf=True)
    #for task_id, (p0, p1) in enumerate(vlocs):
    #    tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nvir, nvir)
    #    Wabef[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
    #    tmp = None

    return Wabef

def cc_Wovvo(t1T, t2T, eris, vlocs=None):
    """
    mb[e]j.
    """
    nvir_seg, nvir, nocc, _ = t2T.shape
    if vlocs is None:
        ntasks = mpi.pool.size
        vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]

    #Wmbej = einsum('efmn, fj -> mnej', eris.xvoo, -t1T)
    #Wmbej = einsum('mnej, bn -> mbej', Wmbej, t1T)
    
    #Wmbej -= einsum('mbfe, fj -> mbej', eris.ovvx, t1T)
    #Wmbej += einsum('bn, mnje -> mbej', t1T, eris.ooox)
    
    Wmbej = 0.0

    for task_id, t2T_tmp, p0, p1 in _rotate_vir_block(t2T, vlocs=vlocs):
        Wmbej -= einsum('fbjn, efmn -> mbej', t2T_tmp, eris.xvoo[:, p0:p1])
        t2T_tmp = None
    Wmbej *= 0.5

    Wmbej -= np.asarray(eris.oxov).transpose(2, 3, 1, 0)
    return Wmbej

@mpi.parallel_call(skip_args=[3], skip_kwargs=['eris'])
def energy(mycc, t1=None, t2=None, eris=None):
    '''CCD correlation energy'''
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
    #if rank == 0:
    #    e = np.einsum('ia, ia', fock[:nocc, nocc:], t1, optimize=True)
    #else:
    #    e = 0.0
    e = 0.0
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir+1))))
    for p0, p1 in lib.prange(0, loc1-loc0, blksize):
        eris_vvoo = eris.xvoo[p0:p1]
        e += np.einsum('ijab, abij', t2[:, :, p0:p1], eris_vvoo, optimize=True)
        #e += 0.50 * np.einsum('ia, jb, abij', t1[:, loc0+p0:loc0+p1], t1,
        #                      eris_vvoo, optimize=True)
    e = comm.allreduce(e) * 0.25

    if rank == 0 and abs(e.imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in CCD energy %s', e)
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
    eia = mo_e[:nocc, None] - mo_e[None, nocc:]
    #t1T = eris.fock[nocc:, :nocc] / eia.T
    t1T = np.zeros_like(eris.fock[nocc:, :nocc])
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

def _init_gccd(ccd_obj):
    from pyscf import gto
    from mpi4pyscf.tools import mpi
    from mpi4pyscf.cc import gccd
    if mpi.rank == 0:
        mpi.comm.bcast((ccd_obj.mol.dumps(), ccd_obj.pack()))
    else:
        ccd_obj = gccd.GCCD.__new__(gccd.GCCD)
        ccd_obj.t1 = ccd_obj.t2 = None
        mol, cc_attr = mpi.comm.bcast(None)
        ccd_obj.mol = gto.mole.loads(mol)
        ccd_obj.unpack_(cc_attr)
    if True:  # If also to initialize cc._scf object
        if mpi.rank == 0:
            if hasattr(ccd_obj._scf, '_scf'):
                # ZHC FIXME a hack, newton need special treatment to broadcast
                e_tot = ccd_obj._scf.e_tot
                ccd_obj._scf = ccd_obj._scf._scf
                ccd_obj._scf.e_tot = e_tot
            mpi.comm.bcast((ccd_obj._scf.__class__, _pack_scf(ccd_obj._scf)))
        else:
            mf_cls, mf_attr = mpi.comm.bcast(None)
            ccd_obj._scf = mf_cls(ccd_obj.mol)
            ccd_obj._scf.__dict__.update(mf_attr)

    key = id(ccd_obj)
    mpi._registry[key] = ccd_obj
    regs = mpi.comm.gather(key)
    return regs

class GCCD(mpigccsd.GCCSD):
    """
    MPI GCCD.
    """
    # ************************************************************************
    # Initialization
    # ************************************************************************

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        assert isinstance(mf, scf.ghf.GHF)
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        regs = mpi.pool.apply(_init_gccd, (self,), (None,))
        self._reg_procs = regs
        
    # ************************************************************************
    # core functions
    # ************************************************************************

    init_amps = init_amps
    energy = energy
    update_amps = update_amps

    def ccd(self, t1=None, t2=None, eris=None):
        assert self.mo_coeff is not None
        assert self.mo_occ is not None
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        
        if self.e_hf is None:
            self.e_hf = self._scf.e_tot
        
        if t1 is not None:
            t1 = np.zeros_like(t1)

        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        if rank == 0:
            self._finalize()
        return self.e_corr, self.t1, self.t2

    def kernel(self, t1=None, t2=None, eris=None):
        return self.ccd(t1, t2, eris)

    # ************************************************************************
    # Lambda, rdm, ip ea
    # ************************************************************************

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        
        if t1 is not None:
            t1 = np.zeros_like(t1)
        if l1 is not None:
            l1 = np.zeros_like(l1)

        self.converged_lambda, self.l1, self.l2 = \
                mpigccd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                      max_cycle=self.max_cycle,
                                      tol=self.conv_tol_normt,
                                      verbose=self.verbose)
        return self.l1, self.l2

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''Un-relaxed 1-particle density matrix in MO space'''
        if t1 is not None:
            t1 = np.zeros_like(t1)
        if l1 is not None:
            l1 = np.zeros_like(l1)
        return mpigccd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        if t1 is not None:
            t1 = np.zeros_like(t1)
        if l1 is not None:
            l1 = np.zeros_like(l1)
        return mpigccd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr)
    
CCD = GCCD

def _init_ggccd(ccd_obj):
    from pyscf import gto
    from mpi4pyscf.tools import mpi
    from mpi4pyscf.cc import gccd
    if mpi.rank == 0:
        mpi.comm.bcast((ccd_obj.mol.dumps(), ccd_obj.pack()))
    else:
        ccd_obj = gccd.GGCCD.__new__(gccd.GGCCD)
        ccd_obj.t1 = ccd_obj.t2 = None
        mol, cc_attr = mpi.comm.bcast(None)
        ccd_obj.mol = gto.mole.loads(mol)
        ccd_obj.unpack_(cc_attr)
    if True:  # If also to initialize cc._scf object
        if mpi.rank == 0:
            if hasattr(ccd_obj._scf, '_scf'):
                # ZHC FIXME a hack, newton need special treatment to broadcast
                e_tot = ccd_obj._scf.e_tot
                ccd_obj._scf = ccd_obj._scf._scf
                ccd_obj._scf.e_tot = e_tot
            mpi.comm.bcast((ccd_obj._scf.__class__, _pack_scf(ccd_obj._scf)))
        else:
            mf_cls, mf_attr = mpi.comm.bcast(None)
            ccd_obj._scf = mf_cls(ccd_obj.mol)
            ccd_obj._scf.__dict__.update(mf_attr)
    
    key = id(ccd_obj)
    mpi._registry[key] = ccd_obj
    regs = mpi.comm.gather(key)
    return regs

class GGCCD(GCCD):
    """
    MPI GGCCD.
    """
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        assert isinstance(mf, scf.ghf.GHF)
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        regs = mpi.pool.apply(_init_ggccd, (self,), (None,))
        self._reg_procs = regs

    def ao2mo(self, mo_coeff=None):
        mpigccsd._make_eris_incore_ghf(self, mo_coeff)
        return 'Done'

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

