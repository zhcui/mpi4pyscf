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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Jun Yang
#         Zhi-Hao Cui <zhcui0408@gmail.com>
#

import numpy as np
from pyscf import lib
from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.lib import diis
from mpi4pyscf.cc.ccsd import (_task_location, _sync_, _diff_norm, 
                               _rotate_vir_block)

einsum = lib.einsum
#from functools import partial
#einsum = partial(np.einsum, optimize=True)

comm = mpi.comm
rank = mpi.rank

def _gamma1_intermediates(mycc, t1, t2, l1, l2):
    t1T = t1.T
    t2T = t2.transpose(2, 3, 0, 1)
    l1T = l1.T
    l2T = l2.transpose(2, 3, 0, 1)
    t1 = t2 = l1 = l2 = None

    #doo  = -np.dot(l1T.T, t1T)
    doo = -mpi.allreduce(einsum('efim, efjm -> ij', l2T, t2T) * 0.5)

    #dvv  = np.dot(t1T, l1T.T)
    dvv = mpi.allreduce(einsum('eamn, ebmn -> ab', t2T, l2T) * 0.5)

    #xt1  = mpi.allreduce(einsum('efmn, efin -> mi', l2T, t2T) * 0.5)
    #xt2  = mpi.allreduce(einsum('famn, femn -> ae', t2T, l2T) * 0.5)
    #xt2 += np.dot(t1T, l1T.T)

    #dvo  = mpi.allgather(np.einsum('aeim, em -> ai', t2T, l1T, optimize=True))
    #dvo -= np.dot(t1T, xt1)
    #dvo -= np.dot(xt2, t1T)
    #dvo += t1T

    #dov = l1T.T
    nvir, nocc = t1T.shape
    dvo = np.zeros((nvir, nocc), dtype=t1T.dtype)
    dov = np.zeros((nocc, nvir), dtype=t1T.dtype)
    return doo, dov, dvo, dvv

# gamma2 intermediates in Chemist's notation
# When computing intermediates, the convention
# dm2[q,p,s,r] = <p^\dagger r^\dagger s q> is assumed in this function.
# It changes to dm2[p,q,r,s] = <p^\dagger r^\dagger s q> in _make_rdm2
def _gamma2_intermediates(mycc, t1, t2, l1, l2):
    tau = t2 + np.einsum('ia, jb -> ijab', t1, t1 * 2.0, optimize=True)
    miajb = einsum('ikac, kjcb -> iajb', l2, t2)

    goovv = 0.25 * (l2.conj() + tau)
    tmp = einsum('kc,kica->ia', l1, t2)
    goovv += einsum('ia,jb->ijab', tmp, t1)
    tmp = einsum('kc,kb->cb', l1, t1)
    goovv += einsum('cb,ijca->ijab', tmp, t2) * .5
    tmp = einsum('kc,jc->kj', l1, t1)
    goovv += einsum('kiab,kj->ijab', tau, tmp) * .5
    tmp = np.einsum('ldjd->lj', miajb, optimize=True)
    goovv -= einsum('lj,liba->ijab', tmp, tau) * .25
    tmp = np.einsum('ldlb->db', miajb, optimize=True)
    goovv -= einsum('db,jida->ijab', tmp, tau) * .25
    goovv -= einsum('ldia,ljbd->ijab', miajb, tau) * .5
    tmp = einsum('klcd,ijcd->ijkl', l2, tau) * .25**2
    goovv += einsum('ijkl,klab->ijab', tmp, tau)
    goovv = goovv.conj()

    gvvvv = einsum('ijab,ijcd->abcd', tau, l2) * 0.125
    goooo = einsum('klab,ijab->klij', l2, tau) * 0.125

    gooov  = einsum('jkba,ib->jkia', tau, l1) * -0.25
    gooov += einsum('iljk,la->jkia', goooo, t1)
    tmp = np.einsum('icjc->ij', miajb, optimize=True) * .25
    gooov -= einsum('ij,ka->jkia', tmp, t1)
    gooov += einsum('icja,kc->jkia', miajb, t1) * .5
    gooov = gooov.conj()
    gooov += einsum('jkab,ib->jkia', l2, t1) * .25

    govvo  = einsum('ia,jb->ibaj', l1, t1)
    govvo += np.einsum('iajb->ibaj', miajb, optimize=True)
    govvo -= einsum('ikac,jc,kb->ibaj', l2, t1, t1)

    govvv  = einsum('ja,ijcb->iacb', l1, tau) * .25
    govvv += einsum('bcad,id->iabc', gvvvv, t1)
    tmp = np.einsum('kakb->ab', miajb, optimize=True) * .25
    govvv += einsum('ab,ic->iacb', tmp, t1)
    govvv += einsum('kaib,kc->iabc', miajb, t1) * .5
    govvv = govvv.conj()
    govvv += einsum('ijbc,ja->iabc', l2, t1) * .25

    dovov = goovv.transpose(0,2,1,3) - goovv.transpose(0,3,1,2)
    dvvvv = gvvvv.transpose(0,2,1,3) - gvvvv.transpose(0,3,1,2)
    doooo = goooo.transpose(0,2,1,3) - goooo.transpose(0,3,1,2)
    dovvv = govvv.transpose(0,2,1,3) - govvv.transpose(0,3,1,2)
    dooov = gooov.transpose(0,2,1,3) - gooov.transpose(1,2,0,3)
    dovvo = govvo.transpose(0,2,1,3)
    dovov =(dovov + dovov.transpose(2,3,0,1)) * .5
    dvvvv = dvvvv + dvvvv.transpose(1,0,3,2).conj()
    doooo = doooo + doooo.transpose(1,0,3,2).conj()
    dovvo =(dovvo + dovvo.transpose(3,2,1,0).conj()) * .5
    doovv = None # = -dovvo.transpose(0,3,2,1)
    dvvov = None
    return (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov)

def _gamma2_outcore(mycc, t1, t2, l1, l2, h5fobj):
    t1T = t1.T
    t2T = np.asarray(t2.transpose(2, 3, 0, 1), order='C')
    nvir_seg, nvir, nocc = t2T.shape[:3]
    t1 = t2 = None
    l1T = l1.T
    l2T = np.asarray(l2.transpose(2, 3, 0, 1), order='C')
    l1 = l2 = None
    fswap = lib.H5TmpFile()
    
    ntasks = mpi.pool.size
    vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vloc0, vloc1 = vlocs[rank]
    assert vloc1 - vloc0 == nvir_seg
    
    tauT = t2T + np.einsum('ia, jb -> ijab', t1T[vloc0:vloc1] * 2.0, t1T, optimize=True)
    
    #miajb = einsum('acik, cbkj -> iajb', l2T, t2T) # i[a]jb
    mbjai = 0.0 # [b]jai
    for task_id, l2T_tmp, p0, p1 in _rotate_vir_block(l2T, vlocs=vlocs):
        mbjai += einsum('bckj, caik -> bjai', t2T[:, p0:p1], l2T_tmp)
        l2T_tmp = None

    gvvoo = 0.25 * (l2T.conj() + tauT)
    tmp = einsum('ck, acki -> ai', l1T, t2T)
    gvvoo -= np.einsum('ai, bj -> abij', tmp, t1T, optimize=True)
    
    tmp = np.dot(l1T, t1T.T)
    gvvoo -= einsum('cb, acij -> abij', tmp * 0.5, t2)
    
    tmp = np.dot(l1T.T, t1T)
    gvvoo += einsum('abki, kj -> abij', tauT, tmp * 0.5)

    #tmp_oo = mpi.allreduce(np.einsum('ldjd -> lj', miajb[:, :, :, vloc0:vloc1]))
    tmp_oo = mpi.allreduce(np.einsum('djdl -> lj', mbjai[:, :, vloc0:vloc1], optimize=True))
    gvvoo += einsum('lj, abli -> abij', tmp_oo * 0.25, tauT)

    #tmp_vv = mpi.allgather(np.einsum('ldlb -> db', miajb))
    tmp_vv = mpi.allgather(np.einsum('bldl -> bd', mbjai, optimize=True)).T # db
    gvvoo += einsum('db, adji -> abij', tmp * 0.25, tauT)
    #gvvoo -= einsum('ldia, bdlj -> abij', miajb, tauT) * 0.5
    # gvvvv shape (nvir, nvir, nvir_seg, nvir)
    gvvvv = fswap.create_dataset('gvvvv', (nvir, nvir, nvir_seg, nvir), t1.dtype,
                                 chunks=(nvir, nvir, 1, nvir))
    for task_id, tauT_tmp, p0, p1 in _rotate_vir_block(tauT, vlocs=vlocs):
        #gvvoo[:, p0:p1] -= einsum('ldia, bdlj -> abij', miajb[:, :, :, vloc0:vloc1], tauT_tmp) * 0.5
        gvvoo[:, p0:p1] -= einsum('aidl, bdlj -> abij', mbjai, tauT_tmp) * 0.5
        gvvvv[p0:p1] = einsum('abij, cdij -> abcd', tauT_tmp, l2T * 0.125)
        tauT_tmp = None

    #dvvvv = gvvvv.transpose(0,2,1,3) - gvvvv.transpose(0,3,1,2)
    #dvvvv = dvvvv + dvvvv.transpose(1,0,3,2).conj()
    # abcd
    # dvvvv = (a[c]bd - adb[c]) + ([c]adb - da[c]b)
    dvvvv = h5fobj.create_dataset('dvvvv', (nvir_seg, nvir, nvir, nvir), t1.dtype,
                                  chunks=(1, nvir, nvir, nvir))
    


    tmp = einsum('cdkl, cdij -> klij', l2T, tauT) * 0.25**2
    gvvoo += einsum('abkl, klij -> abij', tauT, tmp)
    gvvoo = gvvoo.conj()
    tmp = None

    #dovov = goovv.transpose(0,2,1,3) - goovv.transpose(0,3,1,2)
    dovov = h5fobj.create_dataset('dovov', (nocc, nvir_seg, nocc, nvir), t1.dtype,
                                  chunks=(nocc, 1, nocc, nvir))
    
    # dovov = 0.5 * [(iajb - ibja) + (jbia - jaib)]
    # abij -> iajb and -jaib
    dovov[:] = gvvoo.transpose(2, 0, 3, 1)
    dovov   -= gvvoo.transpose(3, 0, 2, 1)
    gvvooT = mpi.alltoall([gvvoo[:, p0:p1] for p0, p1 in vlocs],
                          split_recvbuf=True)
    gvvoo = None
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = gvvooT[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        # abij -> baij -> -ibja and abij -> baij -> +jbia
        dovov[:, :, :, p0:p1] -= tmp.transpose(2, 1, 3, 0)
        dovov[:, :, :, p0:p1] += tmp.transpose(3, 1, 2, 0)
        tmp = None
    gvvooT = None
    dovov *= 0.5

    #gvvvv = einsum('abij, cdij -> abcd', tauT, l2T * 0.125) #see line 170
    goooo = mpi.allreduce(einsum('abkl, abij -> klij', l2T, tauT) * 0.125)

    gvooo  = einsum('abjk, bi -> aikj', tauT, l1T * 0.25)
    gvooo += einsum('al, iljk -> aikj', t1T, goooo)

    tmp = tmp_oo * 0.25
    tmp_oo = None
    gvooo -= np.einsum('ak, ij -> aikj', t1T, tmp, optimize=True)
    gvooo += einsum('ajci, ck -> aikj', mbjai, t1T * 0.5)
    gvooo = gvooo.conj()
    gvooo += einsum('abjk, bi -> aikj', l2T, t1T * 0.25)
    
    # jkia -> jika
    # jkia -> kija
    #dooov = gooov.transpose(0,2,1,3) - gooov.transpose(1,2,0,3)
    dooov = h5fobj.create_dataset('dooov', (nocc, nocc, nocc, nvir_seg), t1.dtype,
                                  chunks=(nocc, nocc, nocc, 1))
    # aikj -> jika
    dooov[:] = gvooo.transpose(3, 1, 2, 0)
    # aikj -> kija
    dooov -= gvooo.transpose(2, 1, 3, 0)
    gvooo = None
    
    # i[b]aj
    govvo  = np.einsum('ai, bj -> ibaj', l1T, t1T[vloc0:vloc1], optimize=True)
    govvo += np.einsum('bjai -> ibaj', mbjai, optimize=True)
    #govvo -= einsum('acik, cj, bk -> ibaj', l2T, t1T, t1T)
    tmp = einsum('acik, cj -> aikj', l2T, t1T)
    for task_id, tmp_tmp, p0, p1 in _rotate_vir_block(tmp, vlocs=vlocs):
        govvo[:, :, p0:p1] -= einsum('aikj, bk -> ibaj', tmp_tmp, t1T[vloc0:vloc1])
        tmp_tmp = None
    tmp = None
    dovvo = h5fobj.create_dataset('dovvo', (nocc, nvir_seg, nvir, nocc), t1.dtype,
                                  chunks=(nocc, 1, nvir, nocc))
    # ibaj -> iabj
    # (iabj + jbai) * 0.5
    #dovvo = govvo.transpose(0,2,1,3)
    #dovvo =(dovvo + dovvo.transpose(3,2,1,0).conj()) * .5
    dovvo[:] = govvo.transpose(3, 1, 2, 0)
    govvoT = mpi.alltoall([govvo[:, :, p0:p1] for p0, p1 in vlocs],
                          split_recvbuf=True)
    govvo = None
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = govvoT[task_id].reshape(nocc, p1-p0, nvir_seg, nocc)
        dovvo[:, :, p0:p1] += tmp.transpose(0, 2, 1, 3)
        tmp = None
    govvoT = None 
    
    # govvv has shape ia[b]c
    govvv  = einsum('aj, cbij -> iacb', l1T * 0.25, tauT)
    govvv += einsum('bcad, di -> iabc', fswap["gvvvv"], t1T)
    tmp = tmp_vv * 0.25
    tmp_vv = None
    govvv += np.einsum('ab, ci -> iacb', tmp, t1T[vloc0:vloc1], optimize=True)
    govvv += einsum('biak, ck -> iabc', mbjai, t1T * 0.5)
    govvv = govvv.conj()
    govvv += einsum('bcij, aj -> iabc', l2T, t1T * 0.25)
    
    # ia[b]c -> i[b]ac
    # ia[b]c -> ica[b]
    # ibac - icab
    #dovvv = govvv.transpose(0,2,1,3) - govvv.transpose(0,3,1,2)
    dovvv = h5fobj.create_dataset('dovvv', (nocc, nvir_seg, nvir, nvir), t1.dtype,
                                  chunks=(nocc, 1, nvir, nvir))
    dovvv[:] = govvv.transpose(0, 2, 1, 3)
    # exchange b and c, so that ia[b]c become iab[c]
    govvvT = mpi.alltoall([govvv[:, :, :, p0:p1] for p0, p1 in vlocs],
                          split_recvbuf=True)
    govvv = None
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = govvvT[task_id].reshape(nocc, nvir, p1-p0, nvir_seg)
        dovvv[:, :, :, p0:p1] += tmp.transpose(0, 3, 1, 2)
        tmp = None
    govvvT = None 
    
    doooo = goooo.transpose(0,2,1,3) - goooo.transpose(0,3,1,2)
    doooo = doooo + doooo.transpose(1,0,3,2).conj()
    doovv = None # = -dovvo.transpose(0,3,2,1)
    dvvov = None
    return (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov)

@mpi.parallel_call
def make_rdm1(mycc, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
    r'''
    One-particle density matrix in the molecular spin-orbital representation
    (the occupied-virtual blocks from the orbital response contribution are
    not included).

    dm1[p,q] = <q^\dagger p>  (p,q are spin-orbitals)

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if l1 is None:
        l1 = mycc.l1
    if l2 is None:
        l2 = mycc.l2
    if l1 is None:
        l1, l2 = mycc.solve_lambda(t1, t2)
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2)
    return _make_rdm1(mycc, d1, with_frozen=True, ao_repr=ao_repr)

@mpi.parallel_call
def make_rdm2(mycc, t1, t2, l1, l2, ao_repr=False):
    r'''
    Two-particle density matrix in the molecular spin-orbital representation

    dm2[p,q,r,s] = <p^\dagger r^\dagger s q>

    where p,q,r,s are spin-orbitals. p,q correspond to one particle and r,s
    correspond to another particle.  The contraction between ERIs (in
    Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if l1 is None:
        l1 = mycc.l1
    if l2 is None:
        l2 = mycc.l2
    if l1 is None:
        l1, l2 = mycc.solve_lambda(t1, t2)
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2)
    d2 = _gamma2_intermediates(mycc, t1, t2, l1, l2)
    return _make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True,
                      ao_repr=ao_repr)

def _make_rdm1(mycc, d1, with_frozen=True, ao_repr=False):
    r'''
    One-particle density matrix in the molecular spin-orbital representation
    (the occupied-virtual blocks from the orbital response contribution are
    not included).

    dm1[p,q] = <q^\dagger p>  (p,q are spin-orbitals)

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    doo, dov, dvo, dvv = d1
    nocc, nvir = dov.shape
    nmo = nocc + nvir

    dm1 = np.empty((nmo, nmo), dtype=doo.dtype)
    dm1[:nocc, :nocc] = doo + doo.conj().T
    dm1[:nocc, nocc:] = dov + dvo.conj().T
    dm1[nocc:, :nocc] = dm1[:nocc, nocc:].conj().T
    dm1[nocc:, nocc:] = dvv + dvv.conj().T
    dm1 *= .5
    dm1[np.diag_indices(nocc)] += 1

    if with_frozen and mycc.frozen is not None:
        nmo = mycc.mo_occ.size
        nocc = np.count_nonzero(mycc.mo_occ > 0)
        rdm1 = np.zeros((nmo,nmo), dtype=dm1.dtype)
        rdm1[np.diag_indices(nocc)] = 1
        moidx = np.where(mycc.get_frozen_mask())[0]
        rdm1[moidx[:,None], moidx] = dm1
        dm1 = rdm1

    if ao_repr:
        mo = mycc.mo_coeff
        dm1 = einsum('pi, ij, qj -> pq', mo, dm1, mo.conj())
    return dm1

def _make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True, ao_repr=False):
    r'''
    dm2[p,q,r,s] = <p^\dagger r^\dagger s q>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    nocc, nvir = dovov.shape[:2]
    nmo = nocc + nvir

    dm2 = np.empty((nmo,nmo,nmo,nmo), dtype=doooo.dtype)

    dovov = np.asarray(dovov)
    dm2[:nocc,nocc:,:nocc,nocc:] = dovov
    dm2[nocc:,:nocc,nocc:,:nocc] = dm2[:nocc,nocc:,:nocc,nocc:].transpose(1,0,3,2).conj()
    dovov = None

    dovvo = np.asarray(dovvo)
    dm2[:nocc,:nocc,nocc:,nocc:] =-dovvo.transpose(0,3,2,1)
    dm2[nocc:,nocc:,:nocc,:nocc] =-dovvo.transpose(2,1,0,3)
    dm2[:nocc,nocc:,nocc:,:nocc] = dovvo
    dm2[nocc:,:nocc,:nocc,nocc:] = dovvo.transpose(1,0,3,2).conj()
    dovvo = None

    dm2[nocc:,nocc:,nocc:,nocc:] = dvvvv
    dm2[:nocc,:nocc,:nocc,:nocc] = doooo

    dovvv = np.asarray(dovvv)
    dm2[:nocc,nocc:,nocc:,nocc:] = dovvv
    dm2[nocc:,nocc:,:nocc,nocc:] = dovvv.transpose(2,3,0,1)
    dm2[nocc:,nocc:,nocc:,:nocc] = dovvv.transpose(3,2,1,0).conj()
    dm2[nocc:,:nocc,nocc:,nocc:] = dovvv.transpose(1,0,3,2).conj()
    dovvv = None

    dooov = np.asarray(dooov)
    dm2[:nocc,:nocc,:nocc,nocc:] = dooov
    dm2[:nocc,nocc:,:nocc,:nocc] = dooov.transpose(2,3,0,1)
    dm2[:nocc,:nocc,nocc:,:nocc] = dooov.transpose(1,0,3,2).conj()
    dm2[nocc:,:nocc,:nocc,:nocc] = dooov.transpose(3,2,1,0).conj()

    if with_frozen and mycc.frozen is not None:
        nmo, nmo0 = mycc.mo_occ.size, nmo
        nocc = np.count_nonzero(mycc.mo_occ > 0)
        rdm2 = np.zeros((nmo,nmo,nmo,nmo), dtype=dm2.dtype)
        moidx = np.where(mycc.get_frozen_mask())[0]
        idx = (moidx.reshape(-1,1) * nmo + moidx).ravel()
        lib.takebak_2d(rdm2.reshape(nmo**2,nmo**2),
                       dm2.reshape(nmo0**2,nmo0**2), idx, idx)
        dm2 = rdm2

    if with_dm1:
        dm1 = _make_rdm1(mycc, d1, with_frozen)
        dm1[np.diag_indices(nocc)] -= 1

        for i in range(nocc):
# Be careful with the convention of dm1 and the transpose of dm2 at the end
            dm2[i,i,:,:] += dm1
            dm2[:,:,i,i] += dm1
            dm2[:,i,i,:] -= dm1
            dm2[i,:,:,i] -= dm1.T

        for i in range(nocc):
            for j in range(nocc):
                dm2[i,i,j,j] += 1
                dm2[i,j,j,i] -= 1

    # dm2 was computed as dm2[p,q,r,s] = < p^\dagger r^\dagger s q > in the
    # above. Transposing it so that it be contracted with ERIs (in Chemist's
    # notation):
    #   E = einsum('pqrs,pqrs', eri, rdm2)
    dm2 = dm2.transpose(1,0,3,2)
    if ao_repr:
        from pyscf.cc import ccsd_rdm
        dm2 = ccsd_rdm._rdm2_mo2ao(dm2, mycc.mo_coeff)
    return dm2


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf.cc import gccsd

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1.)
    mf = scf.addons.convert_to_ghf(mf)

    mycc = gccsd.GCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    dm1 = make_rdm1(mycc, t1, t2, l1, l2)
    dm2 = make_rdm2(mycc, t1, t2, l1, l2)
    nao = mol.nao_nr()
    mo_a = mf.mo_coeff[:nao]
    mo_b = mf.mo_coeff[nao:]
    nmo = mo_a.shape[1]
    eri = ao2mo.kernel(mf._eri, mo_a+mo_b, compact=False).reshape([nmo]*4)
    orbspin = mf.mo_coeff.orbspin
    sym_forbid = (orbspin[:,None] != orbspin)
    eri[sym_forbid,:,:] = 0
    eri[:,:,sym_forbid] = 0
    hcore = scf.RHF(mol).get_hcore()
    h1 = reduce(np.dot, (mo_a.T.conj(), hcore, mo_a))
    h1+= reduce(np.dot, (mo_b.T.conj(), hcore, mo_b))
    e1 = np.einsum('ij,ji', h1, dm1)
    e1+= np.einsum('ijkl,ijkl', eri, dm2) * .5
    e1+= mol.energy_nuc()
    print(e1 - mycc.e_tot)

    #TODO: test 1pdm, 2pdm against FCI
