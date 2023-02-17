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
from pyscf.cc import gccsd_rdm
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.ccsd import (_task_location, _rotate_vir_block)

einsum = lib.einsum

comm = mpi.comm
rank = mpi.rank

def _gamma1_intermediates(mycc, t1, t2, l1, l2):
    t1T = t1.T
    t2T = t2.transpose(2, 3, 0, 1)
    l1T = l1.T
    l2T = l2.transpose(2, 3, 0, 1)
    t1 = t2 = l1 = l2 = None

    doo  = -np.dot(l1T.T, t1T)
    doo -= mpi.allreduce_inplace(einsum('efim, efjm -> ij', l2T, t2T) * 0.5)

    dvv  = np.dot(t1T, l1T.T)
    dvv += mpi.allreduce_inplace(einsum('eamn, ebmn -> ab', t2T, l2T) * 0.5)

    xt1  = mpi.allreduce_inplace(einsum('efmn, efin -> mi', l2T, t2T) * 0.5)
    xt2  = mpi.allreduce_inplace(einsum('famn, femn -> ae', t2T, l2T) * 0.5)
    xt2 += np.dot(t1T, l1T.T)

    dvo  = mpi.allgather(np.einsum('aeim, em -> ai', t2T, l1T, optimize=True))
    dvo -= np.dot(t1T, xt1)
    dvo -= np.dot(xt2, t1T)
    dvo += t1T

    dov = l1T.T
    return doo, dov, dvo, dvv

#def _gamma2_new(mycc, t1, t2, l1, l2):
#    t1T = t1.T
#    t2T = np.asarray(t2.transpose(2, 3, 0, 1), order='C')
#    nvir_seg, nvir, nocc = t2T.shape[:3]
#    t1 = t2 = None
#    l1T = l1.T
#    l2T = np.asarray(l2.transpose(2, 3, 0, 1), order='C')
#    l1 = l2 = None
#    
#    ntasks = mpi.pool.size
#    vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
#    vloc0, vloc1 = vlocs[rank]
#    assert vloc1 - vloc0 == nvir_seg
#    
#    tauT = t2T + np.einsum('ai, bj -> abij', t1T[vloc0:vloc1] * 2.0, t1T, optimize=True)
#    
#    # gvvoo [a]bij
#    gvvoo = 0.25 * (l2T.conj() + tauT)
#    
#    #tmp = einsum('kc,kica->ia', l1, t2)
#    #goovv += einsum('ia,jb->ijab', tmp, t1)
#    tmp = einsum('ck, acki -> ai', -l1T, t2T)
#    gvvoo += einsum('ai, bj -> abij', tmp, t1T)
#    
#    #tmp = einsum('kc,kb->cb', l1, t1)
#    #goovv += einsum('cb,ijca->ijab', tmp, t2) * .5
#    tmp = einsum('ck, bk -> cb', l1T, t1T) * (-0.5)
#    gvvoo += einsum('cb, acij -> abij', tmp, t2T)
#    
#    #tmp = einsum('kc,jc->kj', l1, t1)
#    #goovv += einsum('kiab,kj->ijab', tau, tmp) * .5
#    tmp = einsum('ck, cj -> kj', l1T, t1T) * 0.5
#    gvvoo += einsum('abki, kj -> abij', tauT, tmp)
#    
#    #miajb = einsum('ikac, kjcb -> iajb', l2, t2)
#    #mbjai = einsum('acik, cbkj -> bjai', l2T, t2T)
#    mbjai = 0.0 # [b]jai
#    for task_id, l2T_tmp, p0, p1 in _rotate_vir_block(l2T, vlocs=vlocs):
#        mbjai += einsum('bckj, caik -> bjai', t2T[:, p0:p1], l2T_tmp)
#        l2T_tmp = None
#    
#    #tmp = np.einsum('ldjd->lj', miajb, optimize=True)
#    #goovv -= einsum('lj,liba->ijab', tmp, tau) * .25
#    tmp = mpi.allreduce(np.einsum('djdl -> lj', mbjai[:, :, vloc0:vloc1], optimize=True)) * 0.25
#    gvvoo += einsum('lj, abli -> abij', tmp, tauT)
#    
#    #tmp = np.einsum('ldlb->db', miajb, optimize=True)
#    #goovv -= einsum('db,jida->ijab', tmp, tau) * .25
#    tmp = mpi.allgather(np.einsum('bldl -> bd', mbjai, optimize=True)) * 0.25
#    gvvoo += einsum('bd, adji -> abij', tmp, tauT)
#    
#    for task_id, tauT_tmp, p0, p1 in _rotate_vir_block(tauT, vlocs=vlocs):
#        #gvvoo -= einsum('aidl, bdlj -> abij', mbjai, tauT) * .5
#        gvvoo[:, p0:p1] -= einsum('aidl, bdlj -> abij', mbjai, tauT_tmp) * 0.5
#        tauT_tmp = None
#
#    #tmp = einsum('klcd, ijcd -> ijkl', l2, tau) * .25**2
#    #goovv += einsum('ijkl,klab->ijab', tmp, tau)
#    tmp = mpi.allreduce(einsum('cdkl, cdij -> ijkl', l2T, tauT)) * (0.25**2)
#    gvvoo += einsum('ijkl, abkl -> abij', tmp, tauT)
#    
#    gvvoo = gvvoo.conj()
#    
#    #dovov = goovv.transpose(0, 2, 1, 3) - goovv.transpose(0, 3, 1, 2)
#    #dovov =(dovov + dovov.transpose(2,3,0,1)) * .5
#    # ZHC NOTE 
#    # i[a]jb -> iajb - ibja 
#    # dovov = 0.5 * (  g.transpose(0, 2, 1, 3) - g.transpose(0, 3, 1, 2)
#    #                + g.transpose(1, 3, 0, 2) - g.transpose(1, 2, 0, 3) )
#    #                  iajb - ibja + jbia - jaib
#    dovov  = gvvoo.transpose(2, 0, 3, 1) - gvvoo.transpose(3, 0, 2, 1)
#    
#    gvvooT = mpi.alltoall([gvvoo[:, p0:p1] for p0, p1 in vlocs],
#                          split_recvbuf=True)
#    gvvoo = None
#    for task_id, (p0, p1) in enumerate(vlocs):
#        tmp = gvvooT[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
#        dovov[:, :, :, p0:p1] -= tmp.transpose(2, 1, 3, 0)
#        dovov[:, :, :, p0:p1] += tmp.transpose(3, 1, 2, 0)
#        tmp = None
#    gvvooT = None
#    dovov *= 0.5
#
#    #gvvvv = einsum('ijab,ijcd->abcd', tau, l2) * 0.125
#    #dvvvv = gvvvv.transpose(0,2,1,3) - gvvvv.transpose(0,3,1,2)
#    #dvvvv = dvvvv + dvvvv.transpose(1,0,3,2).conj()
#    gvvvv = np.empty((nvir_seg, nvir, nvir, nvir), dtype=t1.dtype)
#    for task_id, l2T_tmp, p0, p1 in _rotate_vir_block(l2T, vlocs=vlocs):
#        gvvvv[:, :, p0:p1] = einsum('abij, cdij -> abcd', tauT, l2T_tmp * 0.125)
#        l2T_tmp = None
#    dvvvv = gvvvv.transpose(0, 2, 1, 3) - gvvvv.transpose(0, 3, 1, 2)
#    #dvvvv = dvvvv + dvvvv.transpose(1,0,3,2).conj()
#    dvvvv_tmp = mpi.alltoall([dvvvv[:, p0:p1] for p0, p1 in vlocs],
#                             split_recvbuf=True)
#    for task_id, (p0, p1) in enumerate(vlocs):
#        tmp = dvvvv_tmp[task_id].reshape(p1-p0, nvir_seg, nvir, nvir)
#        dvvvv[:, p0:p1] += tmp.transpose(1, 0, 3, 2).conj()
#        tmp = None
#    dvvvv_tmp = None
#    
#    
#    #goooo = einsum('klab,ijab->klij', l2, tau) * 0.125
#    #doooo = goooo.transpose(0,2,1,3) - goooo.transpose(0,3,1,2)
#    #doooo = doooo + doooo.transpose(1,0,3,2).conj()
#    goooo = mpi.allreduce(einsum('abkl, abij -> klij', l2T, tauT)) * 0.125
#    doooo = goooo.transpose(0, 2, 1, 3) - goooo.transpose(0, 3, 1, 2)
#    goooo = None
#    doooo = doooo + doooo.transpose(1, 0, 3, 2).conj()
#
#    #gooov  = einsum('jkba,ib->jkia', tau, l1) * -0.25
#    #gooov += einsum('iljk,la->jkia', goooo, t1)
#    #tmp = np.einsum('icjc->ij', miajb, optimize=True) * .25
#    #gooov -= einsum('ij,ka->jkia', tmp, t1)
#    #gooov += einsum('icja,kc->jkia', miajb, t1) * .5
#    #gooov = gooov.conj()
#    #gooov += einsum('jkab,ib->jkia', l2, t1) * .25
#    #dooov = gooov.transpose(0,2,1,3) - gooov.transpose(1,2,0,3)
#    gvooo  = einsum('abjk, bi -> aijk', tauT, l1T * 0.25)
#    gvooo += einsum('iljk, al -> aijk', goooo, t1T)
#    tmp = mpi.allreduce(np.einsum('cjci -> ij', mbjai[:, :, vloc0:vloc1], optimize=True)) * 0.25
#
#    gvooo -= np.einsum('ij, ak -> aijk', tmp, t1T, optimize=True)
#    gvooo += einsum('ajci, ck -> aijk', mbjai, t1T * 0.5)
#    gvooo = gvooo.conj()
#    gvooo += einsum('abjk, bi -> aijk', l2T, t1T * 0.25)
#    
#    # jkia -> jika - kija
#    # aijk -> 2130 - 3120
#    #dooov = gooov.transpose(0,2,1,3) - gooov.transpose(1,2,0,3)
#    dooov = gvooo.transpose(2, 1, 3, 0) - gvooo.transpose(3, 1, 2, 0)
#    gvooo = None
#    
#    #govvo  = einsum('ia,jb->ibaj', l1, t1)
#    #govvo += np.einsum('iajb->ibaj', miajb, optimize=True)
#    #govvo -= einsum('ikac,jc,kb->ibaj', l2, t1, t1)
#    #dovvo = govvo.transpose(0,2,1,3)
#    #dovvo =(dovvo + dovvo.transpose(3,2,1,0).conj()) * .5
#    govvo  = np.einsum('ai, bj -> ibaj', l1T, t1T[vloc0:vloc1], optimize=True)
#    govvo += np.einsum('bjai -> ibaj', mbjai, optimize=True)
#    #govvo -= einsum('acik, cj, bk -> ibaj', l2T, t1T, t1T)
#    tmp = einsum('acik, cj -> ajik', l2T, t1T)
#    for task_id, tmp_tmp, p0, p1 in _rotate_vir_block(tmp, vlocs=vlocs):
#        govvo[:, :, p0:p1] -= einsum('ajik, bk -> ibaj', tmp_tmp, t1T[vloc0:vloc1])
#        tmp_tmp = None
#    
#    # iabj
#    # jbai
#    dovvo = govvo.transpose(3, 1, 2, 0).conj()
#    govvoT = mpi.alltoall([govvo[:, :, p0:p1] for p0, p1 in vlocs],
#                          split_recvbuf=True)
#    govvo = None
#    for task_id, (p0, p1) in enumerate(vlocs):
#        tmp = govvoT[task_id].reshape(nocc, p1-p0, nvir_seg, nocc)
#        dovvo[:, :, p0:p1] += tmp.transpose(0, 2, 1, 3)
#        tmp = None
#    govvoT = None
#    dovvo *= 0.5
#
#    #govvv  = einsum('ja,ijcb->iacb', l1, tau) * .25
#    #govvv += einsum('bcad,id->iabc', gvvvv, t1)
#    #tmp = np.einsum('kakb->ab', miajb, optimize=True) * .25
#    #govvv += einsum('ab,ic->iacb', tmp, t1)
#    #govvv += einsum('kaib,kc->iabc', miajb, t1) * .5
#    #govvv = govvv.conj()
#    #govvv += einsum('ijbc,ja->iabc', l2, t1) * .25
#    #dovvv = govvv.transpose(0,2,1,3) - govvv.transpose(0,3,1,2)
#    # [b]cai
#    gvvvo  = einsum('aj, cbij -> cbai', l1T * (0.25), tauT)
#    gvvvo += einsum('bcad, di -> bcai', gvvvv, t1T)
#    tmp = mpi.allgather(np.einsum('bkak -> ba', mbjai, optimize=True)) * 0.25
#    gvvvo += np.einsum('ba, ci -> cbai', tmp, t1T[vloc0:vloc1], optimize=True)
#    
#    gvvvo += einsum('biak, ck -> bcai', mbjai, t1T * 0.5)
#    gvvvo = gvvvo.conj()
#
#    gvvvo += einsum('bcij, aj -> bcai', l2T, t1T * 0.25)
#
#    # govvv   iabc 
#    # dovvv = ibac - icab
#    # gvvvo   bcai
#    dovvv = gvvvo.transpose(3, 0, 2, 1) - gvvvo.transpose(3, 1, 2, 0)
#    
#    doovv = None # = -dovvo.transpose(0,3,2,1)
#    dvvov = None
#    return (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov)

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
    return gccsd_rdm._make_rdm1(mycc, d1, with_frozen=True, ao_repr=ao_repr)

@mpi.parallel_call
def make_rdm1_ref(mycc, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
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
    
    # ZHC TODO
    # use MPI for distributed intermediates 
    t1, t2 = mycc.gather_amplitudes(t1, t2)
    l1, l2 = mycc.gather_lambda(l1, l2)

    if rank == 0:
        np.save("t1_ref.npy", t1)
        np.save("t2_ref.npy", t2)
        np.save("l1_ref.npy", l1)
        np.save("l2_ref.npy", l2)
        d1 = gccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
        rdm1 = gccsd_rdm._make_rdm1(mycc, d1, with_frozen=True, ao_repr=ao_repr)
    else:
        rdm1 = None
    return rdm1

@mpi.parallel_call
def make_rdm2(mycc, t1, t2, l1, l2, ao_repr=False, with_dm1=True):
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
    # ZHC TODO
    # use MPI for distributed intermediates 
    t1, t2 = mycc.gather_amplitudes(t1, t2)
    l1, l2 = mycc.gather_lambda(l1, l2)

    if rank == 0:
        d2 = gccsd_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2)
        rdm2 = gccsd_rdm._make_rdm2(mycc, d1, d2, with_dm1=with_dm1, with_frozen=True,
                                    ao_repr=ao_repr)
    else:
        rdm2 = None
    return rdm2

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
