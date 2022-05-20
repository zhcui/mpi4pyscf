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

import numpy as np
from pyscf import lib
from pyscf.cc import ccsd_lambda

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.lib import diis
from mpi4pyscf.cc.ccsd import (_task_location, _sync_, _diff_norm, 
                               _rotate_vir_block)

comm = mpi.comm
rank = mpi.rank

einsum = lib.einsum
#from functools import partial
#einsum = partial(np.einsum, optimize=True)

@mpi.parallel_call(skip_args=[1], skip_kwargs=['eris'])
def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-6, verbose=None,
           fintermediates=None, fupdate=None, approx_l=False):
    """
    CCSD lambda kernel.
    """
    log = logger.new_logger(mycc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    _sync_(mycc)

    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo(mycc.mo_coeff)
        eris = mycc._eris
    
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if l1 is None:
        if mycc.l1 is None:
            l1 = t1
        else:
            l1 = mycc.l1
    if l2 is None:
        if mycc.l2 is None:
            l2 = t2
        else:
            l2 = mycc.l2
    
    if approx_l:
        mycc.l1 = l1
        mycc.l2 = l2
        conv = True
        return conv, l1, l2
    
    if fintermediates is None:
        fintermediates = make_intermediates
    if fupdate is None:
        fupdate = update_lambda

    imds = fintermediates(mycc, t1, t2, eris)

    if isinstance(mycc.diis, diis.DistributedDIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = diis.DistributedDIIS(mycc, mycc.diis_file)
        adiis.space = mycc.diis_space
    else:
        adiis = None
    cput0 = log.timer('CCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = fupdate(mycc, t1, t2, l1, l2, eris, imds)
        normt = _diff_norm(mycc, l1new, l2new, l1, l2)
        l1, l2 = l1new, l2new
        l1new = l2new = None
        l1, l2 = mycc.run_diis(l1, l2, istep, normt, 0, adiis)
        log.info('cycle = %d  norm(lambda1,lambda2) = %.6g', istep+1, normt)
        cput0 = log.timer('CCSD iter', *cput0)
        if normt < tol:
            conv = True
            break

    mycc.l1 = l1
    mycc.l2 = l2
    log.timer('CCSD lambda', *cput0)
    return conv, l1, l2

# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris):
    t1T = t1.T
    t2T = np.asarray(t2.transpose(2, 3, 0, 1), order='C')
    nvir_seg, nvir, nocc = t2T.shape[:3]
    t1 = t2 = None
    ntasks = mpi.pool.size
    vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vloc0, vloc1 = vlocs[rank]
    assert vloc1 - vloc0 == nvir_seg
    
    class _IMDS:
        pass
    imds = _IMDS()
    imds.ftmp = lib.H5TmpFile()
    dtype = t1T.dtype
    imds.woooo = imds.ftmp.create_dataset('woooo', (nocc, nocc, nocc, nocc), dtype)
    imds.wovvo = imds.ftmp.create_dataset('wovvo', (nocc, nvir_seg, nvir, nocc), dtype)
    imds.wovoo = imds.ftmp.create_dataset('wovoo', (nocc, nvir_seg, nocc, nocc), dtype)
    imds.wvvvo = imds.ftmp.create_dataset('wvvvo', (nvir_seg, nvir, nvir, nocc), dtype)
    
    foo = eris.fock[:nocc, :nocc]
    fov = eris.fock[:nocc, nocc:]
    fvo = eris.fock[nocc:, :nocc]
    fvv = eris.fock[nocc:, nocc:]
    
    tauT = np.einsum('ai, bj -> abij', t1T[vloc0:vloc1] * 2.0, t1T, optimize=True)
    tauT += t2T

    v1  = fvv - np.dot(t1T, fov)
    #tmp = einsum('jbac, cj -> ba', eris.oxvv, t1T)
    tmp = einsum('bjca, cj -> ba', eris.xovv, t1T)
    #v4 = np.zeros((nocc, nvir_seg, nvir, nocc))
    v4 = 0.0

    eris_voov = eris.xvoo.transpose(0, 2, 3, 1)
    for task_id, eri_tmp, p0, p1 in _rotate_vir_block(eris_voov, vlocs=vlocs):
        tmp -= einsum('cjka, bcjk -> ba', eri_tmp, tauT[:, p0:p1]) * 0.5
        v4  += einsum('dljb, cdkl -> jcbk', eri_tmp, t2T[:, p0:p1])
        eri_tmp = None
    eris_voov = None
    v1 -= mpi.allgather(tmp)
    
    v2  = foo + np.dot(fov, t1T)
    #tmp  = einsum('kijb, bk -> ij', eris.ooox, t1T[vloc0:vloc1])
    tmp  = einsum('bjik, bk -> ij', eris.xooo, t1T[vloc0:vloc1])
    tmp -= einsum('bcik, bcjk -> ij', eris.xvoo, tauT) * 0.5
    v2  -= mpi.allreduce_inplace(tmp)
    
    #v4 -= np.asarray(eris.oxov).transpose(0, 1, 3, 2)
    v4 -= np.asarray(eris.xovo).transpose(1, 0, 2, 3)
    
    v5  = fvo + mpi.allgather(einsum('kc, bcjk -> bj', fov, t2T))
    tmp = fvo[vloc0:vloc1] + einsum('cdkl, dl -> ck', eris.xvoo, t1T)
    v5 += mpi.allreduce_inplace(np.einsum('ck, bk, cj -> bj', tmp, t1T, t1T[vloc0:vloc1], optimize=True))
     
    #v5 += mpi.allreduce(einsum('kljc, cbkl -> bj', eris.ooox, t2T)) * 0.5
    v5 += mpi.allreduce_inplace(einsum('cjlk, cbkl -> bj', eris.xooo, t2T) * 0.5)
    #tmp = np.zeros((nvir_seg, nocc))
    tmp = 0.0
    for task_id, t2T_tmp, p0, p1 in _rotate_vir_block(t2T, vlocs=vlocs):
        #tmp += einsum('kbcd, cdjk -> bj', eris.oxvv[:, :, p0:p1], t2T_tmp)
        tmp -= einsum('bkcd, cdjk -> bj', eris.xovv[:, :, p0:p1], t2T_tmp)
        t2T_tmp = None
    tmp *= 0.5
    tmp  = mpi.allgather(tmp)
    v5  -= tmp

    w3  = v5[vloc0:vloc1] + einsum('jcbk, bj -> ck', v4, t1T)
    w3 += np.dot(v1[vloc0:vloc1], t1T)
    w3 -= np.dot(t1T[vloc0:vloc1], v2)
    w3  = mpi.allgather(w3)

    woooo  = einsum('cdij, cdkl -> ijkl', eris.xvoo, tauT)
    woooo *= 0.25
    #woooo += einsum('jilc, ck -> jilk', eris.ooox, t1T[vloc0:vloc1])
    woooo += einsum('clij, ck -> jilk', eris.xooo, t1T[vloc0:vloc1])
    woooo  = mpi.allreduce_inplace(woooo)
    woooo += np.asarray(eris.oooo) * 0.5
    imds.woooo[:] = woooo
    woooo = None
    
    # ZHC NOTE: wovvo, v4 has shape j[c]bk
    #wovvo = v4 + einsum('jcbd, dk -> jcbk', eris.oxvv, t1T)
    wovvo = v4 + einsum('cjdb, dk -> jcbk', eris.xovv, t1T)
    
    tmp = einsum('bdlj, dk -> bklj', eris.xvoo, t1T)
    for task_id, tmp_2, p0, p1 in _rotate_vir_block(tmp, vlocs=vlocs):
        wovvo[:, :, p0:p1] += einsum('bklj, cl -> jcbk', tmp_2, t1T[vloc0:vloc1])
        tmp_2 = None
    tmp = None
    
    eris_vooo = eris.xooo
    for task_id, eri_tmp, p0, p1 in _rotate_vir_block(eris_vooo, vlocs=vlocs):
        wovvo[:, :, p0:p1] -= einsum('bkjl, cl -> jcbk', eri_tmp, t1T[vloc0:vloc1])
        eri_tmp = None
    eris_vooo = None
    imds.wovvo[:] = wovvo
    wovvo = None

    #wovoo = np.zeros((nocc, nvir_seg, nocc, nocc))
    wovoo = 0.0
    for task_id, tauT_tmp, p0, p1 in _rotate_vir_block(tauT, vlocs=vlocs):
        #wovoo += einsum('icdb, dbjk -> icjk', eris.oxvv[:, :, p0:p1], tauT_tmp)
        wovoo -= einsum('cidb, dbjk -> icjk', eris.xovv[:, :, p0:p1], tauT_tmp)
        tauT_tmp = None
    wovoo *= 0.25

    #wovoo += np.asarray(eris.ooox.transpose(2, 3, 0, 1)) * 0.5
    #wovoo += np.asarray(eris.xooo.transpose(1, 0, 3, 2)) * 0.5
    wovoo += np.asarray(eris.xooo.transpose(1, 0, 2, 3)) * (-0.5)
    wovoo += einsum('icbk, bj -> icjk', v4, t1T)

    tauT *= 0.25
    #eris_vooo = eris.ooox.transpose(3, 0, 1, 2)
    eris_vooo = eris.xooo.transpose(0, 3, 2, 1)
    for task_id, eri_tmp, p0, p1 in _rotate_vir_block(eris_vooo, vlocs=vlocs):
        wovoo -= einsum('blij, cbkl -> icjk', eri_tmp, t2T[:, p0:p1])
        imds.wvvvo[:, :, p0:p1] = einsum('bcjl, ajlk -> bcak', tauT, eri_tmp)
        eri_tmp = None
    eris_vooo = None
    imds.wovoo[:] = wovoo
    wovoo = None
    tauT = None

    v4 = v4.transpose(1, 0, 2, 3)  
    for task_id, v4_tmp, p0, p1 in _rotate_vir_block(v4, vlocs=vlocs):
        imds.wvvvo[:, p0:p1] += einsum('bj, cjak -> bcak', t1T[vloc0:vloc1], v4_tmp)
        v4_tmp = None
    v4 = None

    #wvvvo = np.asarray(eris.ovvx).conj().transpose(3, 2, 1, 0) * 0.5
    wvvvo = np.asarray(eris.xvvo) * 0.5
    #eris_ovvv = eris.oxvv
    #for task_id, t2T_tmp, p0, p1 in _rotate_vir_block(t2T, vlocs=vlocs):
    #    wvvvo[:, p0:p1] -= einsum('kbad, cdjk -> bcaj', eris_ovvv, t2T_tmp)
    #    t2T_tmp = None
    eris_vovv = eris.xovv
    for task_id, t2T_tmp, p0, p1 in _rotate_vir_block(t2T, vlocs=vlocs):
        wvvvo[:, p0:p1] -= einsum('bkda, cdjk -> bcaj', eris_vovv, t2T_tmp)
        t2T_tmp = None

    imds.wvvvo -= wvvvo
    wvvvo = None
    
    imds.v1 = v1
    imds.v2 = v2
    imds.w3 = w3
    imds.ftmp.flush()
    return imds

# update L1, L2
def update_lambda(mycc, t1, t2, l1, l2, eris, imds):
    """
    Update GCCSD lambda.
    """
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1T = t1.T
    t2T = np.asarray(t2.transpose(2, 3, 0, 1), order='C')
    t1 = t2 = None
    nvir_seg, nvir, nocc = t2T.shape[:3]
    l1T = l1.T
    l2T = np.asarray(l2.transpose(2, 3, 0, 1), order='C')
    l1 = l2 = None

    ntasks = mpi.pool.size
    vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vloc0, vloc1 = vlocs[rank]
    log.debug2('vlocs %s', vlocs)
    assert vloc1 - vloc0 == nvir_seg
    
    fvo = eris.fock[nocc:, :nocc]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    v1 = imds.v1 - np.diag(mo_e_v)
    v2 = imds.v2 - np.diag(mo_e_o)

    mba = einsum('cakl, cbkl -> ba', l2T, t2T) * 0.5
    mba = mpi.allreduce_inplace(mba)
    mij = einsum('cdki, cdkj -> ij', l2T, t2T) * 0.5
    mij = mpi.allreduce_inplace(mij)
    # m3 [a]bij
    m3  = einsum('abkl, ijkl -> abij', l2T, np.asarray(imds.woooo))
    
    tauT = t2T + np.einsum('ai, bj -> abij', t1T[vloc0:vloc1] * 2.0, t1T, optimize=True)
    tmp = einsum('cdij, cdkl -> ijkl', l2T, tauT)
    tmp = mpi.allreduce_inplace(tmp)
    tauT = None
    
    vvoo = np.asarray(eris.xvoo)
    tmp = einsum('abkl, ijkl -> abij', vvoo, tmp)
    tmp *= 0.25
    m3  += tmp
    tmp = None
    tmp  = einsum('cdij, dk -> ckij', l2T, t1T)
    for task_id, tmp, p0, p1 in _rotate_vir_block(tmp, vlocs=vlocs):
        #m3 -= einsum('kcba, ckij -> abij', eris.ovvx[:, p0:p1], tmp)
        m3 -= einsum('abck, ckij -> abij', eris.xvvo[:, :, p0:p1], tmp)
        tmp = None
    eris_vvvv = eris.xvvv.transpose(2, 3, 0, 1)
    tmp_2 = np.empty_like(l2T) # used for line 387
    for task_id, l2T_tmp, p0, p1 in _rotate_vir_block(l2T, vlocs=vlocs):
        tmp = einsum('cdij, cdab -> abij', l2T_tmp, eris_vvvv[p0:p1])
        tmp *= 0.5
        m3 += tmp
        tmp_2[:, p0:p1] = einsum('acij, cb -> baij', l2T_tmp, v1[:, vloc0:vloc1])
        tmp = l2T_tmp = None
    eris_vvvv = None
    
    l1Tnew = einsum('abij, bj -> ai', m3, t1T)
    l1Tnew = mpi.allgather(l1Tnew)
    l2Tnew = m3
    
    l2Tnew += vvoo
    fvo1 = fvo + mpi.allreduce_inplace(einsum('cbkj, ck -> bj', vvoo, t1T[vloc0:vloc1]))
    
    tmp  = np.einsum('ai, bj -> abij', l1T[vloc0:vloc1], fvo1, optimize=True)
    wvovo = np.asarray(imds.wovvo).transpose(1, 0, 2, 3)
    for task_id, w_tmp, p0, p1 in _rotate_vir_block(wvovo, vlocs=vlocs):
        tmp -= einsum('acki, cjbk -> abij', l2T[:, p0:p1], w_tmp)
        w_tmp = None
    wvovo = None
    tmp  = tmp - tmp.transpose(0, 1, 3, 2)
    l2Tnew += tmp
    tmpT = mpi.alltoall_new([tmp[:, p0:p1] for p0, p1 in vlocs],
                            split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        l2Tnew[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None
    
    #tmp  = einsum('ak, ijkb -> baij', l1T, eris.ooox)
    tmp  = einsum('ak, bkji -> baij', l1T, eris.xooo)
    tmp -= tmp_2
    tmp1vv = mba + np.dot(t1T, l1T.T) # ba

    tmp -= einsum('ca, bcij -> baij', tmp1vv, vvoo)
    l2Tnew += tmp
    tmpT = mpi.alltoall_new([tmp[:, p0:p1] for p0, p1 in vlocs],
                            split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        l2Tnew[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None
    
    #tmp  = einsum('jcab, ci -> baji', eris.ovvx, -l1T)
    tmp  = einsum('bacj, ci -> baji', eris.xvvo, -l1T)
    tmp += einsum('abki, jk -> abij', l2T, v2)
    tmp1oo = mij + np.dot(l1T.T, t1T) # ik
    tmp -= einsum('ik, abkj -> abij', tmp1oo, vvoo)
    l2Tnew += tmp
    l2Tnew -= tmp.transpose(0, 1, 3, 2)
    tmp = None

    l1Tnew += fvo
    #tmp = einsum('bj, ibja -> ai', -l1T[vloc0:vloc1], eris.oxov)
    tmp = einsum('bj, biaj -> ai', -l1T[vloc0:vloc1], eris.xovo)
    l1Tnew += np.dot(v1.T, l1T)
    l1Tnew -= np.dot(l1T, v2.T)
    tmp -= einsum('cakj, icjk -> ai', l2T, imds.wovoo)
    tmp -= einsum('bcak, bcik -> ai', imds.wvvvo, l2T)
    tmp += einsum('baji, bj -> ai', l2T, imds.w3[vloc0:vloc1])
    
    tmp_2  = t1T[vloc0:vloc1] - np.dot(tmp1vv[vloc0:vloc1], t1T)
    tmp_2 -= np.dot(t1T[vloc0:vloc1], mij)
    tmp_2 += einsum('bcjk, ck -> bj', t2T, l1T)

    tmp += einsum('baji, bj -> ai', vvoo, tmp_2)
    tmp_2 = None

    #tmp += einsum('icab, bc -> ai', eris.oxvv, tmp1vv[:, vloc0:vloc1])
    tmp += einsum('ciba, bc -> ai', eris.xovv, tmp1vv[:, vloc0:vloc1])
    l1Tnew += mpi.allreduce_inplace(tmp)
    #l1Tnew -= mpi.allgather(einsum('jika, kj -> ai', eris.ooox, tmp1oo))
    l1Tnew -= mpi.allgather(einsum('akij, kj -> ai', eris.xooo, tmp1oo))
    
    tmp = fvo - mpi.allreduce_inplace(einsum('bakj, bj -> ak', vvoo, t1T[vloc0:vloc1]))
    vvoo = None
    l1Tnew -= np.dot(tmp, mij.T)
    l1Tnew -= np.dot(mba.T, tmp)
    
    eia = mo_e_o[:, None] - mo_e_v
    l1Tnew /= eia.T
    for i in range(vloc0, vloc1):
        l2Tnew[i-vloc0] /= lib.direct_sum('i + jb -> bij', eia[:, i], eia)

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1Tnew.T, l2Tnew.transpose(2, 3, 0, 1)

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    import gccsd

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run()
    mf0 = mf
    mf = scf.addons.convert_to_ghf(mf)
    mycc = gccsd.GCCSD(mf)
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    l1, l2 = mycc.solve_lambda(mycc.t1, mycc.t2, eris=eris)
    l1 = mycc.spin2spatial(l1, mycc.mo_coeff.orbspin)
    l2 = mycc.spin2spatial(l2, mycc.mo_coeff.orbspin)
    print(lib.finger(l1[0]) --0.0030030170069977758)
    print(lib.finger(l1[1]) --0.0030030170069977758)
    print(lib.finger(l2[0]) --0.041444910588788492 )
    print(lib.finger(l2[1]) - 0.1077575086912813   )
    print(lib.finger(l2[2]) --0.041444910588788492 )
    print(abs(l2[1]-l2[1].transpose(1,0,2,3)-l2[0]).max())
    print(abs(l2[1]-l2[1].transpose(0,1,3,2)-l2[0]).max())

    from pyscf.cc import ccsd
    mycc0 = ccsd.CCSD(mf0)
    eris0 = mycc0.ao2mo()
    mycc0.kernel(eris=eris0)
    t1 = mycc0.t1
    t2 = mycc0.t2
    imds = ccsd_lambda.make_intermediates(mycc0, t1, t2, eris0)
    l1, l2 = ccsd_lambda.update_lambda(mycc0, t1, t2, t1, t2, eris0, imds)
    l1ref, l2ref = ccsd_lambda.update_lambda(mycc0, t1, t2, l1, l2, eris0, imds)
    t1 = mycc.spatial2spin(t1, mycc.mo_coeff.orbspin)
    t2 = mycc.spatial2spin(t2, mycc.mo_coeff.orbspin)
    l1 = mycc.spatial2spin(l1, mycc.mo_coeff.orbspin)
    l2 = mycc.spatial2spin(l2, mycc.mo_coeff.orbspin)
    imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    l1 = mycc.spin2spatial(l1, mycc.mo_coeff.orbspin)
    l2 = mycc.spin2spatial(l2, mycc.mo_coeff.orbspin)
    print(abs(l1[0]-l1ref).max())
    print(abs(l2[1]-l2ref).max())
