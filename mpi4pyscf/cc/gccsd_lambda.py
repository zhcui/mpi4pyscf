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

import time
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

@mpi.parallel_call(skip_args=[1], skip_kwargs=['eris'])
def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-6, verbose=None,
           fintermediates=None, fupdate=None):
    """
    CCSD lambda kernel.
    """
    log = logger.new_logger(mycc, verbose)
    cput1 = cput0 = (time.clock(), time.time())
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
        l1 = t1
    if l2 is None:
        l2 = t2
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
    
    #tau = t2 + np.einsum('ia, jb -> ijab', t1, t1 * 2.0)
    tauT = np.einsum('ai, bj -> abij', t1T[vloc0:vloc1] * 2.0, t1T)
    tauT += t2T

    #v1  = fvv - np.dot(t1.T, fov)
    v1  = fvv - np.dot(t1T, fov)
    #v1 -= lib.einsum('jbac, jc -> ba', eris.ovvv, t1)
    #v1 += lib.einsum('jkca, jkbc -> ba', eris.oovv, tau) * 0.5
    tmp = lib.einsum('jbac, cj -> ba', eris.oxvv, t1T)
    v4 = np.zeros((nocc, nvir_seg, nvir, nocc))
    eris_voov = np.empty((nvir_seg, nocc, nocc, nvir))
    
    eris_voov[:] = eris.xvoo.transpose(0, 2, 3, 1)
    for task_id, eri_tmp, p0, p1 in _rotate_vir_block(eris_voov, vlocs=vlocs):
        tmp -= lib.einsum('cjka, bcjk -> ba', eri_tmp, tauT[:, p0:p1]) * 0.5
        v4  += lib.einsum('dljb, cdkl -> jcbk', eri_tmp, t2T[:, p0:p1])
        eri_tmp = None
    eris_voov = None
    v1 -= mpi.allgather(tmp)
    
    #v2  = foo + np.dot(fov, t1.T)
    v2  = foo + np.dot(fov, t1T)

    #v2 -= lib.einsum('kijb, kb -> ij', eris.ooov, t1)
    #v2 += lib.einsum('ikbc, jkbc -> ij', eris.oovv, tau) * 0.5
    tmp  = lib.einsum('kijb, bk -> ij', eris.ooox, t1T[vloc0:vloc1])
    tmp -= lib.einsum('bcik, bcjk -> ij', eris.xvoo, tauT) * 0.5
    v2  -= mpi.allreduce(tmp)
    
    ##v3  = lib.einsum('ijcd, klcd -> ijkl', eris.oovv, tau)
    #v4  = lib.einsum('ljdb, klcd -> jcbk', eris.oovv, t2)
    #v4 += np.asarray(eris.ovvo)
    # ZHC NOTE see line 128
    v4 -= np.asarray(eris.oxov.transpose(0, 1, 3, 2))
    
    #v5  = fvo + lib.einsum('kc, jkbc -> bj', fov, t2)
    #tmp = fov - lib.einsum('kldc, ld -> kc', eris.oovv, t1)
    #v5 += np.einsum('kc, kb, jc -> bj', tmp, t1, t1, optimize=True)
    # ZHC NOTE which one is better?
    #v5  = fvo + mpi.allgather(lib.einsum('kc, bcjk -> bj', fov, t2T))
    #tmp = fvo + mpi.allgather(lib.einsum('klcd, dl -> ck', eris.oovv, t1T))
    #v5 += np.einsum('ck, bk, cj -> bj', tmp, t1T, t1T, optimize=True)
    v5  = fvo + mpi.allgather(lib.einsum('kc, bcjk -> bj', fov, t2T))
    tmp = fvo[vloc0:vloc1] + lib.einsum('cdkl, dl -> ck', eris.xvoo, t1T)
    v5 += mpi.allreduce(np.einsum('ck, bk, cj -> bj', tmp, t1T, t1T[vloc0:vloc1], optimize=True))
     
    #v5 -= lib.einsum('kljc, klbc -> bj', eris.ooov, t2) * 0.5
    v5 += mpi.allreduce(lib.einsum('kljc, cbkl -> bj', eris.ooox, t2T)) * 0.5
    #v5 += lib.einsum('kbdc, jkcd -> bj', eris.ovvv, t2) * 0.5
    tmp = np.zeros((nvir_seg, nocc))
    for task_id, t2T_tmp, p0, p1 in _rotate_vir_block(t2T, vlocs=vlocs):
        tmp += lib.einsum('kbcd, cdjk -> bj', eris.oxvv[:, :, p0:p1], t2T_tmp)
        t2T_tmp = None
    tmp *= 0.5
    tmp  = mpi.allgather(tmp)
    v5  -= tmp

    #w3  = v5 + lib.einsum('jcbk, jb -> ck', v4, t1)
    #w3 += lib.einsum('cb, jb -> cj', v1, t1)
    #w3 -= lib.einsum('jk, jb -> bk', v2, t1)
    # v4 j[c]bk, bj -> [c]k
    w3  = v5[vloc0:vloc1] + lib.einsum('jcbk, bj -> ck', v4, t1T)
    w3 += np.dot(v1[vloc0:vloc1], t1T)
    w3 -= np.dot(t1T[vloc0:vloc1], v2)
    w3  = mpi.allgather(w3)

    #woooo  = np.asarray(eris.oooo) * 0.5
    #woooo += lib.einsum('ijcd, klcd -> ijkl', eris.oovv, tau) * 0.25
    #woooo += lib.einsum('jilc, kc -> jilk', eris.ooov, t1)
    #imds.woooo[:] = woooo
    #woooo = None
    woooo  = lib.einsum('cdij, cdkl -> ijkl', eris.xvoo, tauT) * 0.25
    woooo += lib.einsum('jilc, ck -> jilk', eris.ooox, t1T[vloc0:vloc1])
    woooo  = mpi.allreduce(woooo)
    woooo += np.asarray(eris.oooo) * 0.5
    imds.woooo[:] = woooo
    woooo = None
    
    #wovvo  = v4 - np.einsum('ljdb, lc, kd -> jcbk', eris.oovv, t1, t1, optimize=True)
    #wovvo -= lib.einsum('ljkb, lc -> jcbk', eris.ooov, t1)
    #wovvo += lib.einsum('jcbd, kd -> jcbk', eris.ovvv, t1)
    #imds.wovvo[:] = wovvo
    #wovvo = None
    # ZHC NOTE: wovvo, v4 has shape j[c]bk
    wovvo = v4 + lib.einsum('jcbd, dk -> jcbk', eris.oxvv, t1T)
    
    tmp = lib.einsum('bdlj, dk -> bklj', eris.xvoo, t1T)
    for task_id, tmp_2, p0, p1 in _rotate_vir_block(tmp, vlocs=vlocs):
        wovvo[:, :, p0:p1] += lib.einsum('bklj, cl -> jcbk', tmp_2, t1T[vloc0:vloc1])
        tmp_2 = None
    tmp = None
    
    eris_vooo = np.empty((nvir_seg, nocc, nocc, nocc))
    eris_vooo[:] = eris.ooox.transpose(3, 2, 1, 0)
    for task_id, eri_tmp, p0, p1 in _rotate_vir_block(eris_vooo, vlocs=vlocs):
        wovvo[:, :, p0:p1] -= lib.einsum('bkjl, cl -> jcbk', eri_tmp, t1T[vloc0:vloc1])
        eri_tmp = None
    eris_vooo = None
    imds.wovvo[:] = wovvo
    wovvo = None

    #wovoo  = lib.einsum('icdb, jkdb -> icjk', eris.ovvv, tau) * 0.25
    #wovoo += np.asarray(eris.ooov).conj().transpose(2, 3, 0, 1) * 0.5
    #wovoo += lib.einsum('icbk, jb -> icjk', v4, t1)
    #wovoo -= lib.einsum('lijb, klcb -> icjk', eris.ooov, t2)
    #imds.wovoo[:] = wovoo
    #wovoo = None
    wovoo = np.zeros((nocc, nvir_seg, nocc, nocc))
    for task_id, tauT_tmp, p0, p1 in _rotate_vir_block(tauT, vlocs=vlocs):
        wovoo += lib.einsum('icdb, dbjk -> icjk', eris.oxvv[:, :, p0:p1], tauT_tmp)
        tauT_tmp = None
    wovoo *= 0.25

    wovoo += np.asarray(eris.ooox.transpose(2, 3, 0, 1)) * 0.5
    wovoo += lib.einsum('icbk, bj -> icjk', v4, t1T)

    tauT *= 0.25
    eris_vooo = np.empty((nvir_seg, nocc, nocc, nocc))
    eris_vooo[:] = eris.ooox.transpose(3, 0, 1, 2)
    #eris_vooo = eris["ooox"].transpose(3, 0, 1, 2)
    for task_id, eri_tmp, p0, p1 in _rotate_vir_block(eris_vooo, vlocs=vlocs):
        wovoo -= lib.einsum('blij, cbkl -> icjk', eri_tmp, t2T[:, p0:p1])
        imds.wvvvo[:, :, p0:p1] = lib.einsum('bcjl, ajlk -> bcak', tauT, eri_tmp)
        eri_tmp = None
    eris_vooo = None
    imds.wovoo[:] = wovoo
    wovoo = None
    tauT = None

    #wvvvo  = lib.einsum('jcak, jb -> bcak', v4, t1)
    #v4 = None
    #wvvvo += lib.einsum('jlka, jlbc -> bcak', np.asarray(eris.ooov) * 0.25, tau)
    #wvvvo -= np.asarray(eris.ovvv).conj().transpose(3, 2, 1, 0) * 0.5
    #wvvvo += lib.einsum('kbad, jkcd -> bcaj', eris.ovvv, t2)
    #imds.wvvvo[:] = wvvvo
    #wvvvo = None
    
    # [c]bak
    #wvvvo  = lib.einsum('bj, jcak -> cbak', t1T, v4)
    v4 = v4.transpose(1, 0, 2, 3)  
    for task_id, v4_tmp, p0, p1 in _rotate_vir_block(v4, vlocs=vlocs):
        imds.wvvvo[:, p0:p1] += lib.einsum('bj, cjak -> bcak', t1T[vloc0:vloc1], v4_tmp)
        v4_tmp = None
    v4 = None
    # line 238
    #wvvvo += lib.einsum('jlka, jlbc -> bcak', np.asarray(eris.ooov) * 0.25, tau)

    wvvvo = np.asarray(eris.ovvx).conj().transpose(3, 2, 1, 0) * 0.5
    
    #wvvvo += lib.einsum('kbad, cdjk -> bcaj', eris.ovvv, t2T)
    eris_ovvv = eris.oxvv
    for task_id, t2T_tmp, p0, p1 in _rotate_vir_block(t2T, vlocs=vlocs):
        wvvvo[:, p0:p1] -= lib.einsum('kbad, cdjk -> bcaj', eris_ovvv, t2T_tmp)
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
    time1 = time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    cpu1 = time0

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

    #mba = lib.einsum('klca, klcb -> ba', l2, t2) * 0.5
    #mij = lib.einsum('kicd, kjcd -> ij', l2, t2) * 0.5
    #m3  = lib.einsum('klab, ijkl -> ijab', l2, np.asarray(imds.woooo))
    mba = lib.einsum('cakl, cbkl -> ba', l2T, t2T) * 0.5
    mba = mpi.allreduce(mba)
    mij = lib.einsum('cdki, cdkj -> ij', l2T, t2T) * 0.5
    mij = mpi.allreduce(mij)
    # m3 [a]bij
    m3  = lib.einsum('abkl, ijkl -> abij', l2T, np.asarray(imds.woooo))
    
    #tau = t2 + np.einsum('ia, jb -> ijab', t1, t1 * 2.0)
    #tmp = lib.einsum('ijcd, klcd -> ijkl', l2, tau)
    #tau = None
    tauT = t2T + np.einsum('ai, bj -> abij', t1T[vloc0:vloc1] * 2.0, t1T)
    tmp = lib.einsum('cdij, cdkl -> ijkl', l2T, tauT)
    tmp = mpi.allreduce(tmp)
    tauT = None
    
    #oovv = np.asarray(eris.oovv)
    #m3  += lib.einsum('klab, ijkl -> ijab', oovv, tmp) * 0.25
    #tmp  = lib.einsum('ijcd, kd -> ijck', l2, t1)
    #m3  -= lib.einsum('kcba, ijck -> ijab', eris.ovvv, tmp)
    #tmp  = None
    #m3  += lib.einsum('ijcd, cdab -> ijab', l2, eris.vvvv) * 0.5
    vvoo = np.asarray(eris.xvoo)
    m3  += lib.einsum('abkl, ijkl -> abij', vvoo, tmp) * 0.25
    tmp  = lib.einsum('cdij, dk -> ckij', l2T, t1T)
    for task_id, tmp, p0, p1 in _rotate_vir_block(tmp, vlocs=vlocs):
        m3 -= lib.einsum('kcba, ckij -> abij', eris.ovvx[:, p0:p1], tmp)
        tmp = None
    eris_vvvv = eris.xvvv.transpose(2, 3, 0, 1)
    tmp_2 = np.empty_like(l2T) # used for line 387
    for task_id, l2T_tmp, p0, p1 in _rotate_vir_block(l2T, vlocs=vlocs):
        m3 += lib.einsum('cdij, cdab -> abij', l2T_tmp, eris_vvvv[p0:p1]) * 0.5
        tmp_2[:, p0:p1] = lib.einsum('acij, cb -> baij', l2T_tmp, v1[:, vloc0:vloc1])
        l2T_tmp = None
    eris_vvvv = None
    
    #l1new = lib.einsum('ijab, jb -> ia', m3, t1)
    #l2new = m3
    l1Tnew = lib.einsum('abij, bj -> ai', m3, t1T)
    l1Tnew = mpi.allgather(l1Tnew)
    l2Tnew = m3
    
    #l2new += oovv
    #fov1 = fov + lib.einsum('kjcb, kc -> jb', oovv, t1)
    l2Tnew += vvoo
    fvo1 = fvo + mpi.allreduce(lib.einsum('cbkj, ck -> bj', vvoo, t1T[vloc0:vloc1]))
    
    #tmp  = np.einsum('ia, jb -> ijab', l1, fov1)
    #tmp += lib.einsum('kica, jcbk -> ijab', l2, np.asarray(imds.wovvo))
    #tmp  = tmp - tmp.transpose(1, 0, 2, 3)
    #l2new += tmp
    #l2new -= tmp.transpose(0, 1, 3, 2)
    tmp  = np.einsum('ai, bj -> abij', l1T[vloc0:vloc1], fvo1)
    wvovo = np.empty((nvir_seg, nocc, nvir, nocc))
    wvovo[:] = np.asarray(imds.wovvo).transpose(1, 0, 2, 3)
    for task_id, w_tmp, p0, p1 in _rotate_vir_block(wvovo, vlocs=vlocs):
        tmp -= lib.einsum('acki, cjbk -> abij', l2T[:, p0:p1], w_tmp)
        w_tmp = None
    wvovo = None
    tmp  = tmp - tmp.transpose(0, 1, 3, 2)
    l2Tnew += tmp
    tmpT = mpi.alltoall([tmp[:, p0:p1] for p0, p1 in vlocs],
                        split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        l2Tnew[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None
    
    #tmp  = lib.einsum('ka, ijkb -> ijab', l1, eris.ooov)
    #tmp += lib.einsum('ijca, cb -> ijab', l2, v1)
    #tmp1vv = mba + lib.einsum('ka, kb -> ba', l1, t1)
    #tmp += lib.einsum('ca, ijcb -> ijab', tmp1vv, oovv)
    #l2new -= tmp
    #l2new += tmp.transpose(0,1,3,2)
    tmp  = lib.einsum('ak, ijkb -> baij', l1T, eris.ooox)
    # see line 343
    #tmp -= lib.einsum('acij, cb -> abij', l2T, v1)
    tmp -= tmp_2
    tmp1vv = mba + np.dot(t1T, l1T.T) # ba

    tmp -= lib.einsum('ca, bcij -> baij', tmp1vv, vvoo)
    l2Tnew += tmp
    tmpT = mpi.alltoall([tmp[:, p0:p1] for p0, p1 in vlocs],
                        split_recvbuf=True)
    for task_id, (p0, p1) in enumerate(vlocs):
        tmp = tmpT[task_id].reshape(p1-p0, nvir_seg, nocc, nocc)
        l2Tnew[:, p0:p1] -= tmp.transpose(1, 0, 2, 3)
        tmp = None
    
    #tmp  = lib.einsum('ic, jcba -> jiba', l1, eris.ovvv)
    #tmp += lib.einsum('kiab, jk -> ijab', l2, v2)
    #tmp1oo = mij + lib.einsum('ic, kc -> ik', l1, t1)
    #tmp -= lib.einsum('ik, kjab -> ijab', tmp1oo, oovv)
    #l2new += tmp
    #l2new -= tmp.transpose(1, 0, 2, 3)
    #tmp = None
    
    # bajc jcab
    # xvov -ovvx
    tmp  = lib.einsum('jcab, ci -> baji', eris.ovvx, -l1T)
    tmp += lib.einsum('abki, jk -> abij', l2T, v2)
    tmp1oo = mij + np.dot(l1T.T, t1T) # ik
    tmp -= lib.einsum('ik, abkj -> abij', tmp1oo, vvoo)
    l2Tnew += tmp
    l2Tnew -= tmp.transpose(0, 1, 3, 2)
    tmp = None

    #l1new += fov
    #l1new += lib.einsum('jb, ibaj -> ia', l1, eris.ovvo)
    #l1new += lib.einsum('ib, ba -> ia', l1, v1)
    #l1new -= lib.einsum('ja, ij -> ia', l1, v2)
    #l1new -= lib.einsum('kjca, icjk -> ia', l2, imds.wovoo)
    #l1new -= lib.einsum('ikbc, bcak -> ia', l2, imds.wvvvo)
    #l1new += lib.einsum('jiba, bj -> ia', l2, imds.w3)
    #tmp = (t1 + lib.einsum('kc, kjcb -> jb', l1, t2)
    #          - lib.einsum('bd, jd -> jb', tmp1vv, t1)
    #          - lib.einsum('lj, lb -> jb', mij, t1))
    #l1new += lib.einsum('jiba, jb -> ia', oovv, tmp)
    #l1new += lib.einsum('icab, bc -> ia', eris.ovvv, tmp1vv)
    #l1new -= lib.einsum('jika, kj -> ia', eris.ooov, tmp1oo)
    #tmp = fov - lib.einsum('kjba, jb -> ka', oovv, t1)
    #vvoo = None
    #l1new -= np.dot(mij, tmp)
    #l1new -= np.dot(tmp, mba)
    #l1Tnew = 0.0
    l1Tnew += fvo
    #l1Tnew += mpi.allreduce(lib.einsum('bj, ibaj -> ai', l1T[vloc0:vloc1], eris["oxvo"]))
    # oxvo oxov
    # ibaj ibja
    tmp = lib.einsum('bj, ibja -> ai', -l1T[vloc0:vloc1], eris.oxov)
    l1Tnew += np.dot(v1.T, l1T)
    l1Tnew -= np.dot(l1T, v2.T)
    tmp -= lib.einsum('cakj, icjk -> ai', l2T, imds.wovoo)
    tmp -= lib.einsum('bcak, bcik -> ai', imds.wvvvo, l2T)
    tmp += lib.einsum('baji, bj -> ai', l2T, imds.w3[vloc0:vloc1])
    
    tmp_2  = t1T[vloc0:vloc1] - np.dot(tmp1vv[vloc0:vloc1], t1T)
    tmp_2 -= np.dot(t1T[vloc0:vloc1], mij)
    tmp_2 += lib.einsum('bcjk, ck -> bj', t2T, l1T)

    tmp += lib.einsum('baji, bj -> ai', vvoo, tmp_2)
    tmp_2 = None

    tmp += lib.einsum('icab, bc -> ai', eris.oxvv, tmp1vv[:, vloc0:vloc1])
    l1Tnew += mpi.allreduce(tmp)
    l1Tnew -= mpi.allgather(lib.einsum('jika, kj -> ai', eris.ooox, tmp1oo))
    
    tmp = fvo - mpi.allreduce(lib.einsum('bakj, bj -> ak', vvoo, t1T[vloc0:vloc1]))
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