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
#

"""
MPI paralleled get_j_kpoints, get_k_kpoints (parallel over kpoints).
"""

import h5py
import numpy as np
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import gamma_point, member
from pyscf.pbc import df
from pyscf.df import addons
from pyscf.pbc.df.df_jk import (_format_dms, _format_kpts_band, _format_jks,
                                zdotNN, zdotCN, zdotNC, _ewald_exxdiv_for_G0)

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
comm = mpi.comm
rank = mpi.rank

def _task_location(n, task=rank):
    neach, extras = divmod(n, mpi.pool.size)
    section_sizes = ([0] + extras * [neach+1] + (mpi.pool.size-extras) * [neach])
    div_points = np.cumsum(section_sizes)
    loc0 = div_points[task]
    loc1 = div_points[task + 1]
    return loc0, loc1

def assign_workload(kij_args, n): 
    idx_1 = []
    idx_2 = []
    for i, args in enumerate(kij_args):
        if args[2]:
            idx_2.append(i)
        else:
            idx_1.append(i)
    
    idx_1 = np.asarray(idx_1)
    idx_2 = np.asarray(idx_2)
    n_1 = len(idx_1)
    n_2 = len(idx_2)
    nibz = n_1 + n_2 
    
    klocs = [_task_location(nibz, task_id) for task_id in range(n)]
    ns = [j - i for i, j in klocs]

    kids = [[] for i in range(n)]
    # first assign 1
    for i, idx in enumerate(idx_1):
        kids[i%n].append(idx)

    start = 0 
    for i, kid in enumerate(kids):
        end = start + (ns[i] - len(kid))
        kid.extend(idx_2[start:end])
        start = end 
    return kids

def get_naoaux(gdf):
    """
    The maximum dimension of auxiliary basis for every k-point.
    """
    assert gdf._cderi is not None
    with h5py.File(gdf._cderi, 'r') as f:
        nkptij = f["j3c-kptij"].shape[0]
    naux_k_list = []
    for k in range(nkptij):
        # gdf._cderi['j3c/k_id/seg_id']
        with addons.load(gdf._cderi, 'j3c/%s'%k) as feri:
            if isinstance(feri, h5py.Group):
                naux_k = feri['0'].shape[0]
            else:
                naux_k = feri.shape[0]

        cell = gdf.cell
        if (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum' and
            not isinstance(gdf._cderi, np.ndarray)):
            with h5py.File(gdf._cderi, 'r') as feri:
                if 'j3c-/%s'%k in feri:
                    dat = feri['j3c-/%s'%k]
                    if isinstance(dat, h5py.Group):
                        naux_k += dat['0'].shape[0]
                    else:
                        naux_k += dat.shape[0]
        naux_k_list.append(naux_k)

    naux = np.max(naux_k_list)
    return naux

@mpi.parallel_call
def get_j_kpts(cell, cderi, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
    log = logger.Logger(cell.stdout, cell.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())
    
    # ZHC NOTE
    # mydf is not allowed to pass as argument in the MPI code.
    mydf = df.GDF(cell, kpts)
    mydf._cderi = cderi
    
    if mydf._cderi is None or not mydf.has_kpts(kpts_band):
        if mydf._cderi is not None:
            log.warn('DF integrals for band k-points were not found %s. '
                     'DF integrals will be rebuilt to include band k-points.',
                     mydf._cderi)
        mydf.build(kpts_band=kpts_band)
        t1 = log.timer_debug1('Init get_j_kpts', *t1)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    if mydf.auxcell is None:
        # If mydf._cderi is the file that generated from another calculation,
        # guess naux based on the contents of the integral file.
        naux = get_naoaux(mydf)
    else:
        naux = mydf.auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    j_real = gamma_point(kpts_band) and not np.iscomplexobj(dms)
    
    # ZHC NOTE
    # first partition over kpts
    ntasks = mpi.pool.size
    klocs = [_task_location(nkpts, task_id) for task_id in range(ntasks)]
    kpts_ids_own = np.arange(*klocs[rank])

    dmsR = dms.real.transpose(0,1,3,2).reshape(nset,nkpts,nao**2)
    dmsI = dms.imag.transpose(0,1,3,2).reshape(nset,nkpts,nao**2)
    rhoR = np.zeros((nset,naux))
    rhoI = np.zeros((nset,naux))
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]))
    
    # ZHC NOTE weight is absorbed to sign
    weight = 1./nkpts
    #rhoR *= weight
    #rhoI *= weight
    
    log.alldebug1("kpts ids: %s", kpts_ids_own)
    #for k, kpt in enumerate(kpts):
    for k in kpts_ids_own:
        kpt = kpts[k]
        kptii = np.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI, sign in mydf.sr_loop(kptii, max_memory, False):
            p0, p1 = p1, p1+LpqR.shape[0]
            #:Lpq = (LpqR + LpqI*1j).reshape(-1,nao,nao)
            #:rhoR[:,p0:p1] += np.einsum('Lpq,xqp->xL', Lpq, dms[:,k]).real
            #:rhoI[:,p0:p1] += np.einsum('Lpq,xqp->xL', Lpq, dms[:,k]).imag
            rhoR[:,p0:p1] += (sign * weight) * np.einsum('Lp,xp->xL', LpqR, dmsR[:,k])
            rhoI[:,p0:p1] += (sign * weight) * np.einsum('Lp,xp->xL', LpqR, dmsI[:,k])
            if LpqI is not None:
                rhoR[:,p0:p1] -= (sign * weight) * np.einsum('Lp,xp->xL', LpqI, dmsI[:,k])
                rhoI[:,p0:p1] += (sign * weight) * np.einsum('Lp,xp->xL', LpqI, dmsR[:,k])
            LpqR = LpqI = None
    
    # ZHC NOTE allreduce the rho
    rhoR = mpi.allreduce_inplace(rhoR)
    rhoI = mpi.allreduce_inplace(rhoI)
    t1 = log.timer_debug1('get_j pass 1', *t1)
    
    # ZHC NOTE
    # then partition over kpts_band
    klocs_band = [_task_location(nband, task_id) for task_id in range(ntasks)]
    kpts_band_ids_own = np.arange(*klocs_band[rank])
    nband_now = len(kpts_band_ids_own)

    vjR = np.zeros((nset,nband_now,nao_pair))
    vjI = np.zeros((nset,nband_now,nao_pair))
    #for k, kpt in enumerate(kpts_band):
    log.alldebug1("kpts_band ids: %s", kpts_band_ids_own)
    for i, k in enumerate(kpts_band_ids_own):
        kpt = kpts_band[k]
        kptii = np.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI, sign in mydf.sr_loop(kptii, max_memory, True):
            p0, p1 = p1, p1+LpqR.shape[0]
            #:Lpq = (LpqR + LpqI*1j)#.reshape(-1,nao,nao)
            #:vjR[:,k] += np.dot(rho[:,p0:p1], Lpq).real
            #:vjI[:,k] += np.dot(rho[:,p0:p1], Lpq).imag
            vjR[:,i] += np.dot(rhoR[:,p0:p1], LpqR)
            if not j_real:
                vjI[:,i] += np.dot(rhoI[:,p0:p1], LpqR)
                if LpqI is not None:
                    vjR[:,i] -= np.dot(rhoI[:,p0:p1], LpqI)
                    vjI[:,i] += np.dot(rhoR[:,p0:p1], LpqI)
            LpqR = LpqI = None

    t1 = log.timer_debug1('get_j pass 2', *t1)

    if j_real:
        vj_kpts = vjR
    else:
        vj_kpts = vjR + vjI*1j
    vj_kpts = lib.unpack_tril(vj_kpts.reshape(-1,nao_pair))
    vj_kpts = vj_kpts.reshape(nset,nband_now,nao,nao)
    
    # ZHC NOTE gather
    vj_kpts = mpi.gather_new(vj_kpts.transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3)
    if rank == 0:
        vj_kpts = _format_jks(vj_kpts, dm_kpts, input_band, kpts)
    return vj_kpts

@mpi.parallel_call
def get_k_kpts(cell, cderi, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    log = logger.Logger(cell.stdout, cell.verbose)
    
    # ZHC NOTE
    # mydf is not allowed to pass as argument in the MPI code.
    mydf = df.GDF(cell, kpts)
    mydf._cderi = cderi

    if exxdiv is not None and exxdiv != 'ewald':
        log.warn('GDF does not support exxdiv %s. '
                 'exxdiv needs to be "ewald" or None', exxdiv)
        raise RuntimeError('GDF does not support exxdiv %s' % exxdiv)

    t1 = (logger.process_clock(), logger.perf_counter())
    if mydf._cderi is None or not mydf.has_kpts(kpts_band):
        if mydf._cderi is not None:
            log.warn('DF integrals for band k-points were not found %s. '
                     'DF integrals will be rebuilt to include band k-points.',
                     mydf._cderi)
        mydf.build(kpts_band=kpts_band)
        t1 = log.timer_debug1('Init get_k_kpts', *t1)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    vkR = np.zeros((nset,nband,nao,nao))
    vkI = np.zeros((nset,nband,nao,nao))
    dmsR = np.asarray(dms.real, order='C')
    dmsI = np.asarray(dms.imag, order='C')

    # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )
    bufR = np.empty((mydf.blockdim*nao**2))
    bufI = np.empty((mydf.blockdim*nao**2))
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    
    def make_kpt(ki, kj, swap_2e, inverse_idx=None):
        kpti = kpts[ki]
        kptj = kpts_band[kj]

        for LpqR, LpqI, sign in mydf.sr_loop((kpti,kptj), max_memory, False):
            nrow = LpqR.shape[0]
            pLqR = np.ndarray((nao,nrow,nao), buffer=bufR)
            pLqI = np.ndarray((nao,nrow,nao), buffer=bufI)
            tmpR = np.ndarray((nao,nrow*nao), buffer=LpqR)
            tmpI = np.ndarray((nao,nrow*nao), buffer=LpqI)
            pLqR[:] = LpqR.reshape(-1,nao,nao).transpose(1,0,2)
            pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)

            for i in range(nset):
                zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                       pLqI.reshape(nao,-1), 1, tmpR, tmpI)
                zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                       tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                       sign / float(nkpts), vkR[i,kj], vkI[i,kj], 1)

            if swap_2e:
                tmpR = tmpR.reshape(nao*nrow,nao)
                tmpI = tmpI.reshape(nao*nrow,nao)
                ki_tmp = ki
                kj_tmp = kj
                if inverse_idx:
                    ki_tmp = inverse_idx[0]
                    kj_tmp = inverse_idx[1]
                for i in range(nset):
                    zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                           dmsR[i,kj_tmp], dmsI[i,kj_tmp], 1, tmpR, tmpI)
                    zdotNC(tmpR.reshape(nao,-1), tmpI.reshape(nao,-1),
                           pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                           sign / float(nkpts), vkR[i,ki_tmp], vkI[i,ki_tmp], 1)
    
    kij_args = []
    if kpts_band is kpts:  # normal k-points HF/DFT
        for ki in range(nkpts):
            for kj in range(ki):
                #make_kpt(ki, kj, True)
                kij_args.append((ki, kj, True))
            #make_kpt(ki, ki, False)
            kij_args.append((ki, ki, False))
            #t1 = log.timer_debug1('get_k_kpts: make_kpt ki>=kj (%d,*)'%ki, *t1)
    else:
        idx_in_kpts = []
        for kpt in kpts_band:
            idx = member(kpt, kpts)
            if len(idx) > 0:
                idx_in_kpts.append(idx[0])
            else:
                idx_in_kpts.append(-1)
        idx_in_kpts_band = []
        for kpt in kpts:
            idx = member(kpt, kpts_band)
            if len(idx) > 0:
                idx_in_kpts_band.append(idx[0])
            else:
                idx_in_kpts_band.append(-1)

        for ki in range(nkpts):
            for kj in range(nband):
                if idx_in_kpts[kj] == -1 or idx_in_kpts[kj] == ki:
                    #make_kpt(ki, kj, False)
                    kij_args.append((ki, kj, False))
                elif idx_in_kpts[kj] < ki:
                    if idx_in_kpts_band[ki] == -1:
                        #make_kpt(ki, kj, False)
                        kij_args.append((ki, kj, False))
                    else:
                        #make_kpt(ki, kj, True, (idx_in_kpts_band[ki], idx_in_kpts[kj]))
                        kij_args.append((ki, kj, True, (idx_in_kpts_band[ki], idx_in_kpts[kj])))
                else:
                    if idx_in_kpts_band[ki] == -1:
                        #make_kpt(ki, kj, False)
                        kij_args.append((ki, kj, False))
            #t1 = log.timer_debug1('get_k_kpts: make_kpt (%d,*)'%ki, *t1)

    ntasks = mpi.pool.size
    kij_ids = assign_workload(kij_args, ntasks)
    kij_id_own = kij_ids[rank]
    kij_args_own = [kij_args[i] for i in kij_id_own]
    log.alldebug1("kij pairs: %s", kij_args_own)

    for args in kij_args_own:
        make_kpt(*args)

    if (gamma_point(kpts) and gamma_point(kpts_band) and
        not np.iscomplexobj(dm_kpts)):
        vk_kpts = vkR
    else:
        vk_kpts = vkR + vkI * 1j
    # ZHC NOTE absorbed to sign
    #vk_kpts *= 1./nkpts
    
    vk_kpts = mpi.reduce_inplace(vk_kpts)

    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)


if __name__ == '__main__':
    pass
