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
from mpi4pyscf.cc.ccsd import (_task_location, _sync_, _diff_norm)
from mpi4pyscf.cc.gccsd_lambda import (make_intermediates, update_lambda, 
                                       update_lambda_rk)

comm = mpi.comm
rank = mpi.rank

einsum = lib.einsum

def fill_zero_cas(mycc, t1, t2):
    """
    Fill the t1 and t2 with 0 for cas part.
    inplace change t1 and t2.
    """
    t1T = t1.T
    t2T = np.asarray(t2.transpose(2, 3, 0, 1), order='C')
    nvir_seg, nvir, nocc = t2T.shape[:3]
    ntasks = mpi.pool.size
    ncore = mycc.ncore

    vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vloc0, vloc1 = vlocs[rank]
    assert vloc1 - vloc0 == nvir_seg
    
    nocc_cas, nvir_cas = mycc.t1_cas.shape

    t1T[:nvir_cas, ncore:] = 0.0
    if vloc0 < nvir_cas:
        end = min(vloc1, nvir_cas)
        t2T[:(end - vloc0), :nvir_cas, ncore:, ncore:] = 0.0

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
        mycc.l1 = np.array(l1, copy=True)
        mycc.l2 = np.array(l2, copy=True)
        fill_zero_cas(mycc, mycc.l1, mycc.l2)
        l1 = mycc.l1
        l2 = mycc.l2
        conv = True
        return conv, l1, l2
    else:
        raise NotImplementedError

    if fintermediates is None:
        fintermediates = make_intermediates
    
    if fupdate is None:
        rk = comm.allreduce(getattr(mycc, "rk", None), op=mpi.MPI.LOR)
        if rk:
            fupdate = update_lambda_rk
        else:
            fupdate = update_lambda

    imds = fintermediates(mycc, t1, t2, eris)

    if isinstance(mycc.diis, diis.DistributedDIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = diis.DistributedDIIS(mycc, mycc.diis_file)
        adiis.space = mycc.diis_space
    else:
        adiis = None
    cput1 = log.timer('CCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = fupdate(mycc, t1, t2, l1, l2, eris, imds)
        normt = _diff_norm(mycc, l1new, l2new, l1, l2)
        l1, l2 = l1new, l2new
        l1new = l2new = None
        l1, l2 = mycc.run_diis(l1, l2, istep, normt, 0, adiis)
        log.info('cycle = %d  norm(lambda1,lambda2) = %.6g', istep+1, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if normt < tol:
            conv = True
            break

    mycc.l1 = l1
    mycc.l2 = l2
    log.timer('CCSD lambda', *cput0)
    return conv, l1, l2


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
