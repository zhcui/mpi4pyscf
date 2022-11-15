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

import numpy as np
from scipy import optimize as opt
from scipy.sparse import linalg as spla
from pyscf import lib

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.ccsd import _sync_
from mpi4pyscf.cc.gccsd_lambda import (make_intermediates, update_lambda)
from mpi4pyscf.cc.gccsd_krylov import (make_precond_vec_finv, safe_max_abs)

comm = mpi.comm
rank = mpi.rank

einsum = lib.einsum

@mpi.parallel_call(skip_args=[1], skip_kwargs=['eris'])
def pre_kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
               max_cycle=50, tol=1e-6, verbose=None,
               fintermediates=None, fupdate=None, approx_l=False):
    """
    ao2mo, init l1 l2, imds, distribute vec, cycle = 0.
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
    
    # ZHC NOTE frozen abab
    if comm.allreduce(getattr(mycc, "frozen_abab", False), op=mpi.MPI.LOR):
        mycc.remove_t2_abab(t2)
        mycc.remove_t2_abab(l2)
    if comm.allreduce(getattr(mycc, "frozen_aaaa_bbbb", False), op=mpi.MPI.LOR):
        mycc.remove_t2_aaaa_bbbb(t2)
        mycc.remove_t2_aaaa_bbbb(l2)
    if getattr(mycc, "t1_frozen_list", None) or getattr(mycc, "t2_frozen_list", None):
        mycc.remove_amps(t1, t2, 
                         t1_frozen_list=mycc.t1_frozen_list,
                         t2_frozen_list=mycc.t2_frozen_list)
        mycc.remove_amps(l1, l2, 
                         t1_frozen_list=mycc.t1_frozen_list,
                         t2_frozen_list=mycc.t2_frozen_list)
    
    if approx_l:
        mycc.l1 = l1
        mycc.l2 = l2
        return mycc
    
    if fintermediates is None:
        fintermediates = make_intermediates
    
    imds = fintermediates(mycc, t1, t2, eris)
    mycc._imds = imds

    cput1 = log.timer('CCSD lambda initialization', *cput0)

    mycc.l1 = l1
    mycc.l2 = l2
    vec = mycc.amplitudes_to_vector(l1, l2)
    mycc.vec = mycc.gather_vector(vec)
    # initialize the precond vector
    #mycc.precond_vec = make_precond_vec_finv(mycc, l2, eris)
    mycc.cycle = 0
    return mycc

@mpi.parallel_call(skip_args=[1], skip_kwargs=['x'])
def get_lambda_res(mycc, x):
    """
    Get the residual vector of CC lambda.

    Args:
        x: vector of CC amps (at root).
    Returns:
        res: vector of residual (at root).
    """
    _sync_(mycc)
    log = logger.new_logger(mycc, mycc.verbose)
    eris = getattr(mycc, '_eris', None)
    imds = getattr(mycc, '_imds', None)
    t1, t2 = mycc.t1, mycc.t2
    
    # firs distribute x to l1 and l2
    vec = mycc.distribute_vector_(x, write='l')
    l1, l2 = mycc.l1, mycc.l2

    l1, l2 = update_lambda(mycc, t1, t2, l1, l2, eris, imds)

    # then gather the vector
    res = mycc.amplitudes_to_vector(l1, l2)
    norm = safe_max_abs(res)
    norm = comm.allreduce(norm, op=mpi.MPI.MAX)
    log.info("      cycle = %5d , norm(res) = %15.5g", mycc.cycle, norm)
    mycc.cycle += 1

    res = mycc.gather_vector(res)
    return res

@mpi.parallel_call
def release_imds(mycc):
    mycc._imds = None

def kernel(mycc):
    """
    Krylov kernel.
    """
    froot = opt.root
    tolnormt = mycc.conv_tol_normt
    max_cycle = mycc.max_cycle
    vec_size = mycc.vec.size
    
    if mycc.precond is None:
        M = None
    else:
        M = spla.LinearOperator((vec_size, vec_size), matvec=mycc.mop)
    
    def f_res(x):
        return get_lambda_res(mycc, x)
    
    x0 = mycc.vec

    if mycc.method == 'krylov':
        inner_m = mycc.inner_m
        outer_k = mycc.outer_k
        res = froot(f_res, x0, method='krylov',
                    options={'fatol': tolnormt, 'tol_norm': safe_max_abs, 
                             'disp': True, 'maxiter': max_cycle // inner_m,
                             'line_search': 'wolfe',
                             'jac_options': {'rdiff': 1e-6, 'inner_maxiter': 100, 
                                             'inner_inner_m': inner_m, 'inner_tol': tolnormt * 0.5,
                                             'outer_k': outer_k, 'inner_M': M}
                            })
    else:
        raise ValueError
    
    release_imds(mycc)
    conv = res.success
    mycc.distribute_vector_(res.x, write='l')
    return conv, mycc.l1, mycc.l2

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd_lambda
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
