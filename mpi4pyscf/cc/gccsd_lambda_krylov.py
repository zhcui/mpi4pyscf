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
from mpi4pyscf.cc.gccsd_krylov import (precond_finv, precond_diag, safe_max_abs)

comm = mpi.comm
rank = mpi.rank

einsum = lib.einsum

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

    cput1 = log.timer('CCSD lambda initialization', *cput0)

    nocc, nvir = l1.shape
    cycle = [0]
    
    # ZHC NOTE here is to avoid the overflow of displacement
    if nocc % 2 == 0:
        seg = nocc // 2
    else:
        seg = 1
    
    x0 = mycc.amplitudes_to_vector(l1, l2).reshape(-1, seg)
    sizes = np.cumsum(comm.allgather(x0.shape[0]))
    offsets = []
    for i in range(len(sizes)):
        if i == 0:
            offsets.append((0, sizes[i]))
        else:
            offsets.append((sizes[i-1], sizes[i]))
    x0 = mpi.allgather_new(x0).ravel()
    
    def v2a(x):
        """
        vector (at root) to amps (distributed).
        """
        x = x.reshape(-1, seg)
        vec_send = [x[offset[0]:offset[1]] for offset in offsets]
        vec_data = x
        vec = mpi.scatter_new(vec_send, data=vec_data).ravel()
        l1, l2 = mycc.vector_to_amplitudes(vec)
        return l1, l2

    def a2v(l1, l2, out=None):
        """
        amps (distributed) to vector (at root).
        """
        res = mycc.amplitudes_to_vector(l1, l2, out=out).reshape(-1, seg)
        res = mpi.allgather_new(res).ravel()
        return res

    def f_res(x):
        # first scatter the vector
        l1, l2 = v2a(x)
        
        l1, l2 = fupdate(mycc, t1, t2, l1, l2, eris, imds)
        
        # then gather the vector
        res = a2v(l1, l2, out=None)

        norm = safe_max_abs(res)
        log.info("      cycle = %5d , norm(res) = %15.5g", cycle[0], norm)
        cycle[0] += 1
        return res 
    
    if mycc.precond == 'finv':
        def mop(x):
            l1, l2 = v2a(x)
            l1, l2 = mycc.precond_finv(l1, l2, eris)
            return a2v(l1, l2)
        M = spla.LinearOperator((x0.shape[-1], x0.shape[-1]), matvec=mop)
    elif mycc.precond == 'diag':
        def mop(x):
            l1, l2 = v2a(x)
            l1, l2 = mycc.precond_diag(l1, l2, eris)
            return a2v(l1, l2)
        M = spla.LinearOperator((x0.shape[-1], x0.shape[-1]), matvec=mop)
    else:
        M = None
    
    froot = opt.root
    tolnormt = mycc.conv_tol_normt
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
    elif mycc.method == 'df-sane':
        res = froot(f_res, x0, method='df-sane',
                    options={'fatol': tolnormt, 'disp': True, 'maxfev': max_cycle,
                             'fnorm': safe_max_abs})
    else:
        raise ValueError
    
    conv = res.success
    l1, l2 = v2a(res.x)

    mycc.l1 = l1
    mycc.l2 = l2
    log.timer('CCSD lambda', *cput0)
    return conv, l1, l2

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
