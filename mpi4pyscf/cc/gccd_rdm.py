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
from mpi4pyscf.tools import mpi
#from mpi4pyscf.lib import diis
#from mpi4pyscf.cc.ccsd import (_task_location, _rotate_vir_block)

from libdmet.solver import gccd_rdm

einsum = lib.einsum

comm = mpi.comm
rank = mpi.rank

def _gamma1_intermediates(mycc, t1, t2, l1, l2):
    t1T = t1.T
    t2T = t2.transpose(2, 3, 0, 1)
    l1T = l1.T
    l2T = l2.transpose(2, 3, 0, 1)
    t1 = t2 = l1 = l2 = None

    #doo  = -np.dot(l1T.T, t1T)
    doo = mpi.allreduce(einsum('efim, efjm -> ij', l2T, t2T) * (-0.5))

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
    return gccd_rdm._make_rdm1(mycc, d1, with_frozen=True, ao_repr=ao_repr)

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
    # ZHC TODO
    # use MPI for distributed intermediates 
    t1, t2 = mycc.gather_amplitudes(t1, t2)
    l1, l2 = mycc.gather_lambda(l1, l2)

    if rank == 0:
        d2 = gccd_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2)
        rdm2 = gccd_rdm._make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True,
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
