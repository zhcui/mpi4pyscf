#!/usr/bin/env python

"""
Test GCCSD with MPI.

mpirun -np 2 python test_gccsd.py
"""

import numpy as np
from pyscf import gto
from pyscf import scf, ao2mo
from pyscf import cc as serial_cc
from pyscf.cc import gccsd
from pyscf.cc import gintermediates as imd
from pyscf.cc import gccsd_lambda as serial_lambda
from mpi4pyscf import cc as mpicc
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.gccd import *

np.random.seed(1)
np.set_printoptions(4, linewidth=1000, suppress=True)

einsum = lib.einsum

comm = mpi.comm
rank = mpi.rank

def max_abs(x):
    if np.iscomplexobj(x):
        return np.abs(x).max()
    else:
        return max(np.max(x), abs(np.min(x)))

def split_line(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        print ("\n")
        print ("*" * 79)
        print ("function: ", func.__name__)
        func(*args, **kwargs)
        print ("*" * 79)
    return wrapper

@split_line
def test_converged_ecc(mycc):
    print ("test CC converged energy and t1, t2")
    
    e_cc, t1_cc, t2_cc = mycc.kernel()
    print (abs(e_cc - ecc_ref))
    assert abs(e_cc - ecc_ref) < 1e-8
    
    t1_cc, t2_cc = mycc.gather_amplitudes()

    t1_diff = max_abs(t1_cc - t1_cc_ref)
    print (t1_diff)
    assert t1_diff < 1e-7

    if rank == 0:
        t2_diff = max_abs(t2_cc - t2_cc_ref)
        print (t2_diff)
        assert t2_diff < 1e-7

def test_converged_lambda(mycc, ref=None):
    print ("test CC converged l1, l2")
    
    l1_cc, l2_cc = mycc.solve_lambda()
    l1_cc, l2_cc = mycc.gather_amplitudes(mycc.l1, mycc.l2)

    l1_diff = max_abs(l1_cc - ref[0])
    print ("l1 diff")
    print (l1_diff)
    assert l1_diff < 1e-7

    #if rank == 0:
    #    l2_diff = max_abs(l2_cc - ref[1])
    #    print (l2_diff)
    #    assert l2_diff < 1e-7

@split_line
def test_init_amps(mycc):
    print ("test initial MP2 energy and amplitudes")

    e_mp2, t1_mp2, t2_mp2 = mycc.init_amps()
    t1_mp2, t2_mp2 = mycc.gather_amplitudes(t1_mp2, t2_mp2)
    
    print (abs(e_mp2 - emp2_ref))
    assert abs(e_mp2 - emp2_ref) < 1e-8

    t1_diff = max_abs(t1_mp2 - t1_mp2_ref)
    print (t1_diff)
    assert t1_diff < 1e-7

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

ovlp = mf.get_ovlp()
H0 = mf.energy_nuc()
H1 = mf.get_hcore()
H2 = mf._eri
H2 = ao2mo.restore(4, H2, mol.nao_nr())
from libdmet_solid.system import integral
from libdmet_solid.solver import scf
from libdmet_solid.utils.misc import tile_eri
H2 = tile_eri(H2, H2, H2)
dm0 = mf.make_rdm1()

print (H0)
print (H1.shape)
print (H2.shape)
Ham = integral.Integral(H1.shape[-1], True, False, H0, {"cd": H1[None]},
        {"ccdd": H2[None]}, ovlp=ovlp)

scfsolver = scf.SCF(newton_ah=True)
scfsolver.set_system(mol.nelectron, 0, False, True, max_memory=mol.max_memory)
scfsolver.set_integral(Ham)
E_HF, rhoHF = scfsolver.GGHF(tol=1e-8, InitGuess=dm0)

mf = scfsolver.mf
mf.mol.verbose = mf.verbose = 5

# test class
mf = mf.newton()

from libdmet.solver import cc as cc_solver
mycc = cc_solver.GGCCD(mf)
mycc.conv_tol = 1e-8
mycc.conv_tol_normt = 1e-6
mycc.max_cycle = 50
mycc.kernel()
rdm1_ref = mycc.make_rdm1(ao_repr=True)
rdm2_ref = mycc.make_rdm2(ao_repr=True)

E_ref = mycc.e_corr

mycc = mpicc.gccd.GGCCD(mf)
mycc.conv_tol = 1e-8
mycc.conv_tol_normt = 1e-6
mycc.max_cycle = 50
mycc.kernel()

E = mycc.e_corr

print ("E diff to ref: ", abs(E - E_ref))
assert abs(E - E_ref) < 1e-7

mycc.save_amps()

mycc = mpicc.gccd.GGCCD(mf)
mycc.conv_tol = 1e-8
mycc.conv_tol_normt = 1e-6
mycc.max_cycle = 50
mycc.restore_from_h5(umat=np.eye(mycc.nmo))
mycc.kernel()

mycc.solve_lambda(approx_l=False)
rdm1 = mycc.make_rdm1(ao_repr=True)

print ("rdm1")
print (rdm1)

print ("rdm1 diff to ref: ", max_abs(rdm1 - rdm1_ref))
assert max_abs(rdm1 - rdm1_ref) < 1e-7

rdm2 = mycc.make_rdm2(ao_repr=True)

print ("rdm2")
print (rdm2.shape)

print ("rdm2 diff to ref: ", max_abs(rdm2 - rdm2_ref))
assert max_abs(rdm2 - rdm2_ref) < 1e-7
