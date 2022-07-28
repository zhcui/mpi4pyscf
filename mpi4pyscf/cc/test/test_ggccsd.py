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
from mpi4pyscf.cc.gccsd import *

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

# ref serial GCCSD
from libdmet.solver import cc as cc_solver
mycc = cc_solver.GGCCSD(mf)
mycc.conv_tol = 1e-10
mycc.conv_tol_normt = 1e-8
mycc.max_cycle = 50
mycc.kernel()

l1_ref, l2_ref = mycc.solve_lambda()

rdm1_ref = mycc.make_rdm1(ao_repr=True)
rdm2_ref = mycc.make_rdm2(ao_repr=True)


mycc = mpicc.gccsd.GGCCSD(mf)
mycc.conv_tol = 1e-10
mycc.conv_tol_normt = 1e-8
mycc.max_cycle = 50
mycc.kernel()

# test rotation of amplitudes
from libdmet.solver.cc import transform_t1_to_bo, transform_t2_to_bo
mycc.save_amps()

t1_ref, t2_ref = mycc.gather_amplitudes()
umat = np.random.random((mycc.nmo, mycc.nmo))

t1_trans = transform_t1_to_bo(t1_ref, umat)
t2_trans = transform_t2_to_bo(t2_ref, umat)

mycc.restore_from_h5(umat=umat)
t1, t2 = mycc.gather_amplitudes()

print ("t1 trans diff ", max_abs(t1 - t1_trans))
print ("t2 trans diff ", max_abs(t2 - t2_trans))

assert max_abs(t1 - t1_trans) < 1e-10
assert max_abs(t1 - t1_trans) < 1e-10

mycc = mpicc.gccsd.GGCCSD(mf)
mycc.conv_tol = 1e-10
mycc.conv_tol_normt = 1e-8
mycc.max_cycle = 50
mycc.restore_from_h5(umat=np.eye(mycc.nmo))
mycc.kernel()

print ("E diff: ", abs(mycc.e_corr - -0.134698069373674))
assert abs(mycc.e_corr - -0.134698069373674) < 1e-8
mycc.solve_lambda()

print ("-" * 79)
l1, l2 = mycc.gather_lambda()
print ("l1 diff ", max_abs(l1 - l1_ref))
print ("l2 diff ", max_abs(l2 - l2_ref))

assert max_abs(l1 - l1_ref) < 1e-9
assert max_abs(l2 - l2_ref) < 1e-9

rdm1 = mycc.make_rdm1(ao_repr=True)

print ("rdm1")
print (rdm1)
print ("rdm1 diff to ref", max_abs(rdm1 - rdm1_ref))

print ("-" * 79)
rdm2 = mycc.make_rdm2(ao_repr=True)

print ("rdm2")
print (rdm2.shape)
print ("rdm2 diff to ref", max_abs(rdm2 - rdm2_ref))

assert max_abs(rdm2 - rdm2_ref) < 1e-7

# imag time evolution
from libdmet.solver import cc as cc_solver
mycc = cc_solver.GGCCSDITE_RK(mf, dt=0.01)
mycc.conv_tol = 1e-7
mycc.conv_tol_normt = 1e-6
mycc.max_cycle = 200
#mycc.dt = 0.05
mycc.kernel()
E_ref = mycc.e_corr
rdm1_ref = mycc.make_rdm1(ao_repr=True)

mycc = mpicc.gccsd.GGCCSDITE_RK(mf, dt=0.01)
mycc.conv_tol = 1e-7
mycc.conv_tol_normt = 1e-6
mycc.max_cycle = 200
mycc.kernel()

print ("E diff: ", abs(mycc.e_corr - E_ref))
assert abs(mycc.e_corr - E_ref) < 1e-6

mycc.solve_lambda()
rdm1 = mycc.make_rdm1(ao_repr=True)

print ("rdm1")
print (rdm1)
print ("rdm1 diff to ref", max_abs(rdm1 - rdm1_ref))
assert max_abs(rdm1 - rdm1_ref) < 1e-5

