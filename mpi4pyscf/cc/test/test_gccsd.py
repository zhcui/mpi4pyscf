#!/usr/bin/env python

"""
Test GCCSD with MPI.

mpirun -np 2 python test_gccsd.py
"""

# ZHC TODO
# 1. minimize mpi calls [merge, better functions]
# 2. integral oovv, vvoo?
# 3. outcore intermidiates. 

import numpy as np
from pyscf import gto
from pyscf import scf
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

mycc_ref = serial_cc.GCCSD(mf)
mycc_ref.conv_tol = 1e-8
mycc_ref.conv_tol_normt = 1e-6
eris_ref = mycc_ref.ao2mo()

# initial MP2 
emp2_ref, t1_mp2_ref, t2_mp2_ref = mycc_ref.init_amps()

# 1st step Ecc
ecc_0_ref = mycc_ref.energy(t1_mp2_ref, t2_mp2_ref, eris_ref)

# converged Ecc
ecc_ref, t1_cc_ref, t2_cc_ref = mycc_ref.kernel()

# converged lambda
mycc_ref.max_cycle = 50
l1_cc_ref, l2_cc_ref = mycc_ref.solve_lambda()


# tau_ref
tau_ref = imd.make_tau(t2_cc_ref, t1_cc_ref, t1_cc_ref)

# imd ref
Foo_ref = imd.cc_Foo(t1_cc_ref, t2_cc_ref, eris_ref)
Fvv_ref = imd.cc_Fvv(t1_cc_ref, t2_cc_ref, eris_ref)
Fov_ref = imd.cc_Fov(t1_cc_ref, t2_cc_ref, eris_ref)
Woooo_ref = imd.cc_Woooo(t1_cc_ref, t2_cc_ref, eris_ref)
Wvvvv_ref = imd.cc_Wvvvv(t1_cc_ref, t2_cc_ref, eris_ref)
Wovvo_ref = imd.cc_Wovvo(t1_cc_ref, t2_cc_ref, eris_ref)

# test class
mf = mf.newton()
mycc = mpicc.GCCSD(mf)
mycc.conv_tol = 1e-8
mycc.conv_tol_normt = 1e-6
mycc.max_cycle = 50
mycc.kernel()

mycc.distribute_amplitudes_(t1_cc_ref, t2_cc_ref)

test_converged_lambda(mycc, ref=[l1_cc_ref, l2_cc_ref])

#mycc.test_update_lambda(ref=[l1_cc_ref, l2_cc_ref])

rdm1 = mycc.make_rdm1()
rdm1_ref = mycc_ref.make_rdm1()

print ("rdm1 diff")
print (max_abs(rdm1 - rdm1_ref))
print (rdm1)

#test_init_amps(mycc)
#test_converged_ecc(mycc)
#
#test_converged_lambda(mycc)
#
#imds_ref = serial_lambda.make_intermediates(mycc_ref, t1_cc_ref, t2_cc_ref, eris_ref)
#
#mycc.distribute_amplitudes_(t1_cc_ref, t2_cc_ref)
##imds = mycc.test_lambda_imds(ref=None)
#
#print ("*" * 79)
#print ("test make_intermediates")
#print ("test v1")
#print (max_abs(imds.v1 - imds_ref.v1))
#print ("test v2")
#print (max_abs(imds.v2 - imds_ref.v2))
#print ("test w3")
#print (max_abs(imds.w3 - imds_ref.w3))
#print ("test woooo")
#print (max_abs(np.asarray(imds.woooo) - np.asarray(imds_ref.woooo)))
#print ("test wovvo")
#print (max_abs(np.asarray(imds.wovvo) - np.asarray(imds_ref.wovvo)))
#print ("test wovoo")
#print (max_abs(np.asarray(imds.wovoo) - np.asarray(imds_ref.wovoo)))
#print ("test wvvvo")
#print (max_abs(np.asarray(imds.wvvvo) - np.asarray(imds_ref.wvvvo)))
#
#
#l1new_ref, l2new_ref = serial_lambda.update_lambda(mycc_ref, t1_cc_ref, t2_cc_ref, 
#                                            t1_cc_ref, t2_cc_ref, eris_ref,
#                                            imds_ref)
#
#mycc.test_update_lambda(ref=[l1new_ref, l2new_ref])

print ("test tau")
mycc.test_make_tau(ref=tau_ref)

print ("test Foo")
mycc.test_cc_Foo(ref=Foo_ref)
print ("test Fvv")
mycc.test_cc_Fvv(ref=Fvv_ref)
print ("test Fov")
mycc.test_cc_Fov(ref=Fov_ref)

print ("test Woooo")
mycc.test_cc_Woooo(ref=Woooo_ref)
print ("test Wvvvv")
mycc.test_cc_Wvvvv(ref=Wvvvv_ref)
print ("test Wovvo")
mycc.test_cc_Wovvo(ref=Wovvo_ref)

#print ("test t1")
t1new_ref, t2new_ref = mycc_ref.update_amps(mycc_ref.t1, mycc_ref.t2, eris_ref)

mycc.test_update_amps(ref=[t1new_ref, t2new_ref])


