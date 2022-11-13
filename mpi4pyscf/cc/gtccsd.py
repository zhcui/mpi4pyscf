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
#         Junjie Yang <yangjunjie0320@caltech.edu>
#

"""
Tailored MPI-GCCSD with real intergals.

Usage: mpirun -np 2 python gtccsd.py
"""

from functools import reduce
import numpy as np
import scipy.linalg as la

from pyscf import lib
from pyscf import scf
from pyscf import ao2mo
from pyscf.cc import ccsd, gccsd

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.ccsd import (_task_location, _sync_, _pack_scf)
from mpi4pyscf.cc.gccsd import (GGCCSD, BLKMIN)
from mpi4pyscf.cc.gccsd import update_amps as update_amps_gccsd

from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.ci.cisd import tn_addrs_signs

comm = mpi.comm
rank = mpi.rank

einsum = lib.einsum
einsum_mv = lib.einsum 


def fill_amps_cas(mycc, t1, t2, t1_cas, t2_cas, eris=None):
    """
    Fill the t1 and t2 with t1_cas and t2_cas.
    inplace change t1 and t2.
    """
    if t1_cas is None or t2_cas is None:
        mycc.t1_cas, mycc.t2_cas = mycc.get_cas_amps(eris=eris)
        t1_cas = mycc.t1_cas
        t2_cas = mycc.t2_cas

    t1T = t1.T
    t2T = np.asarray(t2.transpose(2, 3, 0, 1), order='C')
    nvir_seg, nvir, nocc = t2T.shape[:3]
    ntasks = mpi.pool.size
    ncore = mycc.ncore

    vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vloc0, vloc1 = vlocs[rank]
    assert vloc1 - vloc0 == nvir_seg
    
    nocc_cas, nvir_cas = t1_cas.shape

    t1T[:nvir_cas, ncore:] = t1_cas.T
    t2T_cas = t2_cas.transpose(2, 3, 0, 1)
    #if vloc1 <= nvir_cas:
    #    t2T[:, :nvir_cas, ncore:, ncore:] = t2T_cas[vloc0:vloc1]
    #elif vloc0 < nvirt_cas and vloc1 > nvir_cas:
    #    t2T[:(nvir_cas - vloc0), :nvir_cas, ncore:, ncore:] = t2T_cas[vloc0:nvirt_cas]
    if vloc0 < nvir_cas:
        end = min(vloc1, nvir_cas)
        t2T[:(end - vloc0), :nvir_cas, ncore:, ncore:] = t2T_cas[vloc0:end]

def update_amps(mycc, t1, t2, eris):
    """
    Update GTCCSD amplitudes.
    """
    t1, t2 = update_amps_gccsd(mycc, t1, t2, eris)
    mycc.fill_amps_cas(t1, t2, mycc.t1_cas, mycc.t2_cas, eris=eris)
    return t1, t2

@mpi.parallel_call(skip_args=[1], skip_kwargs=['eris'])
def get_cas_amps(mycc, eris):
    """
    Get cas space amplitudes.

    Args:
        mycc: cc object.
        eris: eris.
    
    Returns:
        t1_cas: cas space t1, shape (nocc_cas, nvir_cas)
        t2_cas: cas space t2, shape (nocc_cas, nocc_cas, nvir_cas, nvir_cas)
    """
    if rank == 0:
        ncas = mycc.ncas
        nocc = mycc.nocc
        nocc_cas = mycc.nocc_cas
        nvir_cas = mycc.nvir_cas
        # CI solver
        if mycc.cisolver is None:
            cisolver = direct_spin1.FCI()
            cisolver.verbose = mycc.verbose
            cisolver.max_memory = mycc.max_memory
            cisolver.max_cycle = mycc.max_cycle
            cisolver.conv_tol = mycc.conv_tol * 0.1
        else:
            cisolver = mycc.cisolver
        mycc.cisolver = cisolver

        logger.info(mycc, 'TCCSD CI start.')

        h0 = eris.h0_cas
        h1 = eris.h1_cas
        h2 = eris.h2_cas

        from libdmet.solver.impurity_solver import Block2
        from libdmet.system import integral
        if isinstance(cisolver, Block2):
            Ham = integral.Integral(ncas, False, False, h0, {"cd": h1[None]},
                                    {"ccdd": h2[None]})
            spin = 0 if cisolver.cisolver.use_general_spin else nocc_cas
            _, e_fci = cisolver.run(Ham, spin=spin, nelec=nocc_cas)
        else:
            e_fci, fcivec = cisolver.kernel(h1, h2, ncas, (nocc_cas, 0),
                                            ecore=h0, **mycc.ci_args)
        
        mycc.cisolver.fcivec = fcivec
        logger.info(mycc, 'TCCSD CI energy: %25.15f', e_fci)
        
        # FCI/DMRG-MPS -> CISD -> CCSD
        if isinstance(cisolver, Block2):
            # MPS to CI
            ref_str = "1"*nocc_cas + "0"*nvir_cas
            cisolver.cisolver.mps2ci_run(ref_str, tol=1e-9)
            tmpDir = cisolver.cisolver.tmpDir
            civecFile = os.path.join(tmpDir, "sample-vals.npy")
            civec = np.load(civecFile)
            cidetFile = os.path.join(tmpDir, "sample-dets.npy")
            cidet = np.load(cidetFile)
            idx = np.argsort(np.abs(civec))[::-1]
            max_id  = idx[0]
            max_str = cidet[max_id] 
            max_vec = civec[max_id]
        else:
            max_id = np.unravel_index(np.argmax(np.abs(fcivec)), fcivec.shape)
            max_str = bin(cistring.addr2str(ncas, nocc_cas, max_id[0]))
            max_vec = fcivec[max_id]
 
        logger.info(mycc, "max fcivec det id: %s", max_id)
        logger.info(mycc, "string: %s", max_str)
        logger.info(mycc, "weight: %s", max_vec)

        if isinstance(cisolver, Block2):
            from libdmet.solver.gtccsd import get_cisd_vec_cas 
            c0, cis_a, cid_aa = get_cisd_vec_cas(mycc, civec, cidet)
        else:
            t1addrs, t1signs = tn_addrs_signs(ncas, nocc_cas, 1)
            t2addrs, t2signs = tn_addrs_signs(ncas, nocc_cas, 2)
            
            c0 = fcivec[0, 0]
            logger.info(mycc, 'TCCSD CI reference weight c0: %25.15f', c0)
            cis_a = fcivec[t1addrs, 0] * t1signs
            #logger.info(mycc, "fcivec[t1addrs, 0]\n%s", cis_a)
            cid_aa = fcivec[t2addrs, 0] * t2signs

        cis_a /= c0
        cid_aa /= c0

        t1_cas = cis_a.reshape(nocc_cas, nvir_cas)
        t2_cas  = ccsd._unpack_4fold(cid_aa, nocc_cas, nvir_cas)
        tmp = np.einsum('ia, jb -> ijab', t1_cas, t1_cas)
        tmp = tmp - tmp.transpose(0, 1, 3, 2)
        t2_cas -= tmp
    else:
        t1_cas = None
        t2_cas = None

    comm.Barrier()
    t1_cas = mpi.bcast(t1_cas)
    t2_cas = mpi.bcast(t2_cas)
    return t1_cas, t2_cas

@mpi.parallel_call(skip_args=[1], skip_kwargs=['eris'])
def init_amps(mycc, eris=None):
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris

    time0 = logger.process_clock(), logger.perf_counter()
    mo_e = eris.mo_energy
    nocc = mycc.nocc
    nvir = mo_e.size - nocc
    mo_e_o = mo_e[:nocc]
    mo_e_v = mo_e[nocc:] + mycc.level_shift
    eia = mo_e_o[:, None] - mo_e_v
    t1T = eris.fock[nocc:, :nocc] / eia.T
    loc0, loc1 = _task_location(nvir)

    t2T = np.empty((loc1-loc0, nvir, nocc, nocc))
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir+1))))
    emp2 = 0
    for p0, p1 in lib.prange(0, loc1-loc0, blksize):
        eris_vvoo = eris.xvoo[p0:p1]
        t2T[p0:p1] = (eris_vvoo / lib.direct_sum('ia, jb -> abij', eia[:, loc0+p0:loc0+p1], eia))
        emp2 += np.einsum('abij, abij', t2T[p0:p1], eris_vvoo.conj(), optimize=True).real
        eris_vvoo = None
    
    mycc.emp2 = comm.allreduce(emp2) * 0.25
    logger.info(mycc, 'Init t2, MP2 energy = %.15g', mycc.emp2)
    logger.timer(mycc, 'init mp2', *time0)
    mycc.t1 = t1T.T
    mycc.t2 = t2T.transpose(2, 3, 0, 1)
    
    if mycc.t1_cas is None or mycc.t2_cas is None:
        mycc.t1_cas, mycc.t2_cas = mycc.get_cas_amps(eris=eris)
    
    mycc.fill_amps_cas(mycc.t1, mycc.t2, mycc.t1_cas, mycc.t2_cas, eris=eris)
    
    return mycc.emp2, mycc.t1, mycc.t2

def _init_ggtccsd(ccsd_obj):
    from pyscf import gto
    from mpi4pyscf.tools import mpi
    from mpi4pyscf.cc import gtccsd
    if mpi.rank == 0:
        mpi.comm.bcast((ccsd_obj.mol.dumps(), ccsd_obj.pack()))
    else:
        ccsd_obj = gtccsd.GGTCCSD.__new__(gtccsd.GGTCCSD)
        ccsd_obj.t1 = ccsd_obj.t2 = ccsd_obj.l1 = ccsd_obj.l2 = None
        mol, cc_attr = mpi.comm.bcast(None)
        ccsd_obj.mol = gto.mole.loads(mol)
        ccsd_obj.unpack_(cc_attr)

    if False:  # If also to initialize cc._scf object
        if mpi.rank == 0:
            if hasattr(ccsd_obj._scf, '_scf'):
                # ZHC FIXME a hack, newton need special treatment to broadcast
                e_tot = ccsd_obj._scf.e_tot
                ccsd_obj._scf = ccsd_obj._scf._scf
                ccsd_obj._scf.e_tot = e_tot
            mpi.comm.bcast((ccsd_obj._scf.__class__, _pack_scf(ccsd_obj._scf)))
        else:
            mf_cls, mf_attr = mpi.comm.bcast(None)
            ccsd_obj._scf = mf_cls(ccsd_obj.mol)
            ccsd_obj._scf.__dict__.update(mf_attr)

    key = id(ccsd_obj)
    mpi._registry[key] = ccsd_obj
    regs = mpi.comm.gather(key)
    return regs

class GGTCCSD(GGCCSD):
    """
    MPI tailored GGCCSD.
    """
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 remove_h2=False, save_mem=False, 
                 ncas=0, nelecas=0, nocc=None):
        assert isinstance(mf, scf.ghf.GHF)
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.remove_h2 = remove_h2
        self.save_mem = save_mem
        
        self.ncas = ncas
        self.nelecas = nelecas
        
        # initialize CAS space
        nmo = self.nmo
        if nocc is None:
            nocc = self.nocc
        else:
            self.nocc = nocc

        ncore = nocc - nelecas
        nocc_cas = nelecas
        nvir_cas = ncas - nocc_cas
        nvir = nmo - nocc
    
        self.ncas = ncas
        self.nelecas = nelecas
        self.ncore = ncore
        self.nvir = nvir
        self.nocc_cas = nocc_cas
        self.nvir_cas = nvir_cas
        assert 0 <= self.ncas <= self.nmo
        assert 0 <= self.nelecas <= self.ncas
        assert 0 <= self.ncore <= self.nmo
        assert 0 <= self.nocc_cas <= self.ncas
        assert 0 <= self.nvir_cas <= self.ncas
    
        self.mo_core = self.mo_coeff[:, :ncore]
        self.mo_cas = self.mo_coeff[:, ncore:ncore+ncas]
        self.mo_vir = self.mo_coeff[:, ncore+ncas:]
    
        self.cisolver = None
        self.ci_args = {"ci0": None, "pspace_size": 1000}
        self.t1_cas = None
        self.t2_cas = None

        self._keys = self._keys.union(["remove_h2", "save_mem", "ncas", "nelecas",
                                       "t1_cas", "t2_cas", "ncore"])

        regs = mpi.pool.apply(_init_ggtccsd, (self,), (None,))
        self._reg_procs = regs
    
    def dump_flags(self, verbose=None):
        if rank == 0:
            GGCCSD.dump_flags(self, verbose)
            logger.info(self, 'TCCSD nocc     = %4d, nvir     = %4d, nmo  = %4d',
                        self.nocc, self.nvir, self.nmo)
            logger.info(self, 'TCCSD nocc_cas = %4d, nvir_cas = %4d, ncas = %4d',
                        self.nocc_cas, self.nvir_cas, self.ncas)
        return self
    
    def pack(self):
        return {'verbose'    : self.verbose,
                'max_memory' : self.max_memory,
                'frozen'     : self.frozen,
                'mo_coeff'   : self.mo_coeff,
                'mo_occ'     : self.mo_occ,
                '_nocc'      : self._nocc,
                '_nmo'       : self._nmo,
                'diis_file'  : self.diis_file,
                'level_shift': self.level_shift,
                'direct'     : self.direct,
                'diis_space' : self.diis_space,
                'remove_h2'  : self.remove_h2,
                'save_mem'   : self.save_mem,
                'ncas'       : self.ncas,
                'nelecas'    : self.nelecas,
                'ncore'      : self.ncore,
                't1_cas'     : self.t1_cas,
                't2_cas'     : self.t2_cas
                }
    
    def ao2mo(self, mo_coeff=None):
        _make_eris_incore_ghf(self, mo_coeff)
        return 'Done'
    
    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None, approx_l=False):
        from mpi4pyscf.cc.gtccsd_lambda import kernel as lambda_kernel
        self.converged_lambda, self.l1, self.l2 = \
                      lambda_kernel(self, eris, t1, t2, l1, l2,
                                    max_cycle=self.max_cycle,
                                    tol=self.conv_tol_normt,
                                    verbose=self.verbose, approx_l=approx_l)
        return self.l1, self.l2

    fill_amps_cas = fill_amps_cas
    update_amps = update_amps
    get_cas_amps = get_cas_amps
    init_amps = init_amps

@mpi.parallel_call
def _make_eris_incore_ghf(mycc, mo_coeff=None, ao2mofn=None):
    """
    Make physist eri with incore ao2mo, for GGHF.
    """
    from libdmet.utils import take_eri
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    _sync_(mycc)
    eris = gccsd._PhysicistsERIs()
    
    if rank == 0:
        eris._common_init_(mycc, mo_coeff)
        comm.bcast((eris.mo_coeff, eris.fock, eris.nocc, eris.mo_energy))
    else:
        eris.mol = mycc.mol
        eris.mo_coeff, eris.fock, eris.nocc, eris.mo_energy = comm.bcast(None)
    
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape

    nvir = nmo - nocc
    vlocs = [_task_location(nvir, task_id) for task_id in range(mpi.pool.size)]
    vloc0, vloc1 = vlocs[rank]
    vseg = vloc1 - vloc0
    
    if rank == 0:
        if callable(ao2mofn):
            raise NotImplementedError
        else:
            assert eris.mo_coeff.dtype == np.double
            eri = mycc._scf._eri
            if (nao == nmo) and (la.norm(eris.mo_coeff - np.eye(nmo)) < 1e-12):
                # ZHC NOTE special treatment for OO-CCD,
                # where the ao2mo is not needed for identity mo_coeff.
                fn = take_eri
                o = np.arange(0, nocc)
                v = np.arange(nocc, nmo)
                if eri.size == nmo**4:
                    eri = ao2mo.restore(8, eri, nmo)
            else:
                if mycc.save_mem:
                    # ZHC NOTE the following is slower, although may save some memory.
                    def fn(x, mo0, mo1, mo2, mo3):
                        return ao2mo.general(x, (mo0, mo1, mo2, mo3),
                                             compact=False).reshape(mo0.shape[-1], mo1.shape[-1],
                                                                    mo2.shape[-1], mo3.shape[-1])
                    o = eris.mo_coeff[:, :nocc]
                    v = eris.mo_coeff[:, nocc:]
                    if eri.size == nao**4:
                        eri = ao2mo.restore(8, eri, nao)
                else:
                    fn = take_eri
                    o = np.arange(0, nocc)
                    v = np.arange(nocc, nmo)
                    if mycc.remove_h2:
                        mycc._scf._eri = None
                        _release_regs(mycc, remove_h2=True)
                    eri = ao2mo.kernel(eri, eris.mo_coeff)
                    if eri.size == nmo**4:
                        eri = ao2mo.restore(8, eri, nmo)


        # cas hamiltonian in the MO space
        ncore = mycc.ncore
        ncas = mycc.ncas
        dm_core = mycc._scf.make_rdm1(mycc.mo_core, mycc.mo_occ[:ncore])
        hcore = mycc._scf.get_hcore()
        vhf_core = mycc._scf.get_veff(mycc.mol, dm_core)
        e_core = np.einsum('ij, ji -> ', hcore, dm_core, optimize=True) + \
                 np.einsum('ij, ji -> ', vhf_core, dm_core, optimize=True) * 0.5 + \
                 mycc._scf.energy_nuc()

        mo_cas = mycc.mo_cas
        h1_cas = reduce(np.dot, (mo_cas.conj().T, hcore + vhf_core, mo_cas))
        idx_cas = np.arange(ncore, ncore+ncas)
        h2_cas = take_eri(eri, idx_cas, idx_cas, idx_cas, idx_cas, compact=True)
        eris.h0_cas = e_core
        eris.h1_cas = h1_cas
        eris.h2_cas = h2_cas
        eris.rdm1_core = dm_core
        eris.vhf_core = vhf_core

    comm.Barrier()
    cput2 = log.timer('CCSD ao2mo initialization:     ', *cput0)
    
    # chunck and scatter:
    
    # 1. oooo
    if rank == 0:
        tmp = fn(eri, o, o, o, o)
        eris.oooo = tmp.transpose(0, 2, 1, 3) - tmp.transpose(0, 2, 3, 1)
        tmp = None
        mpi.bcast(eris.oooo)
    else:
        eris.oooo = mpi.bcast(None)
    cput3 = log.timer('CCSD bcast   oooo:              ', *cput2)
    
    # 2. xooo
    if rank == 0:
        tmp = fn(eri, v, o, o, o)
        eri_sliced = [tmp[p0:p1] for (p0, p1) in vlocs]
    else:
        tmp = None
        eri_sliced = None
    tmp = mpi.scatter_new(eri_sliced, root=0, data=tmp)
    eri_sliced = None
    eris.xooo = tmp.transpose(0, 2, 1, 3) - tmp.transpose(0, 2, 3, 1)
    tmp = None
    cput4 = log.timer('CCSD scatter xooo:              ', *cput3)
    
    # 3. xovo
    if rank == 0:
        tmp_vvoo = fn(eri, v, v, o, o)
        tmp_voov = fn(eri, v, o, o, v)
        # ZHC NOTE need to keep tmp_voov for xvoo
        eri_1 = [tmp_vvoo[p0:p1] for (p0, p1) in vlocs]
        eri_2 = [tmp_voov[p0:p1] for (p0, p1) in vlocs]
    else:
        tmp_vvoo = None
        tmp_voov = None
        eri_1 = None
        eri_2 = None

    tmp_1 = mpi.scatter_new(eri_1, root=0, data=tmp_vvoo)
    eri_1 = None
    tmp_vvoo = None
    
    tmp_2 = mpi.scatter_new(eri_2, root=0, data=tmp_voov)
    eri_2 = None
    tmp_voov = None
    
    eris.xovo = tmp_1.transpose(0, 2, 1, 3) - tmp_2.transpose(0, 2, 3, 1)
    tmp_1 = None
    cput5 = log.timer('CCSD scatter xovo:              ', *cput4)
    
    # 4. xvoo
    eris.xvoo = tmp_2.transpose(0, 3, 1, 2) - tmp_2.transpose(0, 3, 2, 1)
    tmp_2 = None
    cput6 = log.timer('CCSD scatter xvoo:              ', *cput5)
    
    # 5. 6. xovv, xvvo
    if rank == 0:
        tmp = fn(eri, v, v, o, v)
        eri_sliced = [tmp[p0:p1] for (p0, p1) in vlocs]
    else:
        tmp = None
        eri_sliced = None
    tmp_1 = mpi.scatter_new(eri_sliced, root=0, data=tmp)
    eri_sliced = None
    eris.xovv = tmp_1.transpose(0, 2, 1, 3) - tmp_1.transpose(0, 2, 3, 1)

    if rank == 0:
        tmp_2 = np.asarray(tmp.transpose(3, 2, 1, 0), order='C') # vovv
        tmp = None
        eri_sliced = [tmp_2[p0:p1] for (p0, p1) in vlocs]
    else:
        tmp_2 = None
        tmp = None
        eri_sliced = None
    tmp_2 = mpi.scatter_new(eri_sliced, root=0, data=tmp_2)
    eri_sliced = None
    
    eris.xvvo = tmp_1.transpose(0, 3, 1, 2) - tmp_2.transpose(0, 2, 3, 1)
    tmp_1 = None
    tmp_2 = None
    cput7 = log.timer('CCSD scatter xovv, xvvo:        ', *cput6)

    # 7. xvvv
    if rank == 0:
        tmp = fn(eri, v, v, v, v)
        if mycc.remove_h2:
            eri = None
            if mycc._scf is not None:
                mycc._scf._eri = None
        eri_sliced = [tmp[p0:p1] for (p0, p1) in vlocs]
    else:
        tmp = None
        eri_sliced = None
    tmp = mpi.scatter_new(eri_sliced, root=0, data=tmp)
    eri_sliced = None
    eris.xvvv = tmp.transpose(0, 2, 1, 3) - tmp.transpose(0, 2, 3, 1)
    tmp = None
    eri = None
    cput8 = log.timer('CCSD scatter xvvv:              ', *cput7)
    
    mycc._eris = eris
    log.timer('CCSD integral transformation   ', *cput0)
    return eris

if __name__ == '__main__':
    from pyscf import gto
    import time
    
    np.random.seed(1)
    np.set_printoptions(4, linewidth=1000, suppress=True)
    
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

    mycc_ref = gccsd.GCCSD(mf)
    mycc_ref.conv_tol = 1e-10
    mycc_ref.conv_tol_normt = 1e-6
    eris_ref = mycc_ref.ao2mo()

    # converged Ecc
    ecc_ref, t1_cc_ref, t2_cc_ref = mycc_ref.kernel()

    # test class
    mycc = GCCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-6

    print ("test CC converged energy and t1, t2")
    
    e_cc, t1_cc, t2_cc = mycc.kernel()
    print (abs(e_cc - ecc_ref))
    assert abs(e_cc - ecc_ref) < 1e-8
    
    t1_cc, t2_cc = mycc.gather_amplitudes()
    t1_diff = la.norm(t1_cc - t1_cc_ref)
    print (t1_diff)
    assert t1_diff < 1e-7

    if rank == 0:
        t2_diff = la.norm(t2_cc - t2_cc_ref)
        print (t2_diff)
        assert t2_diff < 1e-7

