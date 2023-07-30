from simuq.environment import qubit
from simuq.qmachine import *
from simuq.expression import Expression
import numpy as np

def ham_sum(hlist) :
    n = len(hlist)
    if n == 0 :
        return 0
    sites_type = hlist[0].sites_type
    for i in range(n) :
        if hlist[i].sites_type != sites_type :
            raise Exception("Site types do not match")
    ham = []
    for i in range(n) :
        ham += hlist[i].ham

    from simuq.hamiltonian import TIHamiltonian
    return TIHamiltonian(sites_type, ham)

def GenMach(n = 3, inits = None) :
    Rydberg = QMachine()

    C6 = 862690 * 2. * np.pi

    q = [qubit(Rydberg) for i in range(n)]

    l = (C6 / 4) ** (1. / 6) / (2 - 2 * np.cos(2 * np.pi / n)) ** 0.5
    if inits == None :
        x = [(0., 0.)] + [(GlobalVar(Rydberg, init_value = l * (np.cos(i * 2 * np.pi / n) - 1)), GlobalVar(Rydberg, init_value = l * np.sin(i * 2 * np.pi / n))) for i in range(1, n)]
    else :
        x = [(0., 0.)] + [(GlobalVar(Rydberg, init_value = inits[i][0]), GlobalVar(Rydberg, init_value = inits[i][1])) for i in range(1, n)]
    noper = [(q[i].I - q[i].Z) / 2 for i in range(n)]

    hlist = []
    for i in range(n) :
        for j in range(i) :
            dsqr = (x[i][0] - x[j][0])**2 + (x[i][1] - x[j][1])**2
            hlist.append((C6 / (dsqr ** 3)) * noper[i] * noper[j])
    sys_h = ham_sum(hlist)
    
    Rydberg.set_sys_ham(sys_h)

    L = SignalLine(Rydberg)
    ins = Instruction(L, "native", f"Detuning")
    d = LocalVar(ins)
    ham_detuning = 0
    for i in range(n):
        ham_detuning += -d * noper[i]
    ins.set_ham(ham_detuning)

    L = SignalLine(Rydberg)
    ins = Instruction(L, "native", f"Rabi")
    o = LocalVar(ins)
    p = LocalVar(ins)
    ham_rabi = 0
    for i in range(n):
        ham_rabi += o / 2 * (Expression.cos(p) * q[i].X - Expression.sin(p) * q[i].Y)
    ins.set_ham(ham_rabi)

    return Rydberg