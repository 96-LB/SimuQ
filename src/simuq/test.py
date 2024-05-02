import time
from simuq import QSystem
from simuq import Qubit
from simuq.aais import heisenberg, two_pauli
from simuq.solver import generate_as
from simuq.ionq import IonQProvider


def ising(N, T, J, h):
    qs = QSystem()
    q = [Qubit(qs) for _ in range(N)]
    H = 0
    for j in range(N):
        for k in range(N):
            H += J[j, k] * q[j].Z * q[k].Z
    for j in range(N):
        H += h[j] * q[j].X
    qs.add_evolution(H, T)
    return qs


import numpy as np

def correctness_test(ham, aais='two_pauli', timeout=10, verbose=1):
    
    with open('.env', 'r') as f:
        api_key = f.readline().strip()
    
    old = IonQProvider(api_key=api_key)
    old.compile(ham, backend='forte', aais=aais, verbose=verbose, tol=0.001)
    old.run(on_simulator=True, shots=100000, verbose=2)
    
    new = IonQProvider(api_key=api_key)
    new.compile(ham, backend='forte', aais=aais, solver='greedy', verbose=verbose, tol=0.001)
    new.run(on_simulator=True, shots=100000, verbose=2)
    
    old_res = new_res = None
    x = 1
    while timeout and (not old_res or not new_res):
        time.sleep(x)
        old_res = old.results()
        new_res = new.results()
        x *= 2
        timeout -= 1
    
    if not old_res or not new_res:
        raise Exception('Failed to collect results.')
    
    total_a, total_b = 0, 0
    dist = 0
    m = 0
    for key in set(old_res.keys()).union(new_res.keys()):
        old_val = old_res.get(key, 0)
        new_val = new_res.get(key, 0)
        dist += abs(old_val - new_val)/2
        m = max(m, abs(old_val - new_val))
        total_a += old_val
        total_b += new_val
    
    print(m)
    return (total_a, total_b, dist)



def timing_test(ham, new=True, pauli=False, verbose=1):
    if pauli:
        mach = two_pauli.generate_qmachine(ham.num_sites, e=None)
    else:
        mach = heisenberg.generate_qmachine(ham.num_sites, e=None)
    
    alignment = [i for i in range(ham.num_sites)]
    try:
        generate_as(
                ham,
                mach,
                trotter_args={"num": 6, "order": 1, "sequential": True, "randomized": False},
                solver='greedy' if new else 'least_squares',
                solver_args={"tol": 0.01},
                override_layout=alignment,
                verbose=verbose,
            )
    except Exception as e:
        if e.args[0] == 'SUCCESS':
            return e.args[1]
        else:
            raise e

T = 1

def boring_ising_chain(N):
    h = np.ones(N)
    J_chain = np.zeros((N, N))
    for j in range(N - 1):
        J_chain[j, j + 1] = 1
    ising_chain = ising(N, T, J_chain, h)
    return timing_test(ising_chain, new=True, pauli=True, verbose=0)

def random_hamiltonian(N):
    qs = QSystem()
    q = [Qubit(qs) for _ in range(N)]
    H = 0
    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            H += np.random.uniform(-1, 1) * q[j].X * q[k].X
            #H += np.random.uniform(-1, 1) * q[j].X * q[k].Y
            #H += np.random.uniform(-1, 1) * q[j].X * q[k].Z
            #H += np.random.uniform(-1, 1) * q[j].Y * q[k].X
            H += np.random.uniform(-1, 1) * q[j].Y * q[k].Y
            #H += np.random.uniform(-1, 1) * q[j].Y * q[k].Z
            #H += np.random.uniform(-1, 1) * q[j].Z * q[k].X
            #H += np.random.uniform(-1, 1) * q[j].Z * q[k].Y
            H += np.random.uniform(-1, 1) * q[j].Z * q[k].Z
    for j in range(N):
        H += np.random.uniform(-1, 1) * q[j].X
        H += np.random.uniform(-1, 1) * q[j].Y
        H += np.random.uniform(-1, 1) * q[j].Z
    qs.add_evolution(H, T)
    return correctness_test(qs, aais='heisenberg', timeout=100, verbose=0)



def random_hamiltonian_sparse(N):
    qs = QSystem()
    q = [Qubit(qs) for _ in range(N)]
    H = 0
    pairs = set()
    while len(pairs) < min(N*N*3, 10):
        pairs.add((np.random.randint(0, N), np.random.randint(0, N), np.random.randint(0, 3)))
    for j in range(N):
        for k in range(N):
            if (j, k, 0) in pairs:
                H += np.random.uniform(-1, 1) * q[j].X * q[k].X
            if (j, k, 1) in pairs:
                H += np.random.uniform(-1, 1) * q[j].Y * q[k].Y
            if (j, k, 2) in pairs:
                H += np.random.uniform(-1, 1) * q[j].Z * q[k].Z
    for j in range(N):
        if (j, 0, 0) in pairs:
            H += np.random.uniform(-1, 1) * q[j].X
        if (j, 0, 1) in pairs:
            H += np.random.uniform(-1, 1) * q[j].Y
        if (j, 0, 2) in pairs:
            H += np.random.uniform(-1, 1) * q[j].Z
    qs.add_evolution(H, T)
    return correctness_test(qs, aais='heisenberg', timeout=10, verbose=0)



step, end = 2, 16
for i in range(0, end, step):
    N = i + step
    t = 0
    print(N, boring_ising_chain(N))
