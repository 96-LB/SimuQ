from simuq import QSystem
from simuq import Qubit


def ising(N, T, J, h):
    qs = QSystem()
    q = [Qubit(qs) for _ in range(N)]
    H = 0
    for j in range(N):
        for k in range(N):
            H += J[j, k] * q[j].Z * q[k].Z
    for j in range(N):
        H += h[j] * q[j].X
        H += h[j]/2 * q[j].Y
        H += h[j] * 10 * q[j].Z
    qs.add_evolution(H, T)
    qs.add_evolution(H*2, T)
    return qs


import numpy as np

N, T = 4, 1

h = np.ones(N)
J_chain = np.zeros((N, N))
for j in range(N - 1):
    J_chain[j, j + 1] = 1
ising_chain = ising(N, T, J_chain, h)

J_cycle = np.copy(J_chain)
J_cycle[0, N - 1] = 1
ising_cycle = ising(N, T, J_cycle, h)


from simuq.ionq import IonQProvider

qtpp = IonQProvider(api_key = 'cuMTzR2UR95jv9hgYIPBQyMycn8Sb56a')

qtpp.compile(ising_cycle, backend='forte', aais='heisenberg')

qtpp.run(on_simulator=True, shots=1000)
print(qtpp.results())


a = {'0000': 0.28200000524520874, '1000': 0.0989999994635582, '0100': 0.0989999994635582, '0010': 0.0989999994635582, '1010': 0.017000000923871994, '1110': 0.052000001072883606, '0001': 0.0989999994635582, '0101': 0.017000000923871994, '1101': 0.052000001072883606, '1011': 0.052000001072883606, '0111': 0.052000001072883606, '1111': 0.0820000022649765}
b = {'0000': 0.28200000524520874, '1000': 0.0989999994635582, '0100': 0.0989999994635582, '0010': 0.0989999994635582, '1010': 0.017000000923871994, '1110': 0.052000001072883606, '0001': 0.0989999994635582, '0101': 0.017000000923871994, '1101': 0.052000001072883606, '1011': 0.052000001072883606, '0111': 0.052000001072883606, '1111': 0.08299999684095383}

for key in a:
    print(a[key] - b[key])