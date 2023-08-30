from simuq.environment import boson, qubit
from simuq.qsystem import QSystem

# Jaynes-Cummings model
qs = QSystem()
atom = qubit(qs)
cav = boson(qs)

omega_a = 0.1
omega_c = 0.01
g = 0.001
h = omega_a * atom.Z / 2 + omega_c * cav.a * cav.c + g * (cav.a + cav.c) * atom.X
t = 0.1
qs.add_evolution(h, t)
