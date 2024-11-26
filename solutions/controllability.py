import numpy as np
import sympy as sp

params = {
    'Lf': 0.3,
    'Lr': 0.3,
    'Cyf': 100.0,
    'Cyr': 100.0,
    'mass': 11.1,
    'Iz': 0.5,
    'tau': 0.1,
}

# Params.
# Lf = params['Lf']
# Lr = params['Lr']
# mass = params['mass']
# Cyf = params['Cyf']
# Cyr = params['Cyr']
# Cy = (Cyf + Cyr) * 0.5
# Iz = params['Iz']
# tau = params['tau']
# v_x = 2.0

Cyf, Cyr, Lf, Lr, mass, Iz, v_x = sp.symbols('Cyf Cyr Lf Lr mass Iz v_x')


B = sp.Matrix([[0.0],
              [2 * Cyf / mass],
              [0.0],
              [2 * Cyf * Lf / Iz]])

A = sp.zeros(4, 4)
A[0, 1] = 1.0
A[1, 1] = - (2 * Cyf + 2 * Cyr) / (mass * v_x)
A[1, 2] = (2 * Cyf + 2 * Cyr) / mass
A[1, 3]= - (2 * Cyf * Lf - 2 * Cyr * Lr) / (mass * v_x)

A[2, 3] = 1.0
A[3, 1] = - (2 * Cyf * Lf - 2 * Cyr * Lr) / (Iz * v_x)
A[3, 2] = (2 * Cyf * Lf - 2 * Cyr * Lr) / Iz
A[3, 3] = - (2 * Cyf * Lf ** 2 + 2 * Cyr * Lr ** 2) / (Iz * v_x)

sp.pprint(A)
sp.pprint(B)

column1 = B
column2 = A @ B
column3 = A @ A @ B
column4 = A @ A @ A @ B

C = sp.Matrix.hstack(column1, column2, column3, column4)

# Check controllability
rank_C = C.rank()
is_controllable = rank_C == A.shape[0]

# Print results
sp.pprint(C)
print(f"Rank of Controllability Matrix: {rank_C}")
print(f"System is controllable: {is_controllable}")