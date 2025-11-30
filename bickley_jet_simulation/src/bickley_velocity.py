
"""
Eulerian velocity field for the Bickley Jet.
"""

import numpy as np
from src.bickley_parameters import params

def BickleyJet_velocity_field(X, Y, t, params=params):
    U, L, r0, k1, k2, k3, e1, e2, e3, c3, c2, c1 = params

    U0 = -U * (1 - np.tanh(Y/L)**2)
    V0 = 0

    sx1 = e1*np.cos(k1*(X + c1*t))
    sx2 = e2*np.cos(k2*(X + c2*t))
    sx3 = e3*np.cos(k3*(X + c3*t))

    U1 = ((-2*U*np.tanh(Y/L)) / (np.cosh(Y/L)**2)) * (sx1 + sx2 + sx3)

    sy1 = e1*k1*np.sin(k1*(X + c1*t))
    sy2 = e2*k2*np.sin(k2*(X + c2*t))
    sy3 = e3*k3*np.sin(k3*(X + c3*t))

    V1 = (U*L / (np.cosh(Y/L)**2)) * (sy1 + sy2 + sy3)

    return U0 + U1, V0 + V1
