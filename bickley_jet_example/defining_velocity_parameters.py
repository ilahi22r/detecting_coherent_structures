# This just defines the Bickley Jet stream function and velocity function

import EtecDualPeriodicBCv2 as Et # import from detecting_coherent_structures/src/etec_dual_periodic_bc_V2.py
import LoopCombine as LC # import from detecting_coherent_structures/src/loop_combine.py
import numpy as np
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import math
import cmath
import random
import scipy.io as sio
import copy
from scipy.spatial import ConvexHull
from scipy.integrate import odeint

#global parameters for the Bickley Jet (units will all be in Mm,hours)
U_b = 0.06266*60*60/1000 #This is a characteristic velocity measured in Mm/hour (converted from km/s)
L_b = 1.770  #This is a characteristic length (in Mm)
r0 = 6.371   #The mean radius of Earth in Mm
k_1 = 2/r0   #the Rossby wave number k_1
k_2 = 4/r0   #the Rossby wave number k_2
k_3 = 6/r0   #the Rossby wave number k_3
#These wavenumbers are taken for a lattitude of -pi/3 (south from the equator ... -pi/2 would be south pole)
k_b = [k_1,k_2,k_3]  #the Rossby wave numbers [k_1,k_2,k_3]
c_3 = 0.4*U_b
c_2 = 0.2*U_b
c_1 = c_3 + ((math.sqrt(5)-1)/2)*(k_b[2]/k_b[1])*(c_2-c_3)
#c_1 = 0.3*U_b
c_b = [c_1,c_2,c_3]   #The Rossby wave speeds
eps_1 = 0.075   #the 1st Rossby wave amplitude
eps_2 = 0.4     #the 2nd Rossby wave amplitude
eps_3 = 0.3     #the 3st Rossby wave amplitude
eps_b = [eps_1,eps_2,eps_3]   #The Rossby wave amplitudes
params = [U_b,L_b,k_b,c_b,eps_b]  #just repackaging the parameters to pass into the various functions

#The characteristic time
T_tot = math.pi*r0/U_b  #the time it takes for a particle moving with the characteristic speed to traverse the x-direction

#Computational Parameters
T_i = 0  #Starting time
nP = 5  #number of multiples of the characteristic time to integrate over
T_f = nP*T_tot  #Final time in hours (converted from days)

x_l = 0   #Rectangular Domain, x-range left (in Mm)
x_r = math.pi*r0   #Rectangular Domain, x-range right (in Mm) ... this works out to be about 20 Mm
y_b = -3   #Rectangular Domain, y-range bottom (in Mm)
y_t = 3   #Rectangular Domain, y-range top (in Mm)

T_num = 1000 #The number of time-steps to take (will use equal time-step evolution for advected particles)
times = np.linspace(T_i, T_f,T_num)

#The stream function, and velocity vector functions
def StreamFunc(z,t,p):
    x,y = z
    U,L,k,c,eps = p
    psi0 = -U*L*math.tanh(y/L)
    psi1_0 = sum([eps[i]*cmath.exp(1j*k[i]*(x-c[i]*t)) for i in range(1,3)])
    psi1 = U*L*math.cosh(y/L)**(-2)*psi1_0.real
    return psi0 + psi1

#the velocity function
def BickleyJet(z,t,p):
    x,y = z
    U,L,k,c,eps = p
    vx0 = U*math.cosh(y/L)**(-2)
    vx1_0 = sum([eps[i]*cmath.exp(1j*k[i]*(x-c[i]*t)) for i in range(1,3)])
    vx1 = 2*U*math.cosh(y/L)**(-3)*math.sinh(y/L)*vx1_0.real
    vy1_0 = sum([eps[i]*1j*k[i]*cmath.exp(1j*k[i]*(x-c[i]*t)) for i in range(1,3)])
    vy1 = U*L*math.cosh(y/L)**(-2)*vy1_0.real
    return [vx0+vx1,vy1]


