# This code generates 400 random initial particle positions and numerically advects them through the Bickley Jet velocity field using odeint, which repeatedly evalutes the defined velocity functions. After integrating each trajectory, it wraps the x-coordinates back into the periodic fundamental doman while leaving y unmodified.

# X and Y bounds

xBound = [x_l, x_r]
yBound = [y_b, y_t]

# numTraj, Icr, Traj

numTraj = 400
ICr = []
Traj = []

# Appends the initial conditions of all the trajectories to ICr

for i in range(numTraj):
    ICr.append([np.random.uniform(xBound[0], xBound[1]), np.random.uniform(yBound[0], yBound[1])])

# odeint solves initial value problems of ordinary differential equations
# Appends the solution to odeint to Traj

for i in range(len(ICr)):
    Traj.append(odeint(BickleyJet, ICr[i], times, args = (params,)))



# Now, let's mod these trajectories to be in the fundamental domain


for i in range(numTraj): # Iterates over each trajectory
    for t in range(len(Traj[i])): # Iterates over the time step
        Traj[i][t][0] = Traj[i][t][0]%xBound[1] # Calculates the fundamental domain in the x direction
                                                # Doesn't calculate the fundamental domain in the y direction as it's only a singly periodic function
