# This code recenters and vertically shifts all particle trajectories so their y-value fit inside a clean, dynamic range, then adds stationary "top" and "bottom" boundary points that remain fixed in time to help etec track edges more reliably. It then constructs a time-slice array and plots the initial positions of all advected and stationary points for the Bickley Jet flow.


# Now, let's define the dynamic range (in the y direction)
# We are doing this to add stationary points to the top and the bottom

maxVal = 0
minVal = 0
for i in range(numTraj):
    for t in range(len(Traj[i])):
        if Traj[i][t][1] > maxVal:
            maxVal = Traj[i][t][1]
        elif minVal > Traj[i][t][1]:
            minVal = Traj[i][t][1]


dynamicRange = maxVal - minVal
delta = dynamicRange/10

# Transforming all the y coordinates upwards

for i in range(numTraj):
    for t in range(len(Traj[i])):
        Traj[i][t][1] = Traj[i][t][1] + delta - minVal

newXBound = [xBound[0], xBound[1]]
newYBound = [0, maxVal - minVal + 2*delta]

# Let's create the points

NumberPoints = 20
deltax = newXBound[1]/NumberPoints
tinyOffset = deltax/4 # for the top points so that the top and bottom points aren't all aligned
topPoints = [[i*deltax + delta/2, delta + dynamicRange + delta/2] for i in range(NumberPoints)]
bottomPoints = [[i*deltax + tinyOffset, delta/2] for i in range(NumberPoints)]

# Now, lets append the points to Traj

# For the bottomPoints

for i in range(NumberPoints):
    tempTraj = []
    for t in range(len(times)):
        tempTraj.append(bottomPoints[i])

    Traj.append(tempTraj)


# For the topPoints

for i in range(NumberPoints):
    tempTraj = []
    for t in range(len(times)):
        tempTraj.append(topPoints[i])

    Traj.append(tempTraj)


tSlicePlot = []

for t in range(len(Traj[0])):
    xTemp = []
    yTemp = []

    for i in range(len(Traj)):
        xTemp.append(Traj[i][t][0])
        yTemp.append(Traj[i][t][1])

    tSlicePlot.append([xTemp, yTemp])

fig = plt.figure(figsize = (10, 3))

plt.scatter(tSlicePlot[0][0], tSlicePlot[0][1], s=2)
plt.ylim(newYBound[0], newYBound[1])
plt.xlim(newXBound[0], newXBound[1])
plt.title("Bickley Jet Flow with Stationary Points")


# Changes data structure for input to Dual Etec

tSlices = [[] for i in range(len(times))]
for i in range(len(Traj)):
    for t in range(len(Traj[i])):
        tSlices[t].append([Traj[i][t][0], Traj[i][t][1]])
