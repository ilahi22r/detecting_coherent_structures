fundamentalDomain = [newXBound[1], newYBound[1]]
triInit = Et.triangulation2D(tSlices[0], fundamentalDomain) # initialising a triangulation (this one won't be evolved forward in time)
tri = Et.triangulation2D(tSlices[0], fundamentalDomain) # initialising a triangulation (this one will be evolved forward in time)

# Now, let's plot the initial traingulation

if triInit is not None:

    Domain = [[newXBound[0], newYBound[0]], [newXBound[1], newYBound[1]]]
    figSize = [14, 14]
    tri.TriangulationPlot(Bounds = Domain, FigureSize = figSize, labels = False)
    #tri.MakeDelaunay()

else:
    print("Could not form triangulation")


skipprint = 20  #this just has a print statement occur every skipprint times through the loop.
for i in range(1, len(tSlices)):  #loop over all the time slices
    tri.Evolve(tSlices[i])  # pass in the new time slices in order to evolve
    tri.MakeDelaunay()
    if i%skipprint == 0:
        print("Tri Evolution timestep at: ",i, " of ", len(tSlices))  #just printing out the status of the overall evolution


