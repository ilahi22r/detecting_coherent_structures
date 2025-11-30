# We start by finding each point pair that is adjacent in the triangulation, and does not have any points in the boundary.

HaveCountedEdge = [False for i in range(triInit.totalnumedges)]
AdjacentPoints = []
for simp in triInit.simplist:
    for j in range(3):
        if not HaveCountedEdge[simp.edgeids[j]]:
            HaveCountedEdge[simp.edgeids[j]] = True
            ptpair = [simp.points[(j+1)%3], simp.points[(j+2)%3]]
            if ptpair[0] < numTraj and ptpair[1] < numTraj:
                AdjacentPoints.append(ptpair)

#now let's see how much each of these loops have stretched out
LengthRatio = []
for ptpair in AdjacentPoints:
    rbands = [ptpair]  #one band that is bounded by these points.  If you run everything above, you'll need to change these to reflect the new triangulation.
    Loop = Et.Loop(triInit,rbands)
    InitialLen = Loop.GetWeightTotal()
    tri.OperatorAction(Loop)
    FinalLen = Loop.GetWeightTotal()
    LengthRatio.append(FinalLen/InitialLen)


plt.hist(LengthRatio, 70)
plt.show()


print(AdjacentPoints)
print(LengthRatio)

LengthRatio.sort()
print(LengthRatio)
