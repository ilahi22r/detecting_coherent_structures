#This python module contains the DualEtec variant on Etec.  Here the data structures keep track of the number of times a band passes across an edge.  V2: separating out the underlying triangulation object from the Edge Weight object (since evolving forward the triangulation does not require the weights, and the accumulated operators can then operate on the edge weight object).  Also putting in a copy constructor for the triangulation object
#V3 Putting in plotting functionality
#V4 Implimenting the option to use shear coordinates
#PeriodicBC - This variant of Dual E-tec takes in data that live on a torus (doubly periodic boundary conditions).  It is assumed that all the trajectories are given on the fundamental domain of the torus, and that the range is [0,Dx] and [0,Dy] in the x and y directions respectively.  This is a separate module from the main Dual Etec, since we need to keep track of new data structures (altering the data structures enough to justify this new version).
#PeriodicBCv2 - This is a new version in which we get rid of a few bugs that sometimes result in an unlinked simplex structure (triangles that don't have neighbors)

import numpy as np
from scipy.spatial import Delaunay
from operator import itemgetter, attrgetter  #for making sorting more efficient.
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import math
import scipy.io as sio
from scipy.optimize import curve_fit
import random
import copy


#The simplex class.  Each simplex object will hold the point ID of each of the 3 points that define the simplex.  Their order (up to even permutation) will by convention reflect the x-y coordinate system handedness (RHR -> positive z direction).  The class also holds a reference to each of the three adjacent simplices (in the same order as the points they are across from).  This data will be initialized after each object is created (points at the time of object instantiation, and simplex references after all simplices in a triangulation are created).  Finally, each simplex holds the weights through each edge
class simplex2D:
    
    _count = 0
    
    #IDlist is a simple list of the 3 point IDs from the master list (as part of the tranguation class).  It is assumed that IDlist already refers to points in the proper permutation (this need to be checked before being passed in).
    def __init__(self, IDlist,RelPtPos = [[0,0],[0,0],[0,0]]):
        simplex2D._count += 1
        self.points = []
        for i in range(0,len(IDlist)):
            self.points.append(IDlist[i])
        #initialize the list that will hold references to the correct opposite simplices ... for now have None as placeholders
        self.simplices = [None,None,None]
        self.edgeids = [None,None,None]
        self.SLindex = None   #This is needed to be able to quickly retrieve the position in the big simplist of this simplex.  It will hold the index of this simplex in the simplist
        self.relptregion = []
        for i in range(0,len(RelPtPos)):
            self.relptregion.append([RelPtPos[i][0],RelPtPos[i][1]])
            
            
            #this records the relative position of points.  If the edge connecting two points of this simplex crosses the torus boundary, then we record the x,y integer displacement (how many copies of the domain over in the x and y direction) of the second point from the perspective of the first (with the first and second point ordering determined by the CCW simplex ordering, and the unique edge indexed as usual - i.e. associated with the index of the point it is across from). Example: in a fundamental domain of Dx=Dy=1, consider the first point at [0.5,0.7] and second point at [0.5,0.1], where the line connecting them goes through the top boundary (remember, the periodic boundary conditions).  Then the integer pair describing this edge from this triangle would be [0,1],  since we have to go one copy up in the y direction to get to the second point.  This extra bit of information will allow us to associate distances between points.
        
        
    
    def __del__(self):
        simplex2D._count -= 1
    
    def GetCount(self):
        return simplex2D._count
    
    def __eq__(self, other):
        return self is other  #I'm just doing this to have all comparisions be by object id, not value.  Now can use some of the functionality of a list of simplex objects, like remove, more freely
    
    #this function takes an input point id and outputs the internal id of this point (0,1,2)
    def LocalID(self, IDin):
        stpt = -1
        for i in range(0,len(self.points)):
            if IDin == self.points[i]:
                stpt = i
                break
        if not stpt == -1:
            return stpt
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None
    
    #This returns a list comprised of two elements: each a reference to a simplex adjacent to the given point in this simplex (and sharing an edge).  There are two of these (possibly the same), and they are ordered in a CCW manner: simplex1, this simplex, simplex2.  The input is an ID (assumed to be one of the set for this simplex), so we need to search for the point
    def AdjSimp(self, IDin):
        simp2 = []
        stpt = self.LocalID(IDin)
        if not stpt == None:
            simp2.append(self.simplices[(stpt+2)%3])
            simp2.append(self.simplices[(stpt+1)%3])
            return simp2
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None
    
    #returns the reference to the simplex opposite the given ID.  Again, the ID must be checked to see if it is contained in this simplex
    def OppSimp(self, IDin):
        stpt = self.LocalID(IDin)
        if not stpt == None:
            return self.simplices[stpt]
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None      
        
    #This returns the neighboring simplex which shares the edge to the right of the given point (IDin)
    def RightSimp(self, IDin):
        stpt = self.LocalID(IDin)
        if not stpt == None:
            return self.simplices[(stpt+2)%3]
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None         
        
    #This returns the neighboring simplex which shares the edge to the left of the given point (IDin)
    def LeftSimp(self, IDin):
        stpt = self.LocalID(IDin)
        if not stpt == None:
            return self.simplices[(stpt+1)%3]
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None
        
    #This returns the number of edges that emanate from a given point in this simplex.  Alternatively, this counts the number of simplices attached to this point.
    def NumSimpNeighbors(self, IDin):
        stpt = self.LocalID(IDin)
        if not stpt == None:
            simpcounter = 1
            lsimp = self.simplices[(stpt+1)%3]
            while (not self is lsimp) and (not lsimp is None):
                simpcounter += 1
                nsimp = lsimp.LeftSimp(IDin)
                lsimp = nsimp
            if lsimp is None:  #this deals with the boundary simplex case
                rsimp = self.simplices[(stpt+2)%3]
                while (not self is rsimp) and (not rsimp is None):
                    simpcounter += 1
                    nsimp = rsimp.RightSimp(IDin)
                    rsimp = nsimp
            return simpcounter
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None        
       
    
    #This returns a list of all the simplices (in CCW cyclical order) adjacent to a point (IDin).
    def SimpNeighbors(self, IDin):
        NeighborList = []
        stpt = self.LocalID(IDin)
        if not stpt == None:
            NeighborList.append(self)
            lsimp = self.simplices[(stpt+1)%3]
            while (not self is lsimp) and (not lsimp is None):
                NeighborList.append(lsimp)
                nsimp = lsimp.LeftSimp(IDin)
                lsimp = nsimp
            if lsimp is None:  #this deals with the boundary simplex case
                rsimp = self.simplices[(stpt+2)%3]
                while (not self is rsimp) and (not rsimp is None):
                    NeighborList.append(rsimp)
                    nsimp = rsimp.RightSimp(IDin)
                    rsimp = nsimp
            return NeighborList
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None        
 

    #this returns True/False for whether the current simplex is ajacent to the given simplex along an edge
    def IsEdgeAdj(self, SimpIn):
        isadj = False
        for i in range(0,3):
            if self.simplices[i] is SimpIn:
                isadj = True
                break
        return isadj
    
    
#End of simplex2D class ************************************************************************************************************************************************************************************************************************************************************
    
    
    
#This is the WeightOperator class.  Each Operator stores the index of the 5 elements of the weight list that are used to calculate the new central weight
#the methods will define how each operator acts on the weight list
#Shear coordinates: Each operator stores the index of the 5 elements of the weightlist that are to be updated (in order: central diagonal first, then CCW perimeter edges, starting with an edge that, along with its quadrilateral opposite, form a "Z" with the central diagonal edge).
#the method defines how the operators act on the shear weight list
class WeightOperator:
    
    #the constructor
    def __init__(self, IndexSet, TimeIn = None):
        self.es = IndexSet
        self.time = TimeIn
    
    #This updates the given weightlist
    def Update(self,LoopIn,Reverse = False):
        WL = [LoopIn.weightlist[x] for x in self.es]
        if not LoopIn.Shear:
            LoopIn.weightlist[self.es[0]] = max(WL[1]+WL[3],WL[2]+WL[4]) - WL[0]
        else:
            Diag = WL[0]
            if not Diag == 0:
                LoopIn.weightlist[self.es[0]] = -Diag
                if Diag > 0:
                    if not Reverse:
                        LoopIn.weightlist[self.es[2]] += Diag
                        LoopIn.weightlist[self.es[4]] += Diag
                    else:
                        LoopIn.weightlist[self.es[1]] += Diag
                        LoopIn.weightlist[self.es[3]] += Diag
                else:
                    if not Reverse:
                        LoopIn.weightlist[self.es[1]] += Diag
                        LoopIn.weightlist[self.es[3]] += Diag  
                    else:
                        LoopIn.weightlist[self.es[2]] += Diag
                        LoopIn.weightlist[self.es[4]] += Diag
                    
                    #Need to make sure that the weights are accumulated in the correct order
#here are some comparison functions (functions that take WeightOpeerator objects as input)

#Two weight operators commute if the central edge (edge that is flipped) for one operator is not one of the edges of the other operator. (this definition is symmetric despite appearances)
def WOcommute(WO1,WO2):
    if WO1.es[0] in WO2.es:
        return False
    else:
        return True

def WOequiv(WO1,WO2):
    if WO1.es[0] == WO2.es[0]:
        return True
    else:
        return False
                    
                    
#End of WeightOperator class *******************************************************************************************************************************


#this is the Loop class.  It represents a loop for a given triangulation state.  The weight list indices must match up with the weight id for a triangulation2D object to make sense.
#there is an option to make this a shear loop, which initializes the data differently and has a different update method
class Loop:
    
    #constructor
    def __init__(self,tri,rbands = None, Shear = False, mesh = None):
        self.weightlist = [0 for i in range(tri.totalnumedges)]
        self.Shear = Shear
        if not self.Shear:
            if not rbands is None:
                tri.BandWeightInitialize(rbands,self)
            #can accomodate multiple bands, but it is not checked that they do not cross (badness ensues).  The bands could be tangent (to ensure this, we do +=1 for the weights, instead of setting to 1)
        else:
            if not rbands is None and mesh is None:
                #This first creates a regular loop (regular coordinates), then feeds this into the triangulation obejct to get the shear coordinates
                RegLoop = Loop(tri,rbands)
                tri.BandShearWeightsInitialize(RegLoop, self)
            if not mesh is None:
                for i in range(len(self.weightlist)):
                    self.weightlist[i] = -1.0
                    #this sets every edge equal to negative one (even boundaries ... can fix this later), which represents bands pinned to adjacent points in this triangulation.            
            
            
    def GetWeightTotal(self):
        if not self.Shear:
            return sum(self.weightlist)
        else:
            WT = 0
            for i in range(len(self.weightlist)):
                WT += abs(self.weightlist[i])
            return WT
        
        
    #this will find the maximum weight, and divide all the weights by this value
    def ProjectivizeWeights(self):
        mwv = max(max(self.weightlist),abs(min(self.weightlist)))
        self.weightlist = [x/mwv for x in self.weightlist]


#End of Loop class **************************************************************************************************************************************************************************************************************************************************************

   
    
#This is the triangulation class, the central class in the overall algorithm. It is initialized using a Delaunay triangulation
class triangulation2D:

    #The constructor for triangulation2D.  ptlist is the list of [x,y] positions for the points at the initial time.
    #Reminder that the input points are just in the fundamental domain.  We also have the size of the fundamental domain as [0,Dx) and [0,Dy) in the x and y directions repectively.  Important that the points are in this domain.  We also pass in Dx and Dy.  There are no control points, as this will be a triangulation without boundary.
     
    def __init__(self, ptlist, FDsizes, empty = False):
        if empty:
            #this is for creating a place-holder object that will then be used for copying
            self.atstep = 0
            self.printstep = []
            self.FDsizes = [0,0]  #[Dx,Dy]
            self.ptnum = 0
            self.pointpos = None
            self.pointposfuture = None
            self.simplist = []
            self.pointlist = []
            self.totalnumedges = 0
            self.WeightOperatorList = []
        else:            
            #some initial error/debugging related items:
            self.atstep = 0     #This will be incremented each time the evolve method is used (useful for error handling)
            self.printstep = []  #this triggers a bunch of print items at this evolution step or list of steps (ex [105], or [105,107]).
            self.ptnum = len(ptlist)
            self.FDsizes = FDsizes  #[Dx,Dy]
            #Now put all the points together and triangulate
            self.pointpos = ptlist
            self.pointposfuture = self.pointpos    #pointposfuture holds the point positions at the next iterate.  Here we are just initializing it to be equal to the original point positions.
            
            #Now we need to add on the 8 copies of the points in the fundamental domain, displaced to the surrounding rectangles.  The convention will be that we start with the FD, then the copy down and to the left of the fundamental domain, and work our way around the fundamental domain counter clock-wise.
            temppoints = np.array(self.pointpos)
            temppoints2 = np.array(self.pointpos)
            displaceindices = ((0,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0))
            self.dpindices = displaceindices
            for i in range(1,len(displaceindices)):
                temppoints = np.append(temppoints, temppoints2 + np.array([displaceindices[i][0]*self.FDsizes[0],displaceindices[i][1]*self.FDsizes[1]]), axis=0)
            #now we have 9x the number of actual points (+ the 8 neighboring copies)
                
            temptri = Delaunay(temppoints,qhull_options="QJ Pp")   #create the initial Delaunay triangulation.  The option forces the creation of simplices for degenerate points by applying a random perturbation.

            #now we need to detect if there are any degenerate triangles (triangles that have repeated points ... from wrapping around the torus).  This might only happen in the case of very sparse number densities.  If this happens, we can't handle it and a error message will be thrown (in the future, could deal with this by locally re-triangulating, though wouldn't give a Delaunay triangulation).
            #First, let's go through each simplex and label the points by the copy of the fundamental domain they occupy.  This will be a list.  Also a list of the corresponding point in the fundamental domain.
            ptcopypos = []
            ptfdID = []
            degenerate = []   #True if two of the three points map to the same fundamental domain point, False otherwise
            triloc = []     #gives 1 if all 3 points are in the fundamental domain, 0 if the simplex straddles the FD boundary, and -1 if they are all outside the FD
            bdtri = []   #record the id of all the boundary triangles (minus corner triangles)
            fdtri = []   #record the id of all the interior triangles (all pts in the FD)
            cornertri = []  #the id of all the triangles that contain one of the four FD corners
            AnyDegenerate = False
            for i in range(temptri.simplices.shape[0]):
                tempptlist = temptri.simplices[i].tolist()
                fdID = []
                copypos = []
                for j in range(3):
                    fdID.append(tempptlist[j]%self.ptnum)
                    copypos.append(tempptlist[j]//self.ptnum)
                ptfdID.append(fdID)
                ptcopypos.append(copypos)
                if (fdID[0] == fdID[1] or fdID[0] == fdID[2] or fdID[2] == fdID[1]) and (copypos[0] == 0 or copypos[1] == 0 or copypos[2] == 0):
                    degenerate.append(True)
                    AnyDegenerate = True
                    print("Degenerate: ", tempptlist, " with FD points ", fdID)
                else:
                    degenerate.append(False)
                #first sort out the triangles that have all vertices in the FD
                if copypos[0] == 0 and copypos[1] == 0 and copypos[2] == 0:
                    triloc.append(1)
                    fdtri.append(i)
                #next consider the triangles with at least one point in the FD (but not all)
                #these must intersect the boundary.
                #We will check to see if any of these cover the corners
                elif copypos[0] == 0 or copypos[1] == 0 or copypos[2] == 0:
                    tmpptlist = temptri.points[temptri.simplices[i]].tolist()
                    cornerin1 = TriContains([0,0],tmpptlist)
                    cornerin2 = TriContains([self.FDsizes[0],0],tmpptlist)
                    cornerin3 = TriContains([self.FDsizes[0],self.FDsizes[1]],tmpptlist)
                    cornerin4 = TriContains([0,self.FDsizes[1]],tmpptlist)
                    if cornerin1 or cornerin2 or cornerin3 or cornerin4:
                        cornertri.append(i)
                    else:
                        bdtri.append(i)
                    triloc.append(0)
                #here we sort out all triangles that have at least one edge which connects points in diagonal copies of the FD
                #Of these candidates, we sort out any corner-covering ones and any ones that belong in cornertri (cross the FD, but don't have a point in the FD, and don't contain a corner)
                elif abs(displaceindices[copypos[0]][0] - displaceindices[copypos[1]][0]) + abs(displaceindices[copypos[0]][1] - displaceindices[copypos[1]][1]) == 2 or abs(displaceindices[copypos[0]][0] - displaceindices[copypos[2]][0]) + abs(displaceindices[copypos[0]][1] - displaceindices[copypos[2]][1]) == 2 or abs(displaceindices[copypos[2]][0] - displaceindices[copypos[1]][0]) + abs(displaceindices[copypos[2]][1] - displaceindices[copypos[1]][1]) == 2:
                    tmpptlist = temptri.points[temptri.simplices[i]].tolist()
                    cornerin1 = TriContains([0,0],tmpptlist)
                    cornerin2 = TriContains([self.FDsizes[0],0],tmpptlist)
                    cornerin3 = TriContains([self.FDsizes[0],self.FDsizes[1]],tmpptlist)
                    cornerin4 = TriContains([0,self.FDsizes[1]],tmpptlist)
                    if cornerin1 or cornerin2 or cornerin3 or cornerin4:
                        triloc.append(0)
                        cornertri.append(i)
                    else: 
                        EdgeLines = [[[0,0],[self.FDsizes[0],0]],[[self.FDsizes[0],0], [self.FDsizes[0],self.FDsizes[1]]],[[self.FDsizes[0],self.FDsizes[1]],[0,self.FDsizes[1]]],[[0,self.FDsizes[1]],[0,0]]]    
                        TrialEdges = [[tmpptlist[0],tmpptlist[1]],[tmpptlist[0],tmpptlist[2]],[tmpptlist[2],tmpptlist[1]]]
                        doesIntersect = False
                        for k in EdgeLines:
                            for l in TrialEdges:
                                doesIntersect = IsIntersection(k,l)
                                if doesIntersect:
                                    break
                            if doesIntersect:
                                break
                        if doesIntersect:
                            triloc.append(0)
                            bdtri.append(i)
                        else:
                            triloc.append(-1)
                else:
                    triloc.append(-1)
                
            if AnyDegenerate:
                print("there is a Degenerate Triangle!  The current evolution methods will not work.")
                
            #Now we have a list of the fundamental domain triangles (with all pts in the FD), a list of boundary triangles, and a list of the four corner triangles
           
            bdtrisplit = [[],[],[]]
            bdtrisplit[2].append(cornertri)

            NumUnmatched = len(bdtri)
            IsMatched = [False for x in bdtri]
            currentcompare = 0
            while NumUnmatched > 0:
                for i in range(len(bdtri)):
                    if not IsMatched[i]:
                        tempbundle = [bdtri[i]]
                        NumUnmatched -= 1
                        IsMatched[i] = True
                        FDptsi = ptfdID[bdtri[i]]
                        for j in range(i+1,len(bdtri)):
                            if not IsMatched[j]:
                                FDptsj = ptfdID[bdtri[j]]
                                if (FDptsi[0] in FDptsj) and (FDptsi[1] in FDptsj) and (FDptsi[2] in FDptsj):
                                    tempbundle.append(bdtri[j])
                                    NumUnmatched -= 1
                                    IsMatched[j] = True
                        if len(tempbundle) == 3:
                            bdtrisplit[1].append(tempbundle)
                        elif len(tempbundle) == 2:
                            bdtrisplit[0].append(tempbundle)
                        else:
                            print("there shouldn't be a matching with ", len(tempbundle), " triangles!")
                            print(tempbundle)
                                        
                                        
            #debugging:
            #print("there are ", len(fdtri), " interior triangles")
            #print("for the boundary triangles, there are ", len(bdtrisplit[0]), " pair sets, ", len(bdtrisplit[1]), " triple sets, and there are ", len(bdtrisplit[2]), " corner triangle sets.")
            #print("number of triples in the first set ", len(bdtrisplit[1][0]))
            #print("number of quadruples in the first set ", len(bdtrisplit[2][0]))
            
            #Now we need to store the triangulation data in a local data structure 
            numsimp = len(fdtri)+ len(bdtrisplit[0]) + len(bdtrisplit[1]) + len(bdtrisplit[2])  #the number of simplices in the triangulation
            self.simplist = []

            #first create the list of simplex2D objects (not linked together yet ... need to create every object first)
            tempsimplexIDlist = []
            BigtoLittleIndex = [None for i in range(temptri.simplices.shape[0])]
            MultiplesIndex = [None for i in range(temptri.simplices.shape[0])]
            simpcounter = 0
            for i in range(0,len(fdtri)):
                self.simplist.append(simplex2D(temptri.simplices[fdtri[i]].tolist()))
                self.simplist[-1].SLindex = i
                simpcounter += 1
                tempsimplexIDlist.append(fdtri[i])
                BigtoLittleIndex[fdtri[i]] = i

            for i in range(3):
                for j in range(len(bdtrisplit[i])):
                    #we only include one triangle from each group ... the group members will help us create the links
                    IDtemp = bdtrisplit[i][j][0]
                    tempsimplexIDlist.append(IDtemp)
                    RelPos = []
                    for k in range(3):
                        RelPos.append([displaceindices[ptcopypos[IDtemp][(k+2)%3]][0] - displaceindices[ptcopypos[IDtemp][(k+1)%3]][0], displaceindices[ptcopypos[IDtemp][(k+2)%3]][1] - displaceindices[ptcopypos[IDtemp][(k+1)%3]][1]])
                    self.simplist.append(simplex2D(ptfdID[IDtemp],RelPos))
                    self.simplist[-1].SLindex = simpcounter
                    MultiplesIndex[IDtemp] = []
                    for k in range(0,len(bdtrisplit[i][j])):
                        BigtoLittleIndex[bdtrisplit[i][j][k]] = simpcounter
                        MultiplesIndex[IDtemp].append(bdtrisplit[i][j][k])
                        
                    simpcounter += 1
                    
                    
            #Now we need to go through each of the newly created simpices and link them to their neighbors
            
           
            for i in range(0,numsimp):
                linklist = temptri.neighbors[tempsimplexIDlist[i]].tolist()
                for j in range(3):
                    if not BigtoLittleIndex[linklist[j]] is None:
                        self.simplist[i].simplices[j] = self.simplist[BigtoLittleIndex[linklist[j]]]
                    else:
                        p1 = ptfdID[tempsimplexIDlist[i]][j]
                        #now we check the other matching triangles for the analogous neighbor
                        CopySimplices = MultiplesIndex[tempsimplexIDlist[i]]
                        for k in range(len(CopySimplices)):
                            ptsnew = ptfdID[CopySimplices[k]]
                            indmatch = 0
                            for l in range(3):
                                if ptsnew[l] == p1:
                                    indmatch = l
                                    break
                            nlist = temptri.neighbors[CopySimplices[k]].tolist()
                            if not BigtoLittleIndex[nlist[indmatch]] is None:
                                self.simplist[i].simplices[j] = self.simplist[BigtoLittleIndex[nlist[indmatch]]]
                                break
                
                


            #now create the pointlist with links to individual simplices
            #first initialize the list
            self.pointlist = []
            for i in range(0,self.ptnum):
                self.pointlist.append(None)

            #now go through each simplex and add that simplex to each slot in the pointlist that corresponds to an included point if the slot contains None (possibly more efficient way to do this)
            for i in range(0,numsimp):
                for j in range(0,3):
                     if self.pointlist[self.simplist[i].points[j]] is None:
                            self.pointlist[self.simplist[i].points[j]] = self.simplist[i]

            
            #These are two lists that give the relative copy of the FD that the final position is in relative to the initial position (initial pos is thought of as being in the FD).  The first one is relative to the initial positions and does not change in a given time-step (though of course is re-filled at the beginning of each time-step).  The second one starts off as identical, but is updated as points cross the boundary of the FD.  To be more specific, as a point crosses the FD, we still think of it as being in the FD, but the copy of the FD that the final position is in (relative to this now-moved point position) is now different.  The Crossing event list events will update this list.
            self.FuturePosRelCopyLoc = []   #this holds the indices that need to be put into dpindices to get the shifts
            self.UpdatedFuturePosRelCopyLoc = []  #this directly holds the shifts (updated), - this makes it easier to update
            
                            
            #Now we assign each edge an index.  This goes through each simplex object, and assigns an id to each edge (and the same id to the corresponding edge in the adjacent simplex) if it has not already been assigned.  The index is just taken from an incremental counter.
            edgecounter = 0
            for i in range(0,len(self.simplist)):
                for j in range(0,3):
                    tempsimp = self.simplist[i]
                    if tempsimp.edgeids[j] is None:
                        tempsimp.edgeids[j] = edgecounter
                        if not tempsimp.simplices[j] is None:
                            pt = tempsimp.points[(j+1)%3]
                            Lid = (tempsimp.simplices[j].LocalID(pt)+1)%3
                            tempsimp.simplices[j].edgeids[Lid] = edgecounter
                        edgecounter += 1  
            self.totalnumedges = edgecounter        

            #Finally, we initialize the weight operator list (collects the operators that will act on any weight list we throw at it)
            self.WeightOperatorList = []
            Nonecounter = 0
            for i in range(0,len(self.simplist)):
                for j in range(0,3):
                    tempsimp = self.simplist[i]
                    if tempsimp.simplices[j] is None:
                        Nonecounter += 1
            #print("There are ", Nonecounter, " unlinked edges")
            
            #now a few print statements to check that the linking worked alright
            #let's go through each simplex and count up the number of unlinked sides ... should be zero
            
            #now we initialize the topological time weight operator list.
            #this will hold groups of operators that are mutually time-commuting.  The number of groups will (approximate for now) be the topological time ... the minimal number of such groups
            self.TopTimeOpGroups = []
        
    #End of the constructor for triangulation2D **************************************************************************************************************************************************************************************************************************************************************
    
    #Begin the methods for triangulation2D **************************************************************************************************************************************************************************************************************************************************************
    
    #Evolve method.  This assumes that the starting triangulation is good (no negative areas).  It takes in the new time-slice data in ptlist -- the list of [x,y] positions for the points at the next time-step.
    def Evolve(self, ptlist):

        #self.DebugSignedArea()
        
        self.LoadNewPos(ptlist)  #putting the new point positions in pointposfuture
        
        #self.Debugrelptregion()
        
        self.FuturePosRelCopyLoc = self.GetNewPosCopyLoc()
        self.UpdatedFuturePosRelCopyLoc = [[self.dpindices[x][0],self.dpindices[x][1]] for x in self.FuturePosRelCopyLoc] #this will be updated
        #print("future points copy locations: ", self.FuturePosRelCopyLoc)
        
        #self.Debugrelptregion()
        
        #next get all the Events (using the initial and final areas)
        EventList = self.GetEvents()
        
        #self.Debugrelptregion()
        
        #now sort this list by time
        EventList.sort(key=itemgetter(1), reverse=True) #this is in decending order so that removing from the end(smallest times first) inccurs the smallest computational cost
        #print("Event List:", EventList)
        #now we create the Crossing list, which records as events the times that specific points cross over a boundary line, and the line crossed
        CrossingList = self.GetPtCrossList()  #this is already sorted with smallest time last (like the EventList)
        #print("Crossing List:", CrossingList)

        #self.Debugrelptregion()
        
        #next Evolve the eventlist
        self.GEvolve(EventList,CrossingList)
        #print("finished GEvolve step")

        #self.Debugrelptregion()
        
        #next we need to push the future positions to the current positions
        self.UpdatePtPos()
        
        #self.DebugNoneNeighbors()
        #self.DebugSignedArea()
        #self.Debugrelptregion()
        
        
        #self.DebugSignedArea()

        #print("updated positions")
        #print("finished evolution step #", self.atstep)
        self.atstep += 1  #increment this internal counter ... useful for trigering a print message (for debugging) at a particular time interval 
        #self.MakeDelaunay() #after the atstep increment so that the operators will have the correct time-stamp
        
    #this returns a list of current simpices (each element is [simplex, first time for A = 0]) whose area goes through zero sometime between their current and future positions.
    def GetEvents(self):
        badsimplist = []
        for i in range(0,len(self.simplist)):
            AZT = self.AreaZeroTimeTorus1(self.simplist[i])
            if AZT[0]:
                badsimplist.append([self.simplist[i],AZT[1]])
        return badsimplist
            
    #The main method for evolving the Event List (group of simplices that need fixing)
    #remember that the list is sorted in decending order, and we deal with the last element first
    def GEvolve(self,EventListSimp,EventListCrossing):
        delta = 1e-10
        
        while len(EventListSimp)> 0 or len(EventListCrossing) > 0:
            latestSimpEventTime = 1
            latestCrossingEventTime = 1
            if len(EventListSimp)> 0 and len(EventListCrossing) > 0:
                latestSimpEventTime = EventListSimp[-1][1]
                latestCrossingEventTime = EventListCrossing[-1][1]
            elif len(EventListSimp)> 0:
                latestSimpEventTime = EventListSimp[-1][1]
            else:
                latestCrossingEventTime = EventListCrossing[-1][1]
                
            
            if latestSimpEventTime < latestCrossingEventTime:
                #here we deal with simplex collapse events
                neweventssimp = []  #new simpices to check
                dellistsimp = []    #simplices to delete from the simplex event list if they exist in it
                currenttime = latestSimpEventTime
                #deal with simplex collapse events here
                modlist = self.SFix(EventListSimp[-1],currenttime + delta)    #returns ... [[leftsimp,rightsimp],topsimp (old)]
                neweventssimp = modlist[0]
                delsimp = modlist[1]
                
                
                #print("simplex handling")
                #print("EventSimp ", EventListSimp[-1][0].points, " Current Area ", self.GetSimpArea(EventListSimp[-1][0],currenttime-2*delta))
                #print("delsimp ", delsimp.points," Current Area " , self.GetSimpArea(delsimp,currenttime-2*delta))
                #for i in range(0,len(neweventssimp)):
                #    print("neweventssimp ", i , neweventssimp[i].points, " Current Area ", self.GetSimpArea(neweventssimp[i],currenttime-2*delta))
                
                del EventListSimp[-1]  #get rid of the evaluated event

                
                #first find the time of zero area for core simplex event, and delete it if needed
                delsimp2 = []
                AZT = self.AreaZeroTimeTorus2(delsimp,currenttime + delta)
                if AZT[0]:
                    BinarySearchDel(EventListSimp, [delsimp,AZT[1]])
                #print("after binary search delete: ")
                #self.DebugSignedArea(currenttime)
                #now run through the newevents list and see if each object goes through zero area in the remaining time (if so, add to EventList with the calulated time to zero area)
                for i in range(0,len(neweventssimp)):
                    AZT = self.AreaZeroTimeTorus2(neweventssimp[i],currenttime + delta,Verbose = True)
                    if AZT[0]:
                        #insert in the event list at the correct spot
                        BinarySearchIns(EventListSimp,[neweventssimp[i],AZT[1]])
                        #print("after binary search insert: ")
                        #self.DebugSignedArea(currenttime)
            else:
                #print("Debugrelptregion 1:")
                #self.Debugrelptregion()
                #here we deal with the crossing events
                currenttime = latestCrossingEventTime
                ptindex = EventListCrossing[-1][0]
                ptmove = EventListCrossing[-1][2]
                
                
  
                #print("Debugrelptregion 1:")
                #self.Debugrelptregion()
                #print("before point crossing correction ")
                #self.DebugSignedArea(currenttime)
                # update self.UpdatedFuturePosRelCopyLoc
                self.UpdatedFuturePosRelCopyLoc[ptindex][0] -= ptmove[0]
                self.UpdatedFuturePosRelCopyLoc[ptindex][1] -= ptmove[1]
                #print("Debugrelptregion 3:")
                #self.Debugrelptregion()
                #We need to update the relative position data in every simplex that shares this point
                #first get a list of all the simplices that bound this point
                Sset = self.pointlist[ptindex].SimpNeighbors(ptindex)
                
                 
                for simp in Sset:
                    locid = simp.LocalID(ptindex)
                    #print("Debugrelptregion before:")
                    #self.Debugrelptregion()
                    #print("simp.relptregion (one)",simp.points,simp.relptregion)
                    simp.relptregion[(locid+1)%3][0] += ptmove[0]
                    simp.relptregion[(locid+1)%3][1] += ptmove[1]
                    simp.relptregion[(locid+2)%3][0] -= ptmove[0]
                    simp.relptregion[(locid+2)%3][1] -= ptmove[1]
                
  
                #print("Debugrelptregion 2:")
                #self.Debugrelptregion()
                #print("after point crossing correction ")
                #self.DebugSignedArea(currenttime)
                    #print("simp.relptregion (two)",simp.points,simp.relptregion)
                    
                #now we need to delete the crossing event
                del EventListCrossing[-1]

    #quick debugging function that tests that the relptregion for each side of an edge adds to zero, and that the relptregion in a simplex adds up to zero
    def Debugrelptregion(self):
        for i in range(len(self.pointlist)):
            Sset = self.pointlist[i].SimpNeighbors(i)
            for simp in Sset:
                locid = simp.LocalID(i)
                relreg1 = simp.relptregion[(locid+1)%3]
                simpadj = simp.simplices[(locid+1)%3]
                adjlocid = simpadj.LocalID(i)
                relreg2 = simpadj.relptregion[(adjlocid+2)%3]
                diff = [relreg1[k]+relreg2[k] for k in range(2)]
                #print(diff,i)
                if (not diff[0] == 0) or (not diff[1] == 0):
                    print("Error: the simplices ", simp.points, simpadj.points)
                    print("have bad relative point locations ", simp.relptregion, simpadj.relptregion)
        
        for i in range(len(self.simplist)):
            simp = self.simplist[i]
            simpsum = [0,0]
            for j in range(3):
                simpsum[0] += simp.relptregion[j][0]
                simpsum[1] += simp.relptregion[j][1]
            if not simpsum[0] == 0 or not simpsum[1] == 0:
                print("Error: the simplex ", simp.points, "has a bad set of relative point locations", simp.relptregion)
    
    #Another debugging function that goes through each simplex and calculates the signed area of each triangle
    def DebugSignedArea(self,timeIn = 0):
        for i in range(len(self.simplist)):
            simp = self.simplist[i]
            ptlist = simp.points
            rpr = simp.relptregion
            pt1shift = rpr[2]
            pt2shift = [-1*rpr[1][0],-1*rpr[1][1]]
            #now shift the Initial and Final points appropriately
            Dx, Dy = self.FDsizes
            Initpos = None
            if timeIn == 0:
                Initpos = [[self.pointpos[x][0],self.pointpos[x][1]] for x in ptlist]
            else:    
                Initpos = [self.GetCurrentLoc(x,timeIn) for x in ptlist]
            Initpos[1][0] += Dx*pt1shift[0]
            Initpos[1][1] += Dy*pt1shift[1]
            Initpos[2][0] += Dx*pt2shift[0]
            Initpos[2][1] += Dy*pt2shift[1]
            SA = self.TriArea(Initpos)
            if SA < 0:
                print("Inverted triangle ", i, ptlist, rpr)
                if timeIn == 0:
                    print("at time ", timeIn, " and position ", [[self.pointpos[x][0],self.pointpos[x][1]] for x in ptlist])
                else:
                    print("at time ", timeIn, " and position ", [self.GetCurrentLoc(x,timeIn) for x in ptlist])
    
    #this function will go through all of the simplices to see if any have "None" as neighbors
    def DebugNoneNeighbors(self):
        arenoneneighbors = False
        for i in range(len(self.simplist)):
            simp = self.simplist[i]
            for j in range(3):
                if simp.simplices[j] is None:
                    print("There is an unlinked simplex", simp.points,simp.simplices)
                    arenoneneighbors = True
        return arenoneneighbors
                  
    
    def GetSimpArea(self,SimpIn,timeIn):
        ptlist = SimpIn.points
        rpr = SimpIn.relptregion
        pt1shift = rpr[2]
        pt2shift = [-1*rpr[1][0],-1*rpr[1][1]]
        #now shift the Initial and Final points appropriately
        Dx, Dy = self.FDsizes
        
        Initpos = [self.GetCurrentLoc(x,timeIn) for x in ptlist]
        Initpos[1][0] += Dx*pt1shift[0]
        Initpos[1][1] += Dy*pt1shift[1]
        Initpos[2][0] += Dx*pt2shift[0]
        Initpos[2][1] += Dy*pt2shift[1]
        #print("Points ", ptlist, " are at ", Initpos)
        return self.TriArea(Initpos)
                
    #Fixing a simplex and the surrounding effected simplices.  SimpIn is actually a list [simplex,area zero time]
    #This returns the two new simplices, so that they can be possibly added to the local event list, also the bad simplex so it can be removed (if needed from the local event list)
    def SFix(self,SimpIn,timein):
        Simp = SimpIn[0]
        colind = self.CollapsePt(Simp,SimpIn[1])  #this is the local index of the offending point during the area collapse
        Topsimp = Simp.simplices[colind]
        #if Topsimp is None:
        #    print(Simp.simplices)
        #    print(Simp.points)
        globaltime = self.atstep + timein
        newsimps = self.EdgeFlip([Simp,Topsimp],globaltime)  #this does most of the work in flipping the edge and cleaning up linking
 
        #finally, return the two new simplices, so that they can be checked to see if they need to be included in any update to the local event list. Also return the bad simplex to remove any instance from the event list.
        return [newsimps,Topsimp]
    

    #This takes the accumulated operator list stored in WeightOperatorList and operates sucessively on the given Loop
    #the start and stop index can also be specified to break this up into stages
    #Reverse does the operator actions in reverse order (i.e. for loops in the final triangulation)
    def OperatorAction(self,LoopIn,index = None,Reverse = False):
        startind = 0
        endind = len(self.WeightOperatorList)-1
        if not index is None:
            startind = index[0]
            endind = index[1]
        if not Reverse:
            for i in range(startind,endind+1):
                self.WeightOperatorList[i].Update(LoopIn)
        else:
            for i in range(endind,startind-1,-1):
                self.WeightOperatorList[i].Update(LoopIn,Reverse)

    #Same as above, but this also accumulates a weight list along with the global time of each weight
    def OperatorAction2(self,LoopIn,Reverse = False):
        WeightList = []
        if not Reverse:
            WeightList.append([self.GetWeightTotal(LoopIn),0])
            for i in range(len(self.WeightOperatorList)):
                self.WeightOperatorList[i].Update(LoopIn)
                WeightList.append([self.GetWeightTotal(LoopIn),self.WeightOperatorList[i].time])
        else:
            finaltime = math.ceil(self.WeightOperatorList[i].time)
            WeightList.append([self.GetWeightTotal(LoopIn),finaltime])
            for i in range(len(self.WeightOperatorList)-1,-1,-1):
                self.WeightOperatorList[i].Update(LoopIn,Reverse)
                WeightList.append([self.GetWeightTotal(LoopIn),self.WeightOperatorList[i].time])
            
        return WeightList  
    
    #Same as above, but this only returns a weight at the end of a time-step (and doesn't return the time)
    #this is most useful for producing a list that we can directly tie to an external list of times.  This is what we need for extracting the topological entropy
    def OperatorAction3(self,LoopIn, Reverse = False):
        WeightList = []
        if not Reverse:
            prevtime = 0
            for i in range(len(self.WeightOperatorList)):
                thistime = math.ceil(self.WeightOperatorList[i].time)
                if thistime > prevtime:
                    prevtime = thistime
                    currentweight = LoopIn.GetWeightTotal()
                    while len(WeightList) < thistime:
                        WeightList.append(currentweight)
                self.WeightOperatorList[i].Update(LoopIn)
            WeightList.append(LoopIn.GetWeightTotal())         
        else:
            endtime = math.ceil(self.WeightOperatorList[-1].time) 
            prevtime = endtime
            for i in range(len(self.WeightOperatorList)-1,-1,-1):
                thistime = math.floor(self.WeightOperatorList[i].time)
                if thistime < prevtime:
                    prevtime = thistime
                    currentweight = LoopIn.GetWeightTotal()
                    while len(WeightList) < endtime-thistime:
                        WeightList.append(currentweight)
                self.WeightOperatorList[i].Update(LoopIn,Reverse)            
            WeightList.append(LoopIn.GetWeightTotal())            
        return WeightList
    
    
      #this takes in a given loop, and adds on a random perturbation (just to the non-zero weights), such that all the triangle inequalities are still obeyed (and thus the weights represent a valid loop)
    def LoopPerturbation(self,LoopIn,delta = 0.1):
        EdgeDone = [False for i in range(self.totalnumedges)]
        for i in range(len(self.simplist)):
            simp = self.simplist[i]
            for j in range(3):
                if not EdgeDone[simp.edgeids[j]]:
                    if LoopIn.weightlist[simp.edgeids[j]] > 0.0:
                        ptid = simp.points[j]
                        simpadj = simp.simplices[j]
                        adjopptlocid = (simpadj.LocalID(simp.points[(j+1)%3])+1)%3
                        #edge id convention [center, rightlower, rightupper, leftlower, leftupper]
                        EdgeIDs = [simp.edgeids[j],simp.edgeids[(j+1)%3],simp.edgeids[(j+2)%3], simpadj.edgeids[(adjopptlocid+2)%3], simpadj.edgeids[(adjopptlocid+1)%3]]
                        #weights with the same convention
                        LocalWeights = [LoopIn.weightlist[eid] for eid in EdgeIDs]
                        epmaxr = LocalWeights[1]+LocalWeights[2]-LocalWeights[0]
                        epminr = abs(LocalWeights[1]-LocalWeights[2])-LocalWeights[0]
                        epmaxl = LocalWeights[3]+LocalWeights[4]-LocalWeights[0]
                        epminl = abs(LocalWeights[3]-LocalWeights[4])-LocalWeights[0]
                        epmax = min(epmaxr,epmaxl)
                        epmin = max(epminr,epminl)
                        randpert = np.random.uniform(epmin*delta,epmax*delta)
                        LoopIn.weightlist[simp.edgeids[j]] += randpert
                        
                    
                    EdgeDone[simp.edgeids[j]] = True
                
                
                #self.points, self.simplices,self.edgeids
    
    
    
    #self.TopTimeOpGroups
    #this takes the self.WeightOperatorList and filters it into time ordered groups.  The elements of each group are mutually time-commuting with eachother.  Every operator is placed in (pulled through to) the lowest index group such that this is true.  We remove pairs of operators that are equivalent ... so the total number of operators can be reduced.  This is not accounting for higher order reductions in the operator set, so this is not proven to be minimal in length.  But this is a good start at estimating the topological time.
#def WOcommute(WO1,WO2), and  def WOequiv(WO1,WO2):    
    
    def OpListToTopTime(self):
        self.TopTimeOpGroups = []
        self.TopTimeOpGroups.append([self.WeightOperatorList[0]])  #put the first operator in the first group
        annihilationEvents = 0
        for i in range(1,len(self.WeightOperatorList)):
            addindex = 0
            annihilate = False
            annihilateindex = 0
            for j in range(len(self.TopTimeOpGroups)-1,-1,-1):
                commwithall = True
                for k in range(len(self.TopTimeOpGroups[j])):
                    
                    if not WOcommute(self.WeightOperatorList[i],self.TopTimeOpGroups[j][k]):
                        commwithall = False
                        if WOequiv(self.WeightOperatorList[i],self.TopTimeOpGroups[j][k]):
                            annihilate = True
                            annihilateindex = j
                            del self.TopTimeOpGroups[j][k]
                            annihilationEvents += 1
                        else:
                            addindex = j+1
                        break
                if not commwithall:
                    break
            if not annihilate:
                if addindex > len(self.TopTimeOpGroups)-1:
                    self.TopTimeOpGroups.append([self.WeightOperatorList[i]])
                else:
                    self.TopTimeOpGroups[addindex].append(self.WeightOperatorList[i])
            else:
                if len(self.TopTimeOpGroups[annihilateindex]) == 0:
                    del self.TopTimeOpGroups[annihilateindex]
                    
        print("Length of TopTimeOpGroups = ", len(self.TopTimeOpGroups), ", and number of annihilations = ", annihilationEvents)
        return len(self.TopTimeOpGroups)
        
        #we still need to check that this can't be combed tighter after any annihilations, but let's first check to see that this works.
                
    #this takes the operators in TopTimeOpGroups and refills the WeightOperatorList with it       
    def RefillOpList(self):
        if not len(self.TopTimeOpGroups) == 0:
            self.WeightOperatorList = []
            for i in range(len(self.TopTimeOpGroups)):
                for j in range(len(self.TopTimeOpGroups[i])):
                    self.WeightOperatorList.append(self.TopTimeOpGroups[i][j])    
    
    
    
    
    
    ##******Some new functions to help with the treatment of evolving simplices************************
    
    #Function that takes all of the current and future positions and gets the copy of the FD that the future positions are in.
    #Because the positions are constrained to be in the FD, we must find the copy (9 options) that has the smallest distance between the inital and proposed final positions.
    def GetNewPosCopyLoc(self):
        copyloc = []
        for i in range(len(self.pointposfuture)):
            posi = [self.pointpos[i][0],self.pointpos[i][1]]
            posf = [self.pointposfuture[i][0],self.pointposfuture[i][1]]
            distvals = []
            for j in range(len(self.dpindices)):
                posfn = [posf[k]+self.FDsizes[k]*self.dpindices[j][k] for k in range(2)]
                distvals.append([((posfn[0]-posi[0])**2+(posfn[1]-posi[1])**2)**(0.5),j])
            distvals.sort(key=itemgetter(0))
                
            copyloc.append(distvals[0][1])
        return copyloc
    
    #We need to find, for each point, the time(s) it crosses any of the lines and which line it crosses
    def GetPtCrossList(self):
        #first get the list of future pos copy locations
        CrossList = []
        Dx, Dy = self.FDsizes
        Lines = [[[0,-Dy],[0,2*Dy]],[[Dx,-Dy],[Dx,2*Dy]],[[-Dx,0],[2*Dx,0]],[[-Dx,Dy],[2*Dx,Dy]]]
        movedir = [[-1,0],[1,0],[0,-1],[0,1]]
        for i in range(len(self.pointposfuture)):
            if not self.FuturePosRelCopyLoc[i] == 0:
                posi = [self.pointpos[i][0],self.pointpos[i][1]]
                posf = [self.pointposfuture[i][0],self.pointposfuture[i][1]]
                posfn = [posf[k]+self.FDsizes[k]*self.dpindices[self.FuturePosRelCopyLoc[i]][k] for k in range(2)]
                newline = [posi,posfn]
                for j in range(len(Lines)):
                    IsInt = IsIntersection(newline,Lines[j],timeinfo = True)
                    if IsInt[0]:
                        CrossList.append([i,IsInt[1],movedir[j]])  
        CrossList.sort(key=itemgetter(1),reverse=True)    
        return CrossList
        #returns the point index, the time of crossing, and the move direction (-1,0,1 for both the x and y directions)
    
    
    #gets the current position of a given point (pass in a point index) by taking the linear interpolation from the initial position to the final postion, then moding by the boudary size so that the point is in the FD.  timeIn is in [0,1]
    def GetCurrentLoc(self,PtInd,timeIn, mod = True):
        posi = [self.pointpos[PtInd][0],self.pointpos[PtInd][1]]
        posf = [self.pointposfuture[PtInd][0],self.pointposfuture[PtInd][1]]
        posfn = [posf[k]+self.FDsizes[k]*self.dpindices[self.FuturePosRelCopyLoc[PtInd]][k] for k in range(2)]
        if mod:
            return [((posfn[k]-posi[k])*timeIn+posi[k])%self.FDsizes[k] for k in range(2)]
        else:
            return [((posfn[k]-posi[k])*timeIn+posi[k]) for k in range(2)]

    
    

    
    
    
    ##********************************************************************************************************************
    
    #This returns a list of the positions for each point of the given simplex (in the same order)
    def GetSimpPtPos(self,SimpIn):
        SimpPtPos = []
        for i in range(0,len(SimpIn.points)):
            SimpPtPos.append([self.pointpos[SimpIn.points[i]][0],self.pointpos[SimpIn.points[i]][1]])
        return SimpPtPos   

    #This returns a list of positions (in the future/next iteratation configuration of the points) for each point of the given simplex (in the same order)
    def GetSimpPtPosFuture(self,SimpIn):
        SimpPtPos = []
        for i in range(0,len(SimpIn.points)):
            SimpPtPos.append([self.pointposfuture[SimpIn.points[i]][0],self.pointposfuture[SimpIn.points[i]][1]])
        return SimpPtPos
        
    
    #triangulation2D method to calculate the area of a given simplex (SimpIn)
    def SimpArea(self,SimpIn):
        ptlist = self.GetSimpPtPos(SimpIn)
        return self.TriArea(ptlist)
    
    
    #returns the signed area of the given pointlist (3 points)
    def TriArea(self,ptlist):
        return 0.5*((ptlist[1][0]-ptlist[0][0])*(ptlist[2][1]-ptlist[0][1]) - (ptlist[1][1]-ptlist[0][1])*(ptlist[2][0]-ptlist[0][0]))
    
    
    #returns the angle (radians) between the two edges defined by the ordered three points.  If the area is negative, then the angle is given as 2*pi-angle.
    def TriAngle(self,ptlist):
        dp = (ptlist[0][0]-ptlist[1][0])*(ptlist[2][0]-ptlist[1][0]) + (ptlist[0][1]-ptlist[1][1])*(ptlist[2][1]-ptlist[1][1])
        mag1 = math.sqrt((ptlist[0][0]-ptlist[1][0])*(ptlist[0][0]-ptlist[1][0]) + (ptlist[0][1]-ptlist[1][1])*(ptlist[0][1]-ptlist[1][1]))
        mag2 = math.sqrt((ptlist[2][0]-ptlist[1][0])*(ptlist[2][0]-ptlist[1][0]) + (ptlist[2][1]-ptlist[1][1])*(ptlist[2][1]-ptlist[1][1]))
        angle = math.acos(dp/(mag1*mag2))
        if self.TriArea(ptlist) < 0:
            angle = 2*math.pi - angle
        return angle
    
    
    #This function outputs true if the given simplex has the correct orientation (i.e. correct permutation ... signed area is positive)
    def OrderCorrect(self,SimpIn):
        Ocorrect = False
        if self.SimpArea(SimpIn) >= 0:
            Ocorrect = True
        return Ocorrect
    
    
    #This function will load new point positions into pointposfuture
    def LoadNewPos(self,ptlist):
        self.pointposfuture = ptlist
    
  
    #This updates the current point positions to those stored in pointposfuture.  This will be called at the end of the evolve method
    def UpdatePtPos(self):
        self.pointpos = self.pointposfuture
               
    
    #function that takes in a point list, gets the simplices around each point, then returns the merged list
    #can't be stand-alone, since it accesses the pointlist
    def PtListToSimpList(self,PtListIn):
        slist = []
        for i in range(0,len(PtListIn)):
            slist.append(self.pointlist[PtListIn[i]].SimpNeighbors(PtListIn[i]))      
        outlist = slist[0]
        for i in range(1,len(slist)):
            outlist = ListMerge(outlist,slist[i])
        return outlist
    
    
    #this returns a list of all pairs of points that are adjacent in the current triangulation (share an edge)
    def GetAdjacentPoints(self):
        PointPairs = [[] for i in range(self.totalnumedges)]
        for i in range(len(self.simplist)):
            thissimp = self.simplist[i]
            for j in range(3):
                edgeid = thissimp.edgeids[j]
                ptpairs = [thissimp.points[(j+1)%3],thissimp.points[(j+2)%3]]
                if len(PointPairs[edgeid]) == 0:
                    PointPairs[edgeid] = ptpairs
        return PointPairs
            
          

  
    
    
    #Need to be careful with the points/lines/triangles that span the boundary
    #just plots the underlying triangulation
    def TriangulationPlot(self,filename = None, Bounds = None, FigureSize = None, labels = False):
        xpoints = [x[0] for x in self.pointpos]
        ypoints = [x[1] for x in self.pointpos]
        trianglesIn = [x.points for x in self.simplist if x.relptregion == [[0,0],[0,0],[0,0]]]
        
        
        szx = 14
        szy = 14
        if not FigureSize is None:
            szx = FigureSize[0]
            szy = FigureSize[1]
        mpl.rcParams['savefig.pad_inches'] = 0
        plt.figure(figsize=(szx,szy),dpi=300,frameon=False)
        plt.autoscale(tight=True)
        ax = plt.gca()
        if not Bounds is None:
            ax.set_xlim((Bounds[0][0], Bounds[1][0]))
            ax.set_ylim((Bounds[0][1], Bounds[1][1]))
        else:
            ax.set_xlim(0,self.FDsizes[0])
            ax.set_ylim(0,self.FDsizes[1])
        ax.set_aspect('equal')
        plt.tick_params(axis='x', which='both', bottom=False,top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False,right=False, labelleft=False)
        
        if labels:
            #plot the 
            for k in range(len(xpoints)):
                ax.annotate(k,(xpoints[k],ypoints[k]))
        
        
        plt.triplot(xpoints, ypoints, trianglesIn, 'g-', lw=0.5,zorder=1)
        
        
        
        EdgeSimp = [x for x in self.simplist if not x.relptregion == [[0,0],[0,0],[0,0]]]
        for Simp in EdgeSimp:
            for i in range(3):
                xpts = [self.pointpos[ind][0] for ind in Simp.points]
                ypts = [self.pointpos[ind][1] for ind in Simp.points]
                ch1 = Simp.relptregion[(i+2)%3]
                xpts[(i+1)%3] += ch1[0]*self.FDsizes[0]
                ypts[(i+1)%3] += ch1[1]*self.FDsizes[1]
                ch2 = Simp.relptregion[(i+1)%3]
                xpts[(i+2)%3] -= ch2[0]*self.FDsizes[0]
                ypts[(i+2)%3] -= ch2[1]*self.FDsizes[1]
                plt.triplot(xpts,ypts,[[0,1,2]],'g-', lw=0.4,zorder=1)
        plt.scatter(xpoints,ypoints,marker='.', c='k',zorder=2)
        
        plt.xlim(0,self.FDsizes[0])
        plt.ylim(0,self.FDsizes[1])
        
        #This still misses one of the four corner copies, and any other triangles that have an edge passing though the FD without having a point in the FD
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close() 
        #will eventually want to set some constant bounds to make a video possible    
        
        
    #this creates a plot of the background triangulation and in a separate color, the traintracks
    ##needs some modification
    def TrainTracksPlot(self, LoopIn, filename = None):
        xpoints = [x[0] for x in self.pointpos[:len(self.pointpos)-self.extranum]]  #note that we exclude the bounding points
        ypoints = [x[1] for x in self.pointpos[:len(self.pointpos)-self.extranum]]
        triangles = [x.points for x in self.simplist if (len(set(x.points).intersection([(len(self.pointpos)-y) for y in range(1,self.extranum+1)])) == 0)]  #make sure that the list of triangles (triplets of points) do not include the excluded large triangle points
        plt.figure(figsize=(14,14))
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.triplot(xpoints, ypoints, triangles, 'g-', lw=0.3,zorder=1)
        ###Now go through each Triangle and plot the traintrack curves
        for simp in self.simplist:
            if not None in simp.simplices:

                #get the triangle centers
                pt0 = self.pointpos[simp.points[0]]
                pt1 = self.pointpos[simp.points[1]]
                pt2 = self.pointpos[simp.points[2]]
                pt0ac = self.pointpos[simp.simplices[0].points[(simp.simplices[0].LocalID(simp.points[1])+1)%3]]
                pt1ac = self.pointpos[simp.simplices[1].points[(simp.simplices[1].LocalID(simp.points[2])+1)%3]]
                pt2ac = self.pointpos[simp.simplices[2].points[(simp.simplices[2].LocalID(simp.points[0])+1)%3]]
                
                Triselfcenter = self.GetCenter([pt0,pt1,pt2])
                Tri0center = self.GetCenter([pt1,pt0ac,pt2])
                Tri1center = self.GetCenter([pt2,pt1ac,pt0])
                Tri2center = self.GetCenter([pt0,pt2ac,pt1])
                #Now find the intersection along the edges
                pt0edge = self.GetIntersection([Triselfcenter,Tri0center],[pt1,pt2])
                pt1edge = self.GetIntersection([Triselfcenter,Tri1center],[pt0,pt2])
                pt2edge = self.GetIntersection([Triselfcenter,Tri2center],[pt0,pt1])
                
                hpt0 = self.GetCenter([Triselfcenter,pt0edge])
                hpt1 = self.GetCenter([Triselfcenter,pt1edge])
                hpt2 = self.GetCenter([Triselfcenter,pt2edge])
                
                W0 = LoopIn.weightlist[simp.edgeids[0]]
                W1 = LoopIn.weightlist[simp.edgeids[1]]
                W2 = LoopIn.weightlist[simp.edgeids[2]]
                W0p = (W1+W2-W0)/2
                W1p = (W0+W2-W1)/2
                W2p = (W1+W0-W2)/2
                
                if W0 > 0:
                    ax.add_patch(self.MyLine([pt0edge,hpt0]))
                if W1 > 0:
                    ax.add_patch(self.MyLine([pt1edge,hpt1]))
                if W2 > 0:
                    ax.add_patch(self.MyLine([pt2edge,hpt2]))
                if W0p > 0:
                    ax.add_patch(self.MyBez(hpt1,Triselfcenter,hpt2))
                if W1p > 0:
                    ax.add_patch(self.MyBez(hpt2,Triselfcenter,hpt0))
                if W2p > 0:
                    ax.add_patch(self.MyBez(hpt1,Triselfcenter,hpt0))
        
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()    

        
    #this creates a plot of the background triangulation and in a separate color, the traintracks
    def TrainTracksPlot2(self, LoopIn, filename = None, Bounds = None, FigureSize = None):
        xpoints = [x[0] for x in self.pointpos[:len(self.pointpos)-self.extranum]]  #note that we exclude the bounding points
        ypoints = [x[1] for x in self.pointpos[:len(self.pointpos)-self.extranum]]
        triangles = [x.points for x in self.simplist if (len(set(x.points).intersection([(len(self.pointpos)-y) for y in range(1,self.extranum+1)])) == 0)]  #make sure that the list of triangles (triplets of points) do not include the excluded large triangle points
        szx = 14
        szy = 14
        if not FigureSize is None:
            szx = FigureSize[0]
            szy = FigureSize[1]
        plt.figure(figsize=(14,14))
        ax = plt.gca()
        if not Bounds is None:
            ax.set_xlim((Bounds[0][0], Bounds[1][0]))
            ax.set_ylim((Bounds[0][1], Bounds[1][1]))
        ax.set_aspect('equal')
        plt.triplot(xpoints, ypoints, triangles, 'g-', lw=0.3)
        ###Now go through each Triangle and plot the traintrack curves
        for simp in self.simplist:
            W0 = LoopIn.weightlist[simp.edgeids[0]]
            W1 = LoopIn.weightlist[simp.edgeids[1]]
            W2 = LoopIn.weightlist[simp.edgeids[2]]

            if (W0+W1+W2) > 0:
                #get the triangle centers
                pt0 = self.pointpos[simp.points[0]]
                pt1 = self.pointpos[simp.points[1]]
                pt2 = self.pointpos[simp.points[2]]
                Triselfcenter = self.GetCenter([pt0,pt1,pt2])
                #Now find the halfwaypoint along the edges
                pt0edge = self.GetCenter([pt1,pt2])
                pt1edge = self.GetCenter([pt0,pt2])
                pt2edge = self.GetCenter([pt0,pt1])
                #and the points halfway from the center to the center of the edges
                hpt0 = self.GetCenter([Triselfcenter,pt0edge])
                hpt1 = self.GetCenter([Triselfcenter,pt1edge])
                hpt2 = self.GetCenter([Triselfcenter,pt2edge])
                pt0ac = pt1ac = pt2ac = Tri0center = Tri1center = Tri2center = hpt0ac = hpt1ac = hpt2ac = [0,0]
                
                if W0 > 0:
                    pt0ac = self.pointpos[simp.simplices[0].points[(simp.simplices[0].LocalID(simp.points[1])+1)%3]]
                    Tri0center = self.GetCenter([pt1,pt0ac,pt2])
                    hpt0ac = self.GetCenter([Tri0center,pt0edge])
                if W1 > 0:
                    pt1ac = self.pointpos[simp.simplices[1].points[(simp.simplices[1].LocalID(simp.points[2])+1)%3]]
                    Tri1center = self.GetCenter([pt2,pt1ac,pt0])
                    hpt1ac = self.GetCenter([Tri1center,pt1edge])
                if W2 > 0:
                    pt2ac = self.pointpos[simp.simplices[2].points[(simp.simplices[2].LocalID(simp.points[0])+1)%3]]
                    Tri2center = self.GetCenter([pt0,pt2ac,pt1])
                    hpt2ac = self.GetCenter([Tri2center,pt2edge])
                
                
                #now we neet to see if the left or right points for each edge are in the control triangle
                #if so, we will change the pertinent control points
                #first for side 0
                if W0 > 0:
                    #test Left then right
                    if (hpt0ac[0] - hpt0[0])*(pt2[1] - hpt0[1]) - (hpt0ac[1] - hpt0[1])*(pt2[0] - hpt0[0]) < 0:
                        snum = ((Tri0center[1] - Triselfcenter[1])*(pt2[0] - pt0edge[0]) - (Tri0center[0] - Triselfcenter[0])*(pt2[1] - pt0edge[1]))
                        sdenom = ((Tri0center[0] - Triselfcenter[0])*(pt0edge[1] - Triselfcenter[1]) - (Tri0center[1] - Triselfcenter[1])*(pt0edge[0] - Triselfcenter[0]))
                        S = snum/sdenom
                        hpt0ac = [(Tri0center[0]-pt0edge[0])*S+pt0edge[0],(Tri0center[1]-pt0edge[1])*S+pt0edge[1]]
                        hpt0 = [(Triselfcenter[0]-pt0edge[0])*S+pt0edge[0],(Triselfcenter[1]-pt0edge[1])*S+pt0edge[1]]
                    elif (hpt0ac[0] - hpt0[0])*(pt1[1] - hpt0[1]) - (hpt0ac[1] - hpt0[1])*(pt1[0] - hpt0[0]) > 0:
                        snum = ((Tri0center[1] - Triselfcenter[1])*(pt1[0] - pt0edge[0]) - (Tri0center[0] - Triselfcenter[0])*(pt1[1] - pt0edge[1]))
                        sdenom = ((Tri0center[0] - Triselfcenter[0])*(pt0edge[1] - Triselfcenter[1]) - (Tri0center[1] - Triselfcenter[1])*(pt0edge[0] - Triselfcenter[0]))
                        S = snum/sdenom
                        hpt0ac = [(Tri0center[0]-pt0edge[0])*S+pt0edge[0],(Tri0center[1]-pt0edge[1])*S+pt0edge[1]]
                        hpt0 = [(Triselfcenter[0]-pt0edge[0])*S+pt0edge[0],(Triselfcenter[1]-pt0edge[1])*S+pt0edge[1]]
                if W1 > 0:
                    #test Left then right
                    if (hpt1ac[0] - hpt1[0])*(pt0[1] - hpt1[1]) - (hpt1ac[1] - hpt1[1])*(pt0[0] - hpt1[0]) < 0:
                        snum = ((Tri1center[1] - Triselfcenter[1])*(pt0[0] - pt1edge[0]) - (Tri1center[0] - Triselfcenter[0])*(pt0[1] - pt1edge[1]))
                        sdenom = ((Tri1center[0] - Triselfcenter[0])*(pt1edge[1] - Triselfcenter[1]) - (Tri1center[1] - Triselfcenter[1])*(pt1edge[0] - Triselfcenter[0]))
                        S = snum/sdenom
                        hpt1ac = [(Tri1center[0]-pt1edge[0])*S+pt1edge[0],(Tri1center[1]-pt1edge[1])*S+pt1edge[1]]
                        hpt1 = [(Triselfcenter[0]-pt1edge[0])*S+pt1edge[0],(Triselfcenter[1]-pt1edge[1])*S+pt1edge[1]]
                    elif (hpt1ac[0] - hpt1[0])*(pt2[1] - hpt1[1]) - (hpt1ac[1] - hpt1[1])*(pt2[0] - hpt1[0]) > 0:
                        snum = ((Tri1center[1] - Triselfcenter[1])*(pt2[0] - pt1edge[0]) - (Tri1center[0] - Triselfcenter[0])*(pt2[1] - pt1edge[1]))
                        sdenom = ((Tri1center[0] - Triselfcenter[0])*(pt1edge[1] - Triselfcenter[1]) - (Tri1center[1] - Triselfcenter[1])*(pt1edge[0] - Triselfcenter[0]))
                        S = snum/sdenom
                        hpt1ac = [(Tri1center[0]-pt1edge[0])*S+pt1edge[0],(Tri1center[1]-pt1edge[1])*S+pt1edge[1]]
                        hpt1 = [(Triselfcenter[0]-pt1edge[0])*S+pt1edge[0],(Triselfcenter[1]-pt1edge[1])*S+pt1edge[1]]        
                if W2 > 0:
                    #test Left then right
                    if (hpt2ac[0] - hpt2[0])*(pt1[1] - hpt2[1]) - (hpt2ac[1] - hpt2[1])*(pt1[0] - hpt2[0]) < 0:
                        snum = ((Tri2center[1] - Triselfcenter[1])*(pt1[0] - pt2edge[0]) - (Tri2center[0] - Triselfcenter[0])*(pt1[1] - pt2edge[1]))
                        sdenom = ((Tri2center[0] - Triselfcenter[0])*(pt2edge[1] - Triselfcenter[1]) - (Tri2center[1] - Triselfcenter[1])*(pt2edge[0] - Triselfcenter[0]))
                        S = snum/sdenom
                        hpt2ac = [(Tri2center[0]-pt2edge[0])*S+pt2edge[0],(Tri2center[1]-pt2edge[1])*S+pt2edge[1]]
                        hpt2 = [(Triselfcenter[0]-pt2edge[0])*S+pt2edge[0],(Triselfcenter[1]-pt2edge[1])*S+pt2edge[1]]
                    elif (hpt2ac[0] - hpt2[0])*(pt0[1] - hpt2[1]) - (hpt2ac[1] - hpt2[1])*(pt0[0] - hpt2[0]) > 0:
                        snum = ((Tri2center[1] - Triselfcenter[1])*(pt0[0] - pt2edge[0]) - (Tri2center[0] - Triselfcenter[0])*(pt0[1] - pt2edge[1]))
                        sdenom = ((Tri2center[0] - Triselfcenter[0])*(pt2edge[1] - Triselfcenter[1]) - (Tri2center[1] - Triselfcenter[1])*(pt2edge[0] - Triselfcenter[0]))
                        S = snum/sdenom
                        hpt2ac = [(Tri2center[0]-pt2edge[0])*S+pt2edge[0],(Tri2center[1]-pt2edge[1])*S+pt2edge[1]]
                        hpt2 = [(Triselfcenter[0]-pt2edge[0])*S+pt2edge[0],(Triselfcenter[1]-pt2edge[1])*S+pt2edge[1]]                
                

                W0p = (W1+W2-W0)/2
                W1p = (W0+W2-W1)/2
                W2p = (W1+W0-W2)/2
                
                if W0 > 0:
                    ax.add_patch(self.MyBez(hpt0,pt0edge,hpt0ac))
                if W1 > 0:
                    ax.add_patch(self.MyBez(hpt1,pt1edge,hpt1ac))
                if W2 > 0:
                    ax.add_patch(self.MyBez(hpt2,pt2edge,hpt2ac))
                if W0p > 0:
                    ax.add_patch(self.MyBez(hpt1,Triselfcenter,hpt2))
                if W1p > 0:
                    ax.add_patch(self.MyBez(hpt2,Triselfcenter,hpt0))
                if W2p > 0:
                    ax.add_patch(self.MyBez(hpt1,Triselfcenter,hpt0))
        
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()    
        

    #this creates a plot of the background triangulation and in a separate color, the traintracks
    #This version of the train-track plot uses the dual Voronoi diagram as the skeleton (immediate connections to 
    #Voronoi vertices are curved using Bezier curves)
    #This method assumes that the triangulation is Delaunay
    
    
    def TrainTracksPlot3(self, LoopIn, filename = None, Bounds = None, FigureSize = None, markersize = None, labels = False, triplot = True, ptsplit = False, numsplit = None, linewidth = 0.5,weightlabels = False):
        xpoints = [x[0] for x in self.pointpos]
        ypoints = [x[1] for x in self.pointpos]
        triangles = [x.points for x in self.simplist if x.relptregion == [[0,0],[0,0],[0,0]]] 
        
        szx = 14
        szy = 14
        if not FigureSize is None:
            szx = FigureSize[0]
            szy = FigureSize[1]
        plt.figure(figsize=(szx,szy))
        ax = plt.gca()
        if not Bounds is None:
            ax.set_xlim((Bounds[0][0], Bounds[1][0]))
            ax.set_ylim((Bounds[0][1], Bounds[1][1]))
        else:
            ax.set_xlim(0,self.FDsizes[0])
            ax.set_ylim(0,self.FDsizes[1])
        ax.set_aspect('equal')
        plt.tick_params(axis='x', which='both', bottom=False,top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False,right=False, labelleft=False)
        if triplot:
            plt.triplot(xpoints, ypoints, triangles, 'g-', lw=0.5,zorder=1)     
            #this plots all of the triangles that are not exclusively in the FD (id edge triangles)
            EdgeSimp = [x for x in self.simplist if not x.relptregion == [[0,0],[0,0],[0,0]]]
            for Simp in EdgeSimp:
                #this plots three copies of the triangle, with each vertex (in turn) assumed to be in the FD
                #this will double plot the boundary triangles that straddle just one edge of the FD
                for i in range(3):
                    xpts = [self.pointpos[ind][0] for ind in Simp.points]
                    ypts = [self.pointpos[ind][1] for ind in Simp.points]
                    ch1 = Simp.relptregion[(i+2)%3]
                    xpts[(i+1)%3] += ch1[0]*self.FDsizes[0]
                    ypts[(i+1)%3] += ch1[1]*self.FDsizes[1]
                    ch2 = Simp.relptregion[(i+1)%3]
                    xpts[(i+2)%3] -= ch2[0]*self.FDsizes[0]
                    ypts[(i+2)%3] -= ch2[1]*self.FDsizes[1]
                    plt.triplot(xpts,ypts,[[0,1,2]],'g-', lw=0.5,zorder=1)
        
        if labels:
            #plot the labels
            for k in range(len(xpoints)):
                ax.annotate(k,(xpoints[k],ypoints[k]))

        if not ptsplit:
            if not markersize is None:
                plt.scatter(xpoints,ypoints,marker='o',s=markersize, c='k',zorder=2)
            else:
                plt.scatter(xpoints,ypoints,marker='o', c='k',zorder=2)
        else:
            if not markersize is None:
                plt.scatter(xpoints[:numsplit],ypoints[:numsplit],marker='o',s=markersize, c='k',zorder=2)
                plt.scatter(xpoints[numsplit:],ypoints[numsplit:],marker='^',s=markersize, c='b',zorder=2)
            else:
                plt.scatter(xpoints[:numsplit],ypoints[:numsplit],marker='o',s=markersize, c='k',zorder=2)
                plt.scatter(xpoints[numsplit:],ypoints[numsplit:],marker='^',s=markersize, c='b',zorder=2)
        
        epsilon = 0.00001
        ###Now go through each Triangle and plot the traintrack curves
        for simp in self.simplist:
            W0 = LoopIn.weightlist[simp.edgeids[0]]
            W1 = LoopIn.weightlist[simp.edgeids[1]]
            W2 = LoopIn.weightlist[simp.edgeids[2]]

            if (W0+W1+W2) > epsilon:
                if simp.relptregion == [[0,0],[0,0],[0,0]]:  #no need to shift coordinates
                    #get the triangle centers
                    pt0 = [self.pointpos[simp.points[0]][k] for k in range(2)]
                    pt1 = [self.pointpos[simp.points[1]][k] for k in range(2)]
                    pt2 = [self.pointpos[simp.points[2]][k] for k in range(2)]

                    TriselfCircCenter = self.GetCircumCircleCenter([pt0,pt1,pt2])
                    #Now find the neighboring Voronoi centers
                    NSimp = [simp.simplices[i] for i in range(3)]
                    NsimpPoints = [[[None,None],[None,None],[None,None]],[[None,None],[None,None],[None,None]],[[None,None],[None,None],[None,None]]]
                    for i in range(3):
                        for j in range(3):   
                            NsimpPoints[i][j][0] = self.pointpos[NSimp[i].points[j]][0]
                            NsimpPoints[i][j][1] = self.pointpos[NSimp[i].points[j]][1]
                        if not NSimp[i].relptregion == [[0,0],[0,0],[0,0]]:
                            lid = NSimp[i].LocalID(simp.points[(i+2)%3])
                            #print(i,NsimpPoints,NSimp[i].relptregion)
                            NsimpPoints[i][(lid+2)%3][0] += NSimp[i].relptregion[lid][0]*self.FDsizes[0]
                            NsimpPoints[i][(lid+2)%3][1] += NSimp[i].relptregion[lid][1]*self.FDsizes[1]

                        
                    TriNVoronoiCenters = [self.GetCircumCircleCenter(NsimpPoints[i]) for i in range(3)]
                    #and the points halfway from our Voronoi center to the neighboring Voronoi centers
                    Nhpts = [self.GetCenter([TriselfCircCenter,TriNVoronoiCenters[i]]) for i in range(3)]
                    #and again get the halfway points between these and our center point (for Bezier control)
                    Nhpts2 = [self.GetCenter([TriselfCircCenter,Nhpts[i]]) for i in range(3)]

                    ##we could choose the mimimum distance from the above to make the region where we are using Bezier curves 
                    #uniform in each direction ... will start out without this and see how it looks

                    #Now get the Bezier control points for for the lines
                    BzLinePts0 = [Nhpts[0],self.GetCenter([Nhpts[0],Nhpts2[0]]),Nhpts2[0]]
                    BzLinePts1 = [Nhpts[1],self.GetCenter([Nhpts[1],Nhpts2[1]]),Nhpts2[1]]
                    BzLinePts2 = [Nhpts[2],self.GetCenter([Nhpts[2],Nhpts2[2]]),Nhpts2[2]]

                    #This was from the previous function: now we need to see if the left or right points for each edge are in the control triangle
                    #Might impliment this again if there is trouble with the Bezier curves going the wrong way around data points
                    W0p = (W1+W2-W0)/2
                    W1p = (W0+W2-W1)/2
                    W2p = (W1+W0-W2)/2
                    
                    
                    if W0 > epsilon:
                        ax.add_patch(self.MyBez(BzLinePts0[0],BzLinePts0[1],BzLinePts0[2],linewidth))
                        if weightlabels:
                            plt.text(BzLinePts0[0][0], BzLinePts0[0][1],str(W0)[:7])
                    if W1 > epsilon:
                        ax.add_patch(self.MyBez(BzLinePts1[0],BzLinePts1[1],BzLinePts1[2],linewidth))
                        if weightlabels:
                            plt.text(BzLinePts1[0][0], BzLinePts1[0][1],str(W1)[:7])
                    if W2 > epsilon:
                        ax.add_patch(self.MyBez(BzLinePts2[0],BzLinePts2[1],BzLinePts2[2],linewidth))
                        if weightlabels:
                            plt.text(BzLinePts2[0][0], BzLinePts2[0][1],str(W2)[:7])
                    if W0p > epsilon:
                        ax.add_patch(self.MyBez(Nhpts2[1],TriselfCircCenter,Nhpts2[2],linewidth))
                    if W1p > epsilon:
                        ax.add_patch(self.MyBez(Nhpts2[2],TriselfCircCenter,Nhpts2[0],linewidth))
                    if W2p > epsilon:
                        ax.add_patch(self.MyBez(Nhpts2[0],TriselfCircCenter,Nhpts2[1],linewidth))
                            
                else: #here the triangle stradles the boundary.  We will cycle through all three points (treating each in turn as if they were in the FD)

                    for i in range(3):

                        #get the triangle centers
                        #ptset = [[self.pointpos[simp.points[s]][k] for k in range(2)] for s in range(3)]
                        ptset = [[None,None],[None,None],[None,None]]
                        for k in range(3):
                            ptset[k][0] = self.pointpos[simp.points[k]][0]
                            ptset[k][1] = self.pointpos[simp.points[k]][1]

                        #the ith point is our starting point (in the FD) ... the other two points might need to be moved
                        ptset[(i+1)%3][0] += self.FDsizes[0]*simp.relptregion[(i+2)%3][0]
                        ptset[(i+1)%3][1] += self.FDsizes[1]*simp.relptregion[(i+2)%3][1]
                        ptset[(i+2)%3][0] -= self.FDsizes[0]*simp.relptregion[(i+1)%3][0]
                        ptset[(i+2)%3][1] -= self.FDsizes[1]*simp.relptregion[(i+1)%3][1]
                       
                        #now that we have corrected for the simplex points locations, we need to do the same for the extra points on the three adjacent simplices
                        
                        
                        #print("ptset ", ptset)
                        #print("pt indices ", simp.points)
                        
                        TriselfCircCenter = self.GetCircumCircleCenter(ptset)
                        #Now find the neighboring Voronoi centers
                        
                        NSimp = [simp.simplices[k] for k in range(3)]
                        NsimpPoints = [[[None,None],[None,None],[None,None]],[[None,None],[None,None],[None,None]],[[None,None],[None,None],[None,None]]]
                        NsimpPoints[i][0][0] = ptset[(i+2)%3][0]
                        NsimpPoints[i][0][1] = ptset[(i+2)%3][1]
                        NsimpPoints[i][1][0] = ptset[(i+1)%3][0]
                        NsimpPoints[i][1][1] = ptset[(i+1)%3][1]
                        locid = NSimp[i].LocalID(simp.points[(i+2)%3])
                        initpoint = [None,None]
                        initpoint[0] = self.pointpos[NSimp[i].points[(locid+2)%3]][0]
                        initpoint[1] = self.pointpos[NSimp[i].points[(locid+2)%3]][1]
                        disp = [-1*self.FDsizes[k]*(simp.relptregion[(i+1)%3][k] + NSimp[i].relptregion[(locid+1)%3][k]) for k in range(2)] 
                        NsimpPoints[i][2][0] = initpoint[0]+disp[0]
                        NsimpPoints[i][2][1] = initpoint[1]+disp[1]
                        #first outer point done
                        NsimpPoints[(i+1)%3][0][0] = ptset[i][0]
                        NsimpPoints[(i+1)%3][0][1] = ptset[i][1]
                        NsimpPoints[(i+1)%3][1][0] = ptset[(i+2)%3][0]
                        NsimpPoints[(i+1)%3][1][1] = ptset[(i+2)%3][1]
                        locid = NSimp[(i+1)%3].LocalID(simp.points[i])
                        initpoint = [None,None]
                        initpoint[0] = self.pointpos[NSimp[(i+1)%3].points[(locid+2)%3]][0]
                        initpoint[1] = self.pointpos[NSimp[(i+1)%3].points[(locid+2)%3]][1]
                        disp = [-1*self.FDsizes[k]*NSimp[(i+1)%3].relptregion[(locid+1)%3][k] for k in range(2)] 
                        NsimpPoints[(i+1)%3][2][0] = initpoint[0]+disp[0]
                        NsimpPoints[(i+1)%3][2][1] = initpoint[1]+disp[1]
                        #second outer point done
                        NsimpPoints[(i+2)%3][0][0] = ptset[(i+1)%3][0]
                        NsimpPoints[(i+2)%3][0][1] = ptset[(i+1)%3][1]
                        NsimpPoints[(i+2)%3][1][0] = ptset[i][0]
                        NsimpPoints[(i+2)%3][1][1] = ptset[i][1]
                        locid = NSimp[(i+2)%3].LocalID(simp.points[(i+1)%3])
                        initpoint = [None,None]
                        initpoint[0] = self.pointpos[NSimp[(i+2)%3].points[(locid+2)%3]][0]
                        initpoint[1] = self.pointpos[NSimp[(i+2)%3].points[(locid+2)%3]][1]
                        disp = [self.FDsizes[k]*NSimp[(i+2)%3].relptregion[locid][k] for k in range(2)] 
                        NsimpPoints[(i+2)%3][2][0] = initpoint[0]+disp[0]
                        NsimpPoints[(i+2)%3][2][1] = initpoint[1]+disp[1]
                        #now the third outer point is done
                        
                        #print("NsimpPoints ", NsimpPoints)
                        
                        #NsimpPoints = [[self.pointpos[NSimp[i].points[j]] for j in range(3)] for i in range(3)]
                        TriNVoronoiCenters = [self.GetCircumCircleCenter(NsimpPoints[k]) for k in range(3)]
                        #print("TriselfCircCenter",i ,TriselfCircCenter)
 
                        #print("TriNVoronoiCenters", i, TriNVoronoiCenters)
                        
                        #and the points halfway from our Voronoi center to the neighboring Voronoi centers
                        Nhpts = [self.GetCenter([TriselfCircCenter,TriNVoronoiCenters[k]]) for k in range(3)]
                        #and again get the halfway points between these and our center point (for Bezier control)
                        Nhpts2 = [self.GetCenter([TriselfCircCenter,Nhpts[k]]) for k in range(3)]

                        ##we could choose the mimimum distance from the above to make the region where we are using Bezier curves 
                        #uniform in each direction ... will start out without this and see how it looks

                        #Now get the Bezier control points for for the lines
                        BzLinePts0 = [Nhpts[0],self.GetCenter([Nhpts[0],Nhpts2[0]]),Nhpts2[0]]
                        BzLinePts1 = [Nhpts[1],self.GetCenter([Nhpts[1],Nhpts2[1]]),Nhpts2[1]]
                        BzLinePts2 = [Nhpts[2],self.GetCenter([Nhpts[2],Nhpts2[2]]),Nhpts2[2]]

                        #This was from the previous function: now we neet to see if the left or right points for each edge are in the control triangle
                        #Might impliment this again if there is trouble with the Bezier curves going the wrong way around data points

                        W0p = (W1+W2-W0)/2
                        W1p = (W0+W2-W1)/2
                        W2p = (W1+W0-W2)/2

                        if W0 > epsilon:
                            ax.add_patch(self.MyBez(BzLinePts0[0],BzLinePts0[1],BzLinePts0[2],linewidth))
                        if W1 > epsilon:
                            ax.add_patch(self.MyBez(BzLinePts1[0],BzLinePts1[1],BzLinePts1[2],linewidth))
                        if W2 > epsilon:
                            ax.add_patch(self.MyBez(BzLinePts2[0],BzLinePts2[1],BzLinePts2[2],linewidth))
                        if W0p > epsilon:
                            ax.add_patch(self.MyBez(Nhpts2[1],TriselfCircCenter,Nhpts2[2],linewidth))
                        if W1p > epsilon:
                            ax.add_patch(self.MyBez(Nhpts2[2],TriselfCircCenter,Nhpts2[0],linewidth))
                        if W2p > epsilon:
                            ax.add_patch(self.MyBez(Nhpts2[0],TriselfCircCenter,Nhpts2[1],linewidth))

                    
                    
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()        
              
            
            

    def BezFunc(self,p1,p2,t):
        return [(1-t)*p1[i]+t*p2[i] for i in range(2)]
    
    def FlattenFunc(self,t,a=0.2,b=0.8):
        P0 = [0,0]
        P1 = [a,2/3]
        P2 = [b,2/3]
        P3 = [1,1]
        B1a = self.BezFunc(P0,P1,t)
        B2a = self.BezFunc(P1,P2,t)
        B3a = self.BezFunc(P2,P3,t)
        B1b = self.BezFunc(B1a,B2a,t)
        B2b = self.BezFunc(B2a,B3a,t)
        return self.BezFunc(B1b,B2b,t)
    
    def Flatten(self,x):
        t = 0.0001
        dt = 0.01
        y = 2/3
        FV = self.FlattenFunc(t)
        while FV[0] < x:
            t += dt
            FV = self.FlattenFunc(t)
        return FV[1]
            
            
    #this creates a plot of the background triangulation and in a separate color, the traintracks
    #This version of the train-track plot uses the dual Voronoi diagram as the skeleton (immediate connections to 
    #Voronoi vertices are curved using Bezier curves)
    #This method assumes that the triangulation is Delaunay
    #new in method 4: experimenting with the Bezier curves ... trying to make them simpler
    
    
    def TrainTracksPlot4(self, LoopIn, filename = None, Bounds = None, FigureSize = None, markersize = None, labels = False, triplot = True, ptsplit = False, numsplit = None, linewidth = 0.5,weightlabels = False):
        xpoints = [x[0] for x in self.pointpos]
        ypoints = [x[1] for x in self.pointpos]
        triangles = [x.points for x in self.simplist if x.relptregion == [[0,0],[0,0],[0,0]]] 
        
        szx = 14
        szy = 14
        if not FigureSize is None:
            szx = FigureSize[0]
            szy = FigureSize[1]
        plt.figure(figsize=(szx,szy))
        ax = plt.gca()
        if not Bounds is None:
            ax.set_xlim((Bounds[0][0], Bounds[1][0]))
            ax.set_ylim((Bounds[0][1], Bounds[1][1]))
        else:
            ax.set_xlim(0,self.FDsizes[0])
            ax.set_ylim(0,self.FDsizes[1])
        ax.set_aspect('equal')
        plt.tick_params(axis='x', which='both', bottom=False,top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False,right=False, labelleft=False)
        if triplot:
            plt.triplot(xpoints, ypoints, triangles, 'g-', lw=0.5)     
            #this plots all of the triangles that are not exclusively in the FD (id edge triangles)
            EdgeSimp = [x for x in self.simplist if not x.relptregion == [[0,0],[0,0],[0,0]]]
            for Simp in EdgeSimp:
                #this plots three copies of the triangle, with each vertex (in turn) assumed to be in the FD
                #this will double plot the boundary triangles that straddle just one edge of the FD
                for i in range(3):
                    xpts = [self.pointpos[ind][0] for ind in Simp.points]
                    ypts = [self.pointpos[ind][1] for ind in Simp.points]
                    ch1 = Simp.relptregion[(i+2)%3]
                    xpts[(i+1)%3] += ch1[0]*self.FDsizes[0]
                    ypts[(i+1)%3] += ch1[1]*self.FDsizes[1]
                    ch2 = Simp.relptregion[(i+1)%3]
                    xpts[(i+2)%3] -= ch2[0]*self.FDsizes[0]
                    ypts[(i+2)%3] -= ch2[1]*self.FDsizes[1]
                    plt.triplot(xpts,ypts,[[0,1,2]],'g-', lw=0.4)
        
        if labels:
            #plot the labels
            for k in range(len(xpoints)):
                ax.annotate(k,(xpoints[k],ypoints[k]))

        if not ptsplit:
            if not markersize is None:
                plt.scatter(xpoints,ypoints,marker='o',s=markersize, c='k')
            else:
                plt.scatter(xpoints,ypoints,marker='o', c='k')
        else:
            if not markersize is None:
                plt.scatter(xpoints[:numsplit],ypoints[:numsplit],marker='o',s=markersize, c='k')
                plt.scatter(xpoints[numsplit:],ypoints[numsplit:],marker='^',s=markersize, c='b')
            else:
                plt.scatter(xpoints[:numsplit],ypoints[:numsplit],marker='o',s=markersize, c='k')
                plt.scatter(xpoints[numsplit:],ypoints[numsplit:],marker='^',s=markersize, c='b')
        
        epsilon = 0.00001
        ###Now go through each Triangle and plot the traintrack curves
        for simp in self.simplist:
            #W0 = LoopIn.weightlist[simp.edgeids[0]]
            #W1 = LoopIn.weightlist[simp.edgeids[1]]
            #W2 = LoopIn.weightlist[simp.edgeids[2]]
            W = [LoopIn.weightlist[simp.edgeids[k]] for k in range(3)]

            if sum(W) > epsilon:
                if simp.relptregion == [[0,0],[0,0],[0,0]]:  #no need to shift coordinates
                    #get the triangle centers
                    #pt0 = [self.pointpos[simp.points[0]][k] for k in range(2)]
                    #pt1 = [self.pointpos[simp.points[1]][k] for k in range(2)]
                    #pt2 = [self.pointpos[simp.points[2]][k] for k in range(2)]
                    pts = [ [self.pointpos[simp.points[i]][k] for k in range(2)] for i in range(3)]

                    TriselfCircCenter = self.GetCircumCircleCenter(pts)
                    #Now find the neighboring Voronoi centers
                    NSimp = [simp.simplices[i] for i in range(3)]
                    NsimpPoints = [[[None,None],[None,None],[None,None]],[[None,None],[None,None],[None,None]],[[None,None],[None,None],[None,None]]]
                    for i in range(3):
                        for j in range(3):   
                            NsimpPoints[i][j][0] = self.pointpos[NSimp[i].points[j]][0]
                            NsimpPoints[i][j][1] = self.pointpos[NSimp[i].points[j]][1]
                        if not NSimp[i].relptregion == [[0,0],[0,0],[0,0]]:
                            lid = NSimp[i].LocalID(simp.points[(i+2)%3])
                            #print(i,NsimpPoints,NSimp[i].relptregion)
                            NsimpPoints[i][(lid+2)%3][0] += NSimp[i].relptregion[lid][0]*self.FDsizes[0]
                            NsimpPoints[i][(lid+2)%3][1] += NSimp[i].relptregion[lid][1]*self.FDsizes[1]

                        
                    TriNVoronoiCenters = [self.GetCircumCircleCenter(NsimpPoints[i]) for i in range(3)]
                    #and the points halfway from our Voronoi center to the neighboring Voronoi centers
                    Nhpts = [self.GetCenter([TriselfCircCenter,TriNVoronoiCenters[i]]) for i in range(3)]
                    
                    #now we need the weights for the three possible cubic Bezier curves (default weight is 2/3, which gives the usual quadratic bezier curve)
                    
                    wtval = [2/3,2/3,2/3]
                    
                    for i in range(3):
                        diffx =  TriselfCircCenter[0] - Nhpts[i][0]
                        diffy =  TriselfCircCenter[1] - Nhpts[i][1]
                        if diffx == 0:
                            sval = (Nhpts[i][0]-Nhpts[(i+1)%3][0])/(Nhpts[(i+2)%3][0]-Nhpts[(i+1)%3][0])
                        elif diffy == 0:
                            sval = (Nhpts[i][1]-Nhpts[(i+1)%3][1])/(Nhpts[(i+2)%3][1]-Nhpts[(i+1)%3][1])
                        else:
                            avx = (Nhpts[(i+1)%3][0]-Nhpts[i][0])/diffx
                            bvx = (Nhpts[(i+2)%3][0]-Nhpts[(i+1)%3][0])/diffx
                            avy = (Nhpts[(i+1)%3][1]-Nhpts[i][1])/diffy
                            bvy = (Nhpts[(i+2)%3][1]-Nhpts[(i+1)%3][1])/diffy
                            sval = (avy-avx)/(bvx-bvy)
                        dval = [Nhpts[(i+1)%3][k]+sval*(Nhpts[(i+2)%3][k]-Nhpts[(i+1)%3][k]) for k in range(2)]
                        L1 = [TriselfCircCenter[k] - Nhpts[i][k] for k in range(2)]
                        L2 = [dval[k] - TriselfCircCenter[k] for k in range(2)]
                        L1mag = (L1[0]**2+L1[1]**2)**(0.5)
                        L2mag = (L2[0]**2+L2[1]**2)**(0.5)
                        Lratio = L1mag/L2mag
                        #weightval = (2/3)**(Lratio)
                        #now let's modify this 
                        #wtval[i] = self.Flatten(weightval)
                    
                    
                    #W0p = (W1+W2-W0)/2
                    #W1p = (W0+W2-W1)/2
                    #W2p = (W1+W0-W2)/2
                    Wp = [(W[(i+1)%3]+W[(i+2)%3] - W[i])/2 for i in range(3)]
                    
                    
                    for i in range(3):
                        if Wp[i] > epsilon:
                            #hpl = [(1-wtval[i])*Nhpts[(i+1)%3][j]+wtval[i]*TriselfCircCenter[j] for j in range(2)]
                            #hpr = [(1-wtval[i])*Nhpts[(i+2)%3][j]+wtval[i]*TriselfCircCenter[j] for j in range(2)]
                            #ax.add_patch(self.MyBez2(Nhpts[(i+1)%3], hpl, hpr,Nhpts[(i+2)%3],linewidth))
                            ax.add_patch(self.MyBez(Nhpts[(i+1)%3], TriselfCircCenter, Nhpts[(i+2)%3],linewidth))
                            
                    
                    #if W0p > epsilon:
                    #    ax.add_patch(self.MyBez(Nhpts[1],TriselfCircCenter,Nhpts[2],linewidth))
                    #if W1p > epsilon:
                    #    ax.add_patch(self.MyBez(Nhpts[2],TriselfCircCenter,Nhpts[0],linewidth))
                    #if W2p > epsilon:
                    #    ax.add_patch(self.MyBez(Nhpts[0],TriselfCircCenter,Nhpts[1],linewidth))
                            
                else: #here the triangle stradles the boundary.  We will cycle through all three points (treating each in turn as if they were in the FD)

                    for i in range(3):

                        #get the triangle centers
                        #ptset = [[self.pointpos[simp.points[s]][k] for k in range(2)] for s in range(3)]
                        ptset = [[None,None],[None,None],[None,None]]
                        for k in range(3):
                            ptset[k][0] = self.pointpos[simp.points[k]][0]
                            ptset[k][1] = self.pointpos[simp.points[k]][1]

                        #the ith point is our starting point (in the FD) ... the other two points might need to be moved
                        ptset[(i+1)%3][0] += self.FDsizes[0]*simp.relptregion[(i+2)%3][0]
                        ptset[(i+1)%3][1] += self.FDsizes[1]*simp.relptregion[(i+2)%3][1]
                        ptset[(i+2)%3][0] -= self.FDsizes[0]*simp.relptregion[(i+1)%3][0]
                        ptset[(i+2)%3][1] -= self.FDsizes[1]*simp.relptregion[(i+1)%3][1]
                       
                        #now that we have corrected for the simplex points locations, we need to do the same for the extra points on the three adjacent simplices
                        
                        
                        #print("ptset ", ptset)
                        #print("pt indices ", simp.points)
                        
                        TriselfCircCenter = self.GetCircumCircleCenter(ptset)
                        #Now find the neighboring Voronoi centers
                        
                        NSimp = [simp.simplices[k] for k in range(3)]
                        NsimpPoints = [[[None,None],[None,None],[None,None]],[[None,None],[None,None],[None,None]],[[None,None],[None,None],[None,None]]]
                        NsimpPoints[i][0][0] = ptset[(i+2)%3][0]
                        NsimpPoints[i][0][1] = ptset[(i+2)%3][1]
                        NsimpPoints[i][1][0] = ptset[(i+1)%3][0]
                        NsimpPoints[i][1][1] = ptset[(i+1)%3][1]
                        locid = NSimp[i].LocalID(simp.points[(i+2)%3])
                        initpoint = [None,None]
                        initpoint[0] = self.pointpos[NSimp[i].points[(locid+2)%3]][0]
                        initpoint[1] = self.pointpos[NSimp[i].points[(locid+2)%3]][1]
                        disp = [-1*self.FDsizes[k]*(simp.relptregion[(i+1)%3][k] + NSimp[i].relptregion[(locid+1)%3][k]) for k in range(2)] 
                        NsimpPoints[i][2][0] = initpoint[0]+disp[0]
                        NsimpPoints[i][2][1] = initpoint[1]+disp[1]
                        #first outer point done
                        NsimpPoints[(i+1)%3][0][0] = ptset[i][0]
                        NsimpPoints[(i+1)%3][0][1] = ptset[i][1]
                        NsimpPoints[(i+1)%3][1][0] = ptset[(i+2)%3][0]
                        NsimpPoints[(i+1)%3][1][1] = ptset[(i+2)%3][1]
                        locid = NSimp[(i+1)%3].LocalID(simp.points[i])
                        initpoint = [None,None]
                        initpoint[0] = self.pointpos[NSimp[(i+1)%3].points[(locid+2)%3]][0]
                        initpoint[1] = self.pointpos[NSimp[(i+1)%3].points[(locid+2)%3]][1]
                        disp = [-1*self.FDsizes[k]*NSimp[(i+1)%3].relptregion[(locid+1)%3][k] for k in range(2)] 
                        NsimpPoints[(i+1)%3][2][0] = initpoint[0]+disp[0]
                        NsimpPoints[(i+1)%3][2][1] = initpoint[1]+disp[1]
                        #second outer point done
                        NsimpPoints[(i+2)%3][0][0] = ptset[(i+1)%3][0]
                        NsimpPoints[(i+2)%3][0][1] = ptset[(i+1)%3][1]
                        NsimpPoints[(i+2)%3][1][0] = ptset[i][0]
                        NsimpPoints[(i+2)%3][1][1] = ptset[i][1]
                        locid = NSimp[(i+2)%3].LocalID(simp.points[(i+1)%3])
                        initpoint = [None,None]
                        initpoint[0] = self.pointpos[NSimp[(i+2)%3].points[(locid+2)%3]][0]
                        initpoint[1] = self.pointpos[NSimp[(i+2)%3].points[(locid+2)%3]][1]
                        disp = [self.FDsizes[k]*NSimp[(i+2)%3].relptregion[locid][k] for k in range(2)] 
                        NsimpPoints[(i+2)%3][2][0] = initpoint[0]+disp[0]
                        NsimpPoints[(i+2)%3][2][1] = initpoint[1]+disp[1]
                        #now the third outer point is done
                        
                        #print("NsimpPoints ", NsimpPoints)
                        
                        #NsimpPoints = [[self.pointpos[NSimp[i].points[j]] for j in range(3)] for i in range(3)]
                        TriNVoronoiCenters = [self.GetCircumCircleCenter(NsimpPoints[k]) for k in range(3)]
                        #print("TriselfCircCenter",i ,TriselfCircCenter)
 
                        #print("TriNVoronoiCenters", i, TriNVoronoiCenters)
                        
                        #and the points halfway from our Voronoi center to the neighboring Voronoi centers
                        Nhpts = [self.GetCenter([TriselfCircCenter,TriNVoronoiCenters[k]]) for k in range(3)]
                        
                        wtval = [2/3,2/3,2/3]
                        for j in range(3):
                            diffx =  TriselfCircCenter[0] - Nhpts[j][0]
                            diffy =  TriselfCircCenter[1] - Nhpts[j][1]
                            if diffx == 0:
                                sval = (Nhpts[j][0]-Nhpts[(j+1)%3][0])/(Nhpts[(j+2)%3][0]-Nhpts[(j+1)%3][0])
                            elif diffy == 0:
                                sval = (Nhpts[j][1]-Nhpts[(j+1)%3][1])/(Nhpts[(j+2)%3][1]-Nhpts[(j+1)%3][1])
                            else:
                                avx = (Nhpts[(j+1)%3][0]-Nhpts[j][0])/diffx
                                bvx = (Nhpts[(j+2)%3][0]-Nhpts[(j+1)%3][0])/diffx
                                avy = (Nhpts[(j+1)%3][1]-Nhpts[j][1])/diffy
                                bvy = (Nhpts[(j+2)%3][1]-Nhpts[(j+1)%3][1])/diffy
                                sval = (avy-avx)/(bvx-bvy)
                            dval = [Nhpts[(j+1)%3][k]+sval*(Nhpts[(j+2)%3][k]-Nhpts[(j+1)%3][k]) for k in range(2)]
                            L1 = [TriselfCircCenter[k] - Nhpts[j][k] for k in range(2)]
                            L2 = [dval[k] - TriselfCircCenter[k] for k in range(2)]
                            L1mag = (L1[0]**2+L1[1]**2)**(0.5)
                            L2mag = (L2[0]**2+L2[1]**2)**(0.5)
                            Lratio = L1mag/L2mag
                            #weightval = (2/3)**(Lratio)
                            #wtval[j] = self.Flatten(weightval)

                        #W0p = (W1+W2-W0)/2
                        #W1p = (W0+W2-W1)/2
                        #W2p = (W1+W0-W2)/2

                        Wp = [(W[(j+1)%3]+W[(j+2)%3] - W[j])/2 for j in range(3)]
                    
                    
                        for k in range(3):
                            if Wp[k] > epsilon:
                                #hpl = [(1-wtval[k])*Nhpts[(k+1)%3][j]+wtval[k]*TriselfCircCenter[j] for j in range(2)]
                                #hpr = [(1-wtval[k])*Nhpts[(k+2)%3][j]+wtval[k]*TriselfCircCenter[j] for j in range(2)]
                                #ax.add_patch(self.MyBez2(Nhpts[(k+1)%3], hpl, hpr,Nhpts[(k+2)%3],linewidth))
                                ax.add_patch(self.MyBez(Nhpts[(k+1)%3], TriselfCircCenter, Nhpts[(k+2)%3],linewidth))
                        
                        #if W0p > epsilon:
                        #    ax.add_patch(self.MyBez(Nhpts[1],TriselfCircCenter,Nhpts[2],linewidth))
                        #if W1p > epsilon:
                        #    ax.add_patch(self.MyBez(Nhpts[2],TriselfCircCenter,Nhpts[0],linewidth))
                        #if W2p > epsilon:
                        #    ax.add_patch(self.MyBez(Nhpts[0],TriselfCircCenter,Nhpts[1],linewidth))

                    
                    
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()             
        
        
        
    #the following few functions are for plotting help
    def GetCenter(self,ObjIn):
        X = 0
        Y = 0
        for x1 in ObjIn:
            X += x1[0]
            Y += x1[1]
        X = X/len(ObjIn)
        Y = Y/len(ObjIn)
        return [X,Y]
    
    
    #this returns the coordinates of the circumcircle about the given three points
    def GetCircumCircleCenter(self,PtsIn):
        sqval = [z[0]**2+z[1]**2 for z in PtsIn]
        diffy = [PtsIn[(i+1)%3][1]-PtsIn[(i+2)%3][1] for i in range(3)]
        diffx = [-PtsIn[(i+1)%3][0]+PtsIn[(i+2)%3][0] for i in range(3)]
        D = 2*sum([PtsIn[i][0]*diffy[i] for i in range(3)])
        PtOutx = sum([sqval[i]*diffy[i] for i in range(3)])/D
        PtOuty = sum([sqval[i]*diffx[i] for i in range(3)])/D
        return [PtOutx,PtOuty]

    def MyLine(self,LineIn,linewidth = 0.5):
        return mpatches.PathPatch(mpath.Path([(LineIn[0][0], LineIn[0][1]), (LineIn[1][0], LineIn[1][1])], [mpath.Path.MOVETO,mpath.Path.LINETO]), ec="r", fc="none",lw = linewidth)

    def GetIntersection(self,Line1,Line2):
        det = (Line2[1][1]-Line2[0][1])*(Line1[1][0]-Line1[0][0])-(Line1[1][1]-Line1[0][1])*(Line2[1][0]-Line2[0][0])
        bx = (Line1[0][1]*Line1[1][0] - Line1[1][1]*Line1[0][0])
        by = (Line2[0][1]*Line2[1][0] - Line2[1][1]*Line2[0][0])
        xout = (bx*(Line2[1][0]-Line2[0][0]) - by*(Line1[1][0]-Line1[0][0]))/det
        yout = (bx*(Line2[1][1]-Line2[0][1]) - by*(Line1[1][1]-Line1[0][1]))/det
        return [xout,yout]

    def MyBez(self,P1,P2,P3,linewidth = 0.5):
        return mpatches.PathPatch(mpath.Path([(P1[0], P1[1]), (P2[0], P2[1]), (P3[0], P3[1])], [mpath.Path.MOVETO,mpath.Path.CURVE3, mpath.Path.CURVE3]), ec="r",fc="none",lw = linewidth, capstyle = 'round',joinstyle = 'round')
    
    def MyBez2(self,P1,P2,P3,P4,linewidth = 0.5):
        return mpatches.PathPatch(mpath.Path([(P1[0], P1[1]), (P2[0], P2[1]), (P3[0], P3[1]),(P4[0], P4[1])], [mpath.Path.MOVETO,mpath.Path.CURVE4, mpath.Path.CURVE4,mpath.Path.CURVE4]), ec="r",fc="none",lw = linewidth, capstyle = 'round',joinstyle = 'round')

    
    #this function returns a pair [IsSoln, timeOut], where IsSoln is a boolian that is true if one of the two times to go through zero area is between Tin and 1, and false if not.  If true, then timeOut gives this time (if two times, then this gives the smallest)
    def AreaZeroTime2(self,ptlist,Tin = 0,Verbose = False):
        #first get the beginning and end x,y coordinate for each of the three points
        Initpos = [self.pointpos[x] for x in ptlist]
        Finalpos = [self.pointposfuture[x] for x in ptlist]
        return self.BaseAreaZeroTime(Initpos, Finalpos, Tin, Verbose)

    #this is the same as above, but we are dealing with periodic boundary conditions.  We pass in the index of a simplex.
    #this is meant to be used to find the initial list of possible collapsing triangles before the rest of the evolution method
    #another function will be used to find the same info for simplices that are created in between the time-steps
    def AreaZeroTimeTorus1(self,SimpIn,Verbose = False):
        #first, by convention, we are going to take a specific copy of this simplex ... the one where the first point stored in the simplex is considered to be in the fundamental domain.  For boundary simplices, this gives us one copy to think about.
        ptlist = SimpIn.points
        rpr = SimpIn.relptregion
        pt1shift = rpr[2]
        pt2shift = [-1*rpr[1][0],-1*rpr[1][1]]
        #now shift the Initial and Final points appropriately
        Dx, Dy = self.FDsizes
        Initpos = [[self.pointpos[x][0],self.pointpos[x][1]] for x in ptlist]
        Finalpos = [[self.pointposfuture[x][0],self.pointposfuture[x][1]] for x in ptlist]
        Initpos[1][0] += Dx*pt1shift[0]
        Initpos[1][1] += Dy*pt1shift[1]
        Finalpos[1][0] += Dx*pt1shift[0]
        Finalpos[1][1] += Dy*pt1shift[1]
        Initpos[2][0] += Dx*pt2shift[0]
        Initpos[2][1] += Dy*pt2shift[1]
        Finalpos[2][0] += Dx*pt2shift[0]
        Finalpos[2][1] += Dy*pt2shift[1] 
        #Finally, we see if any of the final points have crossed a boundary, and correct for this
        cploc = [self.FuturePosRelCopyLoc[x] for x in ptlist]
        for i in range(3):
            for k in range(2):
                Finalpos[i][k] += self.FDsizes[k]*self.dpindices[cploc[i]][k]
                
        return self.BaseAreaZeroTime(Initpos, Finalpos, 0, Verbose)
    
  
    #same as above, but for simplices sometime in the middle of the evolution step
    def AreaZeroTimeTorus2(self,SimpIn,Tin = 0,Verbose = False):
        #first, by convention, we are going to take a specific copy of this simplex ... the one where the first point stored in the simplex is considered to be in the fundamental domain.  For boundary simplices, this gives us one copy to think about.
        ptlist = SimpIn.points
        #print(ptlist)
        rpr = SimpIn.relptregion
        #print(rpr)
        pt1shift = rpr[2]
        pt2shift = [-1*rpr[1][0],-1*rpr[1][1]]
        #print(pt1shift,pt2shift)
        
        #now shift the Initial and Final points appropriately
        Dx, Dy = self.FDsizes
        
        Initpos = [self.GetCurrentLoc(x,Tin) for x in ptlist]
        #print("Initpos A", Initpos)
        #Initpos = [[self.pointpos[x][0],self.pointpos[x][1]] for x in ptlist]
        Finalpos = [[self.pointposfuture[x][0],self.pointposfuture[x][1]] for x in ptlist]
        #print("Finalpos A",Finalpos)
        
        Initpos[1][0] += Dx*pt1shift[0]
        Initpos[1][1] += Dy*pt1shift[1]
        Finalpos[1][0] += Dx*pt1shift[0]
        Finalpos[1][1] += Dy*pt1shift[1]
        Initpos[2][0] += Dx*pt2shift[0]
        Initpos[2][1] += Dy*pt2shift[1]
        Finalpos[2][0] += Dx*pt2shift[0]
        Finalpos[2][1] += Dy*pt2shift[1] 
        #Finally, we see if any of the final points have crossed a boundary, and correct for this (notice that we used the updated version)
        #print("Initpos B", Initpos)
        #print("Finalpos B",Finalpos)
        cploc = [self.UpdatedFuturePosRelCopyLoc[x] for x in ptlist]
        #print("cploc = ", cploc)
        #for k in range(len(ptlist)):
        #    if (cploc[k][0] > 1 or cploc[k][0] < -1 or cploc[k][1] > 1 or cploc[k][1] < -1):
        #        print("problematic cploc is = ", cploc)
                
        for i in range(3):
            for k in range(2):
                Finalpos[i][k] += self.FDsizes[k]*cploc[i][k] 
        
        #print("Finalpos C",Finalpos)
                
        #self.FDsizes[k]*self.dpindices[j][k]
        AZT = self.BaseAreaZeroTime(Initpos, Finalpos, 0, Verbose)
        #now, the returned time is between 0 and 1, while we need a time between Tin and 1, so we uniformly contract this time
        if AZT[0]:
            newtime = AZT[1]
            AZT[1] = newtime+Tin-Tin*newtime
        return AZT


        
    #this function solves for the zeros of the area polynomial         
    def BaseAreaZeroTime(self, PtsInit, PtsFinal, Tin = 0,Verbose = False):
        ai, bi, ci = PtsInit
        af, bf, cf = PtsFinal
        a = (cf[0]-ci[0]-af[0]+ai[0])*(bf[1]-bi[1]-af[1]+ai[1]) - (cf[1]-ci[1]-af[1]+ai[1])*(bf[0]-bi[0]-af[0]+ai[0])
        b = (ci[0]-ai[0])*(bf[1]-bi[1]-af[1]+ai[1]) + (cf[0]-ci[0]-af[0]+ai[0])*(bi[1]-ai[1]) - (ci[1]-ai[1])*(bf[0]-bi[0]-af[0]+ai[0]) - (cf[1]-ci[1]-af[1]+ai[1])*(bi[0]-ai[0])
        c = (ci[0]-ai[0])*(bi[1]-ai[1]) - (ci[1]-ai[1])*(bi[0]-ai[0])
        
        IsSoln = False  #just declaring this variable
        Tout = Tin
        
        numzero = 0
        if a == 0:
            numzero += 1
        if b == 0:
            numzero += 1            
        if c == 0:
            numzero += 1        
        
        #roots = (-b +- sqrt(b**2 - 4*a*c))/(2*a)
        if numzero == 0:
            q = b**2 - 4*a*c
            if q > 0:
                #two real roots
                try:
                    t1 = 2*c/(-b-math.sqrt(q))
                except ZeroDivisionError:
                    t1 = 100  #since we are only concerned with times between Tin and 1, we set this to 100 so that this root will be ignored
                try:    
                    t2 = 2*c/(-b+math.sqrt(q))                
                except ZeroDivisionError:
                    t2 = 100                
                #t1 = (-b-math.sqrt(q))/(2*a)
                #t2 = (-b+math.sqrt(q))/(2*a)
                t1ok = False
                t2ok = False
                if t1 > Tin and t1 < 1:
                    Tout = t1
                    IsSoln = True
                    t1ok = True
                if t2 > Tin and t2 < 1:
                    IsSoln = True
                    t2ok = True
                    if t1ok:
                        if t2 < t1:
                            Tout = t2
                    else:
                        Tout = t2
                if (not t1ok) and (not t2ok) and Verbose:
                    if self.TriArea([af,bf,cf]) < 0:
                        print("times are not in the interval", Tin, " to 1 ... t1 = ", t1, ", and t2 = ", t2)
                        print("The offending points start at: ",[ai,bi,ci], ", and end up at: ", [af,bf,cf])
                        print("Initial signed area = ", self.TriArea([ai,bi,ci]), ", and Final signed area = ", self.TriArea([af,bf,cf]))
                        
            elif q == 0:
                #one real root
                t = -b/(2*a)
                if t > Tin and t < 1:
                    Tout = t
                    IsSoln = True
            #else:   #two complex solutions, nothing to do
        elif numzero == 1:
            if a == 0:
                t = -c/b
                if t > Tin and t < 1:
                    Tout = t
                    IsSoln = True
            elif b == 0 and -c/a > 0:
                t = math.sqrt(-c/a)
                if t > Tin and t < 1:
                    Tout = t
                    IsSoln = True
            else:
                #c = 0 , discarding the t = 0 soln
                t = -b/a
                if t > Tin and t < 1:
                    Tout = t
                    IsSoln = True
        #else:   #don't need to treat the cases of 2 and 3 of a,b,c being zero ... 
        return [IsSoln, Tout]
       
    
    #this returns the linearly interpolated positions of each point in ptlist (usually 3, but can handle other lengths) at the time 0 < teval < 1.
    def PtPosPart(self,ptlist,teval):
        posout = []
        for i in range(0,len(ptlist)):
            posout.append(self.GetCurrentLoc(ptlist[i],teval,False))
        return posout   

    
    def GetRelCurrentLoc(self,SimpIn,timeIn):
        ptlist = SimpIn.points
        rpr = SimpIn.relptregion
        pt1shift = rpr[2]
        pt2shift = [-1*rpr[1][0],-1*rpr[1][1]]
        #now shift the Initial and Final points appropriately
        Dx, Dy = self.FDsizes
        
        pos = [self.GetCurrentLoc(x,timeIn,True) for x in ptlist]
        pos[1][0] += Dx*pt1shift[0]
        pos[1][1] += Dy*pt1shift[1]
        pos[2][0] += Dx*pt2shift[0]
        pos[2][1] += Dy*pt2shift[1]
        return pos
    
        
    #This returns the point (internal id) that passes through its opposite edge during an area collapse event known to occur at t = tcol
    def CollapsePt(self,SimpIn,tcol):
        #first get the positions of the 3 points at the time of collapse
        colpos = self.GetRelCurrentLoc(SimpIn,tcol)
        #print("the collapse triangle positions ", colpos)
        whichpoint = 0
        d0 = (colpos[2][0] - colpos[0][0])*(colpos[1][0]-colpos[0][0]) + (colpos[2][1] - colpos[0][1])*(colpos[1][1]-colpos[0][1])  
        #print("d0 = ", d0)
        #This is the dot product of (z2-z0) and (z1-z0) ... < 0 if 0 is the middle point
        if d0 < 0:
            whichpoint = 0
        else:
            d1 = (colpos[2][0] - colpos[1][0])*(colpos[0][0]-colpos[1][0]) + (colpos[2][1] - colpos[1][1])*(colpos[0][1]-colpos[1][1])
            #print("d1 = ", d1)
            if d1 < 0:
                whichpoint = 1
            else:
                whichpoint = 2   #don't need to calculate the last dot product.  If the first two are >0, this must be <0
        #print("the collapse point index ", whichpoint)
        return whichpoint
        
           
    #This method sums up all of the weights
    def GetWeightTotal(self,LoopIn):
        return LoopIn.WeightTotal()

    
    #This methods goes through each point (actual data point), and sums up all of the weights that are diagonal on the outgoing edges.
    #This corresponds to the original Etec weight total (~ish)
    #this one under counts due to not double counting each edge contribution
    def GetRealWeightTotal0(self,LoopIn):
        bigsum = 0
        for i in range(0,len(self.pointlist)-self.extranum):
            StartSimp = self.pointlist[i]
            locptid = StartSimp.LocalID(i)
            bigsum += self.DiagWeight(StartSimp,(locptid+2)%3,LoopIn)
            NextSimp = StartSimp.simplices[(locptid+1)%3]
            while not NextSimp is StartSimp:
                locptid = NextSimp.LocalID(i)
                bigsum += self.DiagWeight(NextSimp,(locptid+2)%3,LoopIn)
                NextSimp = NextSimp.simplices[(locptid+1)%3]
        return bigsum//2
    
    #this one works better
    def GetRealWeightTotal(self,LoopIn):
        bigsum = 0
        for simp in self.simplist:
            for i in range(3):
                if not LoopIn.weightlist[simp.edgeids[i]] == 0:
                    WA = LoopIn.weightlist[simp.edgeids[(i+1)%3]]
                    WB = LoopIn.weightlist[simp.edgeids[(i+2)%3]]
                    xsimp = simp.simplices[i]
                    Lid = xsimp.LocalID(simp.points[(i+2)%3])
                    WC = LoopIn.weightlist[xsimp.edgeids[Lid]]
                    WD = LoopIn.weightlist[xsimp.edgeids[(Lid+1)%3]]
                    bigsum += abs(-WA+WB-WC+WD)//2
        return bigsum//2            
                
    #this gets the diagonal weight, given the simplex, and the local id of the point across from the desired edge.
    def DiagWeight(self,Simpin,locid,LoopIn):
        TopSimp = Simpin.simplices[locid]
        righttoplocid = TopSimp.LocalID(Simpin.points[(locid+1)%3])
        B2 = LoopIn.weightlist[Simpin.edgeids[(locid+1)%3]]

        B3 = LoopIn.weightlist[Simpin.edgeids[(locid+2)%3]]
        T2 = LoopIn.weightlist[TopSimp.edgeids[(righttoplocid+2)%3]]
        T3 = LoopIn.weightlist[TopSimp.edgeids[righttoplocid]]
        return abs(T3-T2+B3-B2)//2
            
    
    
    #This returns the two simplices that share the edge between the two points (point indices) in PairIn.  If the two points are not adjacent in the current triangulation, then this returns None
    def GetEdgePairSimp(self,PairIn):
        SL = self.pointlist[PairIn[0]].SimpNeighbors(PairIn[0])
        STM = []
        for k in range(0,len(SL)):
            if PairIn[1] in SL[k].points:
                STM.append(SL[k])
            if len(STM) == 2:
                break
        if len(STM) == 2:
            return STM
        else:
            return None
        
      
    #this determines if the given point (ptin) is to the left of line that goes from the first to second point in linepts
    #Used in determining the edges crossed in an initial rubber band
    def IsLeft(self,linepts,ptin):
        pos1 = self.pointpos[linepts[0]]
        pos2 = self.pointpos[linepts[1]]
        posA = self.pointpos[ptin]
        crossP = (pos2[0] - pos1[0])*(posA[1] - pos1[1]) - (pos2[1] - pos1[1])*(posA[0] - pos1[0])
        ToLeft = False
        if crossP >= 0:
            ToLeft = True
        return ToLeft
    
    #this returns the simplex (and local point id) that contains the first of linepoints, 
    #and has the line (to the second point) passing through it
    def SimpInDir(self,linepoints):
        pt1 = linepoints[0]
        pt2 = linepoints[1]
        StartSimp = self.pointlist[pt1]
        locpt = StartSimp.LocalID(pt1)
        ptright = StartSimp.points[(locpt+1)%3]
        ptleft = StartSimp.points[(locpt+2)%3]
        while not ((not self.IsLeft([pt1,pt2],ptright)) and self.IsLeft([pt1,pt2],ptleft)):
            StartSimp = StartSimp.simplices[(locpt+1)%3]
            locpt = StartSimp.LocalID(pt1)
            ptright = StartSimp.points[(locpt+1)%3]
            ptleft = StartSimp.points[(locpt+2)%3]
        return locpt, StartSimp
    
    
    #This takes the two points in linepoints and adds a weight of one (or non-default value) to any edges that are crossed
    #by the line.
    def AddWeightsAlongLine(self,linepoints, Boolin, LoopIn, wadd = 1.0):
        pt1 = linepoints[0]
        pt2 = linepoints[1]
        #print("linepoints: ", self.pointpos[pt1], self.pointpos[pt2])
        if Boolin[1][0]: #this is the case of adjacent points (i.e. the line between the points is an edge)
            #only if the curvelefts' (Boolin[0], Boolin[2]) are opposite one another, do we add a weight
            if Boolin[0] is not Boolin[2]:
                pt1rtlocid = Boolin[1][1][1].LocalID(pt1)
                edgeindex = Boolin[1][1][1].edgeids[(pt1rtlocid+1)%3]
                LoopIn.weightlist[edgeindex] += wadd
        else:
            #first we need to determine the direction (which simplex) to set out from that has pt1 as a point.
            stlocid, StartSimp = self.SimpInDir([pt1,pt2])
            endlocid, EndSimp = self.SimpInDir([pt2,pt1])
            if not pt2 in StartSimp.points:
                #print("Start Simp: ", self.pointpos[StartSimp.points[0]], self.pointpos[StartSimp.points[1]], self.pointpos[StartSimp.points[2]])
                #print("End Simp: ", self.pointpos[EndSimp.points[0]], self.pointpos[EndSimp.points[1]], self.pointpos[EndSimp.points[2]])
                edgeindex = StartSimp.edgeids[stlocid]
                LoopIn.weightlist[edgeindex] += wadd
                    
                leftpoint = StartSimp.points[(stlocid+2)%3]
                CurrentSimp = StartSimp.simplices[stlocid]
                leftptloc = CurrentSimp.LocalID(leftpoint)
                #print(self.pointpos[CurrentSimp.points[0]], self.pointpos[CurrentSimp.points[1]], self.pointpos[CurrentSimp.points[2]])
                while not CurrentSimp is EndSimp:
                    ptcompare = CurrentSimp.points[(leftptloc+2)%3]
                    indexadd = 0
                    if not self.IsLeft(linepoints,ptcompare):
                        indexadd = 1

                    edgeindex = CurrentSimp.edgeids[(leftptloc+indexadd)%3]
                    LoopIn.weightlist[edgeindex] += wadd
                    leftpoint = CurrentSimp.points[(leftptloc+indexadd+2)%3]
                    CurrentSimp = CurrentSimp.simplices[(leftptloc+indexadd)%3]
                    leftptloc = CurrentSimp.LocalID(leftpoint)
                    #print(self.pointpos[CurrentSimp.points[0]], self.pointpos[CurrentSimp.points[1]], self.pointpos[CurrentSimp.points[2]])
    
    #This takes the central point in pttriple and adds in the weight of wadd to each of the radial edges starting
    #from the edge that is part of the simplex bisected by pt1 and pt2, to the edge that is part of the simplex
    #bisected by pt2 and pt3
    def AddWeightsAroundPoint(self,pttriple, Boolin,LoopIn, wadd = 1.0):
        pt1 = pttriple[0]
        pt2 = pttriple[1]
        pt3 = pttriple[2]
        indadd = 1
        if not Boolin[1]: #curve right triggers this
            indadd = 2
        stlocid = None
        StartSimp = None
        if Boolin[0][0]:
            if not Boolin[1]:
                StartSimp = Boolin[0][1][0]
            else:
                StartSimp = Boolin[0][1][1]
            stlocid = StartSimp.LocalID(pt2)
        else:
            stlocid, StartSimp = self.SimpInDir([pt2,pt1])
        
        endlocid = None
        EndSimp = None
        if Boolin[2][0]:
            if not Boolin[1]:
                EndSimp = Boolin[2][1][0]
            else:
                EndSimp = Boolin[2][1][1]
            endlocid = EndSimp.LocalID(pt2)   
        else:
            endlocid, EndSimp = self.SimpInDir([pt2,pt3])

        edgeindex = StartSimp.edgeids[(stlocid+indadd)%3]
        LoopIn.weightlist[edgeindex] += wadd
        CurrentSimp = StartSimp.simplices[(stlocid+indadd)%3]
        ptloc = CurrentSimp.LocalID(pt2)
        while not CurrentSimp is EndSimp:
            edgeindex = CurrentSimp.edgeids[(ptloc+indadd)%3]
            LoopIn.weightlist[edgeindex] += wadd
            CurrentSimp = CurrentSimp.simplices[(ptloc+indadd)%3]
            ptloc = CurrentSimp.LocalID(pt2)        
    
    #This initializes the edge weights that correspond to a given rubber band (or set of bands).
    #If an EdgeWeightList is passed in, then the weights are added to the appropriate spot in this list, and are not added to the edge weights in the triangulation
    def BandWeightInitialize(self,rbands, LoopIn, wadd = 1.0):
        for i in range(len(rbands)):
            numpoints = len(rbands[i])
            AreAdjacent = []
            CurveLeft = []
            for k in range(numpoints):
                AreAdjacent.append(self.ArePointsAdjacent(rbands[i][k],rbands[i][(k+1)%numpoints]))
                triplepts = [rbands[i][(k+numpoints-1)%numpoints],rbands[i][k],rbands[i][(k+1)%numpoints]]
                CurveLeft.append(self.DoesCurveLeft(triplepts))

            for j in range(numpoints):
                Bool1 = [CurveLeft[j],AreAdjacent[j],CurveLeft[(j+1)%numpoints]]
                Bool2 = [AreAdjacent[(j+numpoints-1)%numpoints],CurveLeft[j],AreAdjacent[j]]
                triplepts = [rbands[i][(j+numpoints-1)%numpoints],rbands[i][j],rbands[i][(j+1)%numpoints]]                
                self.AddWeightsAlongLine([rbands[i][j],rbands[i][(j+1)%numpoints]],Bool1,LoopIn,wadd)
                self.AddWeightsAroundPoint(triplepts, Bool2,LoopIn,wadd)

                
    #this takes the regular edge weights (for some band) encoded in RegLoop, and uses the triangulation connectivity to create LoopIn, which represents the band in shear coordinates
    def BandShearWeightsInitialize(self, RegLoop, LoopIn):
        for simp in self.simplist:
            for i in range(3):
                #LoopIn must be initialized to all zeros (this catches the second time throough)
                if LoopIn.weightlist[simp.edgeids[i]] == 0:
                    #if the value for the regular loop is zero here, then the shear coordinates should be zero (already initialized as zero)
                    if not RegLoop.weightlist[simp.edgeids[i]] == 0:
                        WA = RegLoop.weightlist[simp.edgeids[(i+1)%3]]
                        WB = RegLoop.weightlist[simp.edgeids[(i+2)%3]]
                        xsimp = simp.simplices[i]
                        Lid = xsimp.LocalID(simp.points[(i+2)%3])
                        WC = RegLoop.weightlist[xsimp.edgeids[Lid]]
                        WD = RegLoop.weightlist[xsimp.edgeids[(Lid+1)%3]]
                        LoopIn.weightlist[simp.edgeids[i]] = (-WA+WB-WC+WD)//2

               

        
    def ArePointsAdjacent(self,pt1,pt2):
        AreAdjacent = False
        LRsimps = []
        LRsimpspos = []
        simpposcounter = 0
        StartSimp = self.pointlist[pt1]
        locid = StartSimp.LocalID(pt1)

        if pt2 in StartSimp.points:
            AreAdjacent = True
            LRsimps.append(StartSimp)
            LRsimpspos.append(simpposcounter)
        simpposcounter += 1
        
        NextSimp = StartSimp.simplices[(locid+1)%3]
        locid = NextSimp.LocalID(pt1)
        while not NextSimp is StartSimp and len(LRsimpspos) < 2:
            if pt2 in NextSimp.points:
                AreAdjacent = True
                LRsimps.append(NextSimp)
                LRsimpspos.append(simpposcounter)                
            simpposcounter +=1
            NextSimp = NextSimp.simplices[(locid+1)%3]
            locid = NextSimp.LocalID(pt1)
        if AreAdjacent:
            if LRsimpspos[1] == LRsimpspos[0]+1:
                return [AreAdjacent,[LRsimps[1],LRsimps[0]]]
            else:
                return [AreAdjacent,LRsimps]
        else:
            return [AreAdjacent,None]        

        
     
    def DoesCurveLeft(self,pttriple):
        pt1 = pttriple[0]
        pt2 = pttriple[1]
        pt3 = pttriple[2]
        pos1 = self.pointpos[pt1]
        pos2 = self.pointpos[pt2]
        pos3 = self.pointpos[pt3]
        crossP = (pos3[0] - pos2[0])*(pos1[1] - pos2[1]) - (pos3[1] - pos2[1])*(pos1[0] - pos2[0])
        if crossP >= 0:
            return True
        else:
            return False
    
    
    #given the two adjecent simplices, determine if the configuration is locally Delaunay.  Returns True or False
    def IsLocallyDelaunay(self,AdjSimps):
        #first get the local ids of the points not shared by these simplices
        S1 = AdjSimps[0]
        S2 = AdjSimps[1]
        S1exptlocid = 0
        for i in range(3):
            if not S1.points[i] in S2.points:
                S1exptlocid = i
                break
        S2exptlocid = (S2.LocalID(S1.points[(S1exptlocid + 1)%3])+1)%3
        #now get the angles for these two points
        S13pts = [S1.points[(S1exptlocid+2)%3],S1.points[S1exptlocid],S1.points[(S1exptlocid+1)%3]]
        S23pts = [S2.points[(S2exptlocid+2)%3],S2.points[S2exptlocid],S2.points[(S2exptlocid+1)%3]]
        posS1pts = []
        posS2pts = []
        for i in range(0,3):
            posS1pts.append([self.pointpos[S13pts[i]][0],self.pointpos[S13pts[i]][1]])
            posS2pts.append([self.pointpos[S23pts[i]][0],self.pointpos[S23pts[i]][1]])
        
        #print("S1 points (before): ",S1.points, S1.relptregion, posS1pts)
        #print("S2 points: (before)",S2.points, S2.relptregion, posS2pts)
        
        #before we compute the angles, we need to modify the points to account for points out of the FD
        #take the first point to be in the FD
        posS1pts[1][0] += S1.relptregion[(S1exptlocid+1)%3][0]*self.FDsizes[0]
        posS1pts[1][1] += S1.relptregion[(S1exptlocid+1)%3][1]*self.FDsizes[1]
        posS1pts[2][0] -= S1.relptregion[S1exptlocid][0]*self.FDsizes[0]
        posS1pts[2][1] -= S1.relptregion[S1exptlocid][1]*self.FDsizes[1]
        
        
        posS2pts[0][0] += S2.relptregion[S2exptlocid][0]*self.FDsizes[0]
        posS2pts[0][1] += S2.relptregion[S2exptlocid][1]*self.FDsizes[1]
        posS2pts[1][0] -= S2.relptregion[(S2exptlocid+2)%3][0]*self.FDsizes[0]
        posS2pts[1][1] -= S2.relptregion[(S2exptlocid+2)%3][1]*self.FDsizes[1]
            
        #print("pos s1", posS1pts)
        #print("pos s2", posS2pts)
        S1angle = self.TriAngle(posS1pts)
        S2angle = self.TriAngle(posS2pts)
        IsLocD = False
        if (S1angle + S2angle) <= math.pi:
            IsLocD = True
        return IsLocD
    
    
    
    #!!!Careful!!! this function modifies the given triangulation
    #this takes the given adjacent pair of simplices and flips the shared edge to the other pair of points.  It updates all of the relevant structures
    def EdgeFlip(self,AdjSimps,TimeIn = None):
        #first get the local ids of the points not shared by these simplices
        Simp = AdjSimps[0]
        Topsimp = AdjSimps[1]
        colind = 0
        for i in range(3):
            if not Simp.points[i] in Topsimp.points:
                colind = i
                break  
        cpt = Simp.points[colind]
        rptlid = (colind+1)%3
        lptlid = (colind+2)%3
        rpt = Simp.points[rptlid]
        lpt = Simp.points[lptlid]
        rptuid = Topsimp.LocalID(rpt)
        lptuid = Topsimp.LocalID(lpt)
        tpt = Topsimp.points[(rptuid+1)%3]
        rslist = [cpt,rpt,tpt]
        lslist = [cpt,tpt,lpt]
        rsimp = simplex2D(rslist)  #new right simplex
        lsimp = simplex2D(lslist)  #new left simplex
        #create the list of edge ids for the weight operator
        WeightIDs = [Simp.edgeids[colind], Topsimp.edgeids[lptuid], Topsimp.edgeids[rptuid], Simp.edgeids[rptlid], Simp.edgeids[lptlid]]
        #create the weight operater and append to the list
        self.WeightOperatorList.append(WeightOperator(WeightIDs,TimeIn))
        #now create the links these simplices have to other simplices
        SimpLink(rsimp,Topsimp.simplices[lptuid])
        SimpLink(lsimp,Topsimp.simplices[rptuid])
        SimpLink(rsimp,Simp.simplices[lptlid])
        SimpLink(lsimp,Simp.simplices[rptlid])
        SimpLink(rsimp,lsimp)
        
        #also need to reassign the weight ids
        rsimp.edgeids[0] = WeightIDs[1]  #for all of these, we know which points the local ids correspond to
        rsimp.edgeids[1] = WeightIDs[0]
        rsimp.edgeids[2] = WeightIDs[4]
        lsimp.edgeids[0] = WeightIDs[2]
        lsimp.edgeids[1] = WeightIDs[3]
        lsimp.edgeids[2] = WeightIDs[0]
        
        
        #need to create the relptregion data for the new simplices
        for k in range(2):
            rsimp.relptregion[0][k] = Topsimp.relptregion[lptuid][k]
            rsimp.relptregion[1][k] = Topsimp.relptregion[rptuid][k]+Simp.relptregion[rptlid][k]
            rsimp.relptregion[2][k] = Simp.relptregion[lptlid][k]

            lsimp.relptregion[0][k] = Topsimp.relptregion[rptuid][k]
            lsimp.relptregion[1][k] = Simp.relptregion[rptlid][k]
            lsimp.relptregion[2][k] = Topsimp.relptregion[lptuid][k]+Simp.relptregion[lptlid][k]
        
        #replace the two bad simplices in the simplex list with the two new ones
        Simpindex = Simp.SLindex
        self.simplist[Simpindex] = lsimp
        lsimp.SLindex = Simpindex
        
        Topsimpindex = Topsimp.SLindex
        self.simplist[Topsimpindex] = rsimp
        rsimp.SLindex = Topsimpindex
                
        #look through the simplex point list to see if either of the bad simplices were there and replace if so
        if self.pointlist[cpt] is Simp:
            self.pointlist[cpt] = Simp.simplices[lptlid]
        if (self.pointlist[rpt] is Simp) or (self.pointlist[rpt] is Topsimp):
            self.pointlist[rpt] = Topsimp.simplices[lptuid]
        if self.pointlist[tpt] is Topsimp:
            self.pointlist[tpt] = Topsimp.simplices[rptuid]
        if (self.pointlist[lpt] is Simp) or (self.pointlist[lpt] is Topsimp):
            self.pointlist[lpt] = Simp.simplices[rptlid]
        
               
            
        DNN = self.DebugNoneNeighbors()
        if DNN:
            print("old simp ", Simp, Simp.points, Simp.simplices)
            print("old top simp ",Topsimp, Topsimp.points, Topsimp.simplices)
            print("new lsimp ",lsimp, lsimp.points, lsimp.simplices)
            
            print("new rsimp ",rsimp, rsimp.points, rsimp.simplices)
            
            for k in range(3):
                lsimpn = lsimp.simplices[k]
                print("lsimp neighbors ", lsimpn.points, lsimpn.simplices)
            for k in range(3):
                rsimpn = rsimp.simplices[k]
                print("rsimp neighbors ", rsimpn.points, rsimpn.simplices)
        
        #Next, delete all the references to simplices in both of the bad simplices
        for i in range(0,3):
            Simp.simplices[i] = None
            Topsimp.simplices[i] = None 
        
        return [lsimp,rsimp]
    
    
    #!!!this changes the triangulation!!  
    #This function will take the current triangulation and, through a series of edge flips, change it into the Delaunay triangulation for this point configuration
    def MakeDelaunay(self):
        #first we need to create a list of all point pairs that represent a current edge
        EdgeList = []  #this edge list contains each edge twice ... there should be a better way of accumulating these, but it is not worth it to prune this, as the duplicates will be discarded as we go through the list anyway
        for i in range(len(self.simplist)):
            for j in range(3):
                if not self.simplist[i].simplices[j] is None:
                    EdgeList.append([self.simplist[i].points[(j+1)%3],self.simplist[i].points[(j+2)%3]])
        #now go through the edge list and start flipping edges
        while len(EdgeList) > 0:
            #now get the adjacent simplices
            
            ptpair = EdgeList.pop()
            #print("Considering ", ptpair, " in EdgeList of length ",len(EdgeList))
            #print("before APA", len(EdgeList))
            #self.DebugNoneNeighbors()
            APA = self.ArePointsAdjacent(ptpair[0],ptpair[1])
            #print("after APA",len(EdgeList))
            #self.DebugNoneNeighbors()
            if APA[0]:  #if these points are in the current triangulation
                #print("Points ",ptpair," are adjacent in the triangulation")
                #print(len(EdgeList))
                if not self.IsLocallyDelaunay(APA[1]):  #pass in the pair of adjacent simplices
                    #print("Not Loc Delaunay",len(EdgeList))
                    #self.DebugNoneNeighbors()
                    #now flip the edge
                    #print("Flipping Edge ", ptpair)
                    LRsimps = self.EdgeFlip(APA[1],self.atstep)
                    #print("after EdgeFlip ",len(EdgeList))
                    #self.DebugNoneNeighbors()
                    #and add the surrounding edges to the EdgeList
                    Lloc = 0
                    for k in range(3):
                        if not LRsimps[0].points[k] in LRsimps[1].points:
                            Lloc = k
                            break
                    if not LRsimps[0].simplices[(Lloc+1)%3] is None:
                        EdgeList.append([LRsimps[0].points[(Lloc+2)%3],LRsimps[0].points[Lloc]])
                    if not LRsimps[0].simplices[(Lloc+2)%3] is None:
                        EdgeList.append([LRsimps[0].points[(Lloc+1)%3],LRsimps[0].points[Lloc]])
                    Rloc = (LRsimps[1].LocalID(LRsimps[0].points[(Lloc+1)%3])+1)%3
                    if not LRsimps[1].simplices[(Rloc+1)%3] is None:
                        EdgeList.append([LRsimps[1].points[(Rloc+2)%3],LRsimps[1].points[Rloc]])
                    if not LRsimps[1].simplices[(Rloc+2)%3] is None:
                        EdgeList.append([LRsimps[1].points[(Rloc+1)%3],LRsimps[1].points[Rloc]])
                    #print("after Edgelist accumulate",len(EdgeList))
                    #self.DebugNoneNeighbors()
         #Now we should have a Delaunay Triangulation
                        
            
            
    #This returns a list of the pairs of points (edges) in the current triangulation
    def GetEdgeList(self):
        EdgeList = []
        numpoints = self.ptnum
        EdgeAccList = [[] for i in range(numpoints)]
        for i in range(len(self.simplist)):
            for j in range(3):
                p1 = self.simplist[i].points[(j+1)%3]
                p2 = self.simplist[i].points[(j+2)%3]
                if p1 < numpoints and p2 < numpoints:
                    if p1 < p2:
                        EdgeAccList[p1].append(p2)
                    else:
                        EdgeAccList[p2].append(p1)
        for i in range(len(EdgeAccList)):
            EdgeAccList[i].sort()
            for j in range(len(EdgeAccList[i])-1,0,-1):
                if EdgeAccList[i][j] == EdgeAccList[i][j-1]:
                    del EdgeAccList[i][j]
            for j in range(len(EdgeAccList[i])):
                EdgeList.append([i,EdgeAccList[i][j]])
        return EdgeList
        
        
        
    #this returns a list of pairs of points that are k edges apart or less
    #k = 1 gives the the same list as GetEdgeList (though different implimentation)
    #def GetKPtPairList(self,k=2):
    # go through the pointlist and find all k neighbors working out.  Exclude points from the accumulated set that are
    # smaller than the current point

    
    #*******Needs to be updated
    #this creates a copy of the current triangulation object with or without copying the weight operators
    def TriCopy(self,NoWeightOps = True):
        #first create an empty triangulation object (to be returned at the end)
        TriC = triangulation2D(None,empty = True)
        TriC.atstep = self.atstep
        TriC.extranum = self.extranum
        TriC.ptnum = self.ptnum
        TriC.extrapoints = copy.deepcopy(self.extrapoints)
        TriC.pointpos = copy.deepcopy(self.pointpos)
        TriC.pointposfuture = copy.deepcopy(self.pointposfuture)
        for i in range(len(self.simplist)):
            TriC.simplist.append(simplex2D(self.simplist[i].points))
            TriC.simplist[-1].edgeids = copy.copy(self.simplist[i].edgeids)
            TriC.simplist[-1].SLindex = i
        
        #now create the links
        for i in range(len(self.simplist)):
            for j in range(3):
                if not self.simplist[i].simplices[j] is None:
                    TriC.simplist[i].simplices[j] = TriC.simplist[self.simplist[i].simplices[j].SLindex]

        #now fill the pointlist
        for i in range(len(self.pointlist)):
            TriC.pointlist.append(TriC.simplist[self.pointlist[i].SLindex])
                    
        TriC.totalnumedges = self.totalnumedges

        if not NoWeightOps:
            for i in range(len(self.WeightOperatorList)):
                TriC.WeightOperatorList.append(WeightOperator(copy.copy(self.WeightOperatorList[i].es)))
                
        return TriC
            
    
#End of triangulation2D class **************************************************************************************************************************************************************************************************************************************************************
#Set of general use stand-alone functions **************************************************************************************************************************************************************************************************************************************************************
    
#Stand-alone function to input one of my trajectory files and output a list of points in the proper format for this
#Nan wrote this, modfied just a bit
def OpenTrajectoryFile(fileName):
    #a list of time with evolving x,y coordinates ... time is evenly spaced, and we don't need the actual value right now (though for normalizing the topological entropy the total time elapsed will be useful)
    wholeList = []
    times = []
    #open and record
    with open(fileName,"r") as f:
        #for each time
        for line in f:
            #coordinates
            listXY = []
            a = line.split(" ")
            #delete the first element which is the time
            words = a[1:]
            times.append(float(a[0]))
            m = round((len(words))/2)
            for x in range(0, m):
                listXY.append([float(words[2*x]),float(words[2*x+1])])
            wholeList.append(listXY)
    return [times,wholeList]  #return the times too

    
#this takes two simplices that share an edge and links them (puts eachother in the appropriate neighbor simplex list)
#this assumes that the two simplices do share an edge.  If they don't this can create erronious links
def SimpLink(S1,S2):
    #need to deal with the case of either S1 or S2 being None
    if not ((S1 is None) or (S2 is None)):  #Note that we don't need to do anything for linking a new simplex to None ... it starts out that way
        #first find out which points they share
        locid = 0
        for i in range(0,3):
            if not S1.points[i] in S2.points:
                S1.simplices[i] = S2
                locid = i
                break
        smset = [S1.points[(locid+1)%3],S1.points[(locid+2)%3]]
        for i in range(0,3):
            if not S2.points[i] in smset:
                S2.simplices[i] = S1
                break    

                
                
#This takes two simplices that share an edge and are already linked, and sets their shared edge ID.
def SimpEdgeIDSet(S1,S2,edgeIDnew):
    #need to deal with the case of either S1 or S2 being None
    if not ((S1 is None) or (S2 is None)):  #Just a check for bad simplices
        #first find out which points they share
        locid = 0
        for i in range(0,3):
            if not S1.points[i] in S2.points:
                S1.edgeids[i] = edgeIDnew
                locid = i
                break
        spt1 = S1.points[(locid+1)%3]
        Lid = (S2.LocalID(spt1)+1)%3
        S2.edgeids[Lid] = edgeIDnew    
                
    
#Stand-alone function that takes a single list of simplices and returns a reduced version that excludes any duplicates
def SimpListCompact(SimpListIn):
    newList = []
    goodList = [True for i in range(0,len(SimpListIn))]
    for i in range(0,len(SimpListIn)):
        if goodList[i]:
            newList.append(SimpList[i])
        for j in range(i+1,len(SimpListIn)):
            if goodList[j]:
                if SimplistIn[i] is SimplistIn[j]:
                    goodList[j] = False
    return newList


#Stand-alone function that takes two lists of simplices, and merges them, avoiding duplicates
#can also accomodate lists of anything that are being compared by the is operator and have overlap
def ListMerge(SList1,SList2):
    #assume that each list is already compact (no duplicates)
    newList = [SList1[i] for i in range(0,len(SList1))]
    refList = [SList1[i] for i in range(0,len(SList1))]
    for i in range(0,len(SList2)):
        isgood = True
        for j in range(0,len(refList)):
            if refList[j] is SList2[i]:
                isgood = False
                del refList[j]
                break
        if isgood:
            newList.append(SList2[i])
    return newList


#Stand-alone function that returns True if the two lists have any common elements (used for simplices but can be use for any lists with common element types that can be compared with the is operator)
def ListIntersect(SList1, SList2):
    #assume that each list is already compact (no duplicates)
    Intersection = False
    for i in range(0,len(SList2)):
        for j in range(0,len(SList1)):
            if SList1[j] is SList2[i]:
                Intersection = True
                break
        if Intersection:
            break
    return Intersection
     
    
#Stand-alone function which takes a list of simplices and returns a list of point IDs with no duplicates
def SimpListToPtList(SListIn):
    ptlist = SListIn[0].points
    for i in range(1,len(SListIn)):
        ptlist = ListMerge(ptlist,SListIn[i].points)
    return ptlist

#Stand-alone function which does a binary search on a given sorted list (each element is a list of length 2, were the second item is the ordering parameter).  The list is assumed to be in decending order.  The item that is searched for is also a double [event,time to zero area].  The search is over the time variable, but the event variable is used for direct comparison.  If a match is found, then it is deleted from the list.
def BinarySearchDel(ListIn, ItemIn):
    Lindex = 0
    Rindex = len(ListIn) - 1
    success = False
    matchindex = 0
    while Rindex >= Lindex and not success:
        Mindex = (Rindex+Lindex)//2
        if ListIn[Mindex][0] == ItemIn[0]:
            #have a match
            success = True
            matchindex = Mindex
        else:
            if ItemIn[1] < ListIn[Mindex][1]:
                Lindex = Mindex + 1
            else:
                Rindex = Mindex - 1
    if success:
        del ListIn[matchindex]
    else:
        print("did not delete item from EventList, event was not found")
        print("Item In = ", ItemIn)
        print("Item In Info: ", ItemIn[0].points, ItemIn[0].relptregion)
        print("List In = ", ListIn)
            

#Stand-alone function which does a binary search on a given sorted list (each element is a double, were the second item is the ordering parameter).  The list is assumed to be in decending order.  The item that is searched for is also a double [event,time to zero area].  The binary search finds the adjacent pair of elements inbetween which the input item's time fits.  If such a pair is found, then the ItemIn is inserted into this position.
def BinarySearchIns(ListIn, ItemIn):
    Lindex = 0
    Rindex = len(ListIn) - 1
    if len(ListIn) == 0:
        ListIn.append(ItemIn)
    elif ItemIn[1] < ListIn[Rindex][1]:
        ListIn.append(ItemIn)
    elif ItemIn[1] > ListIn[Lindex][1]:
        ListIn.insert(0,ItemIn)
    else:
        while Rindex - Lindex > 1:
            Mindex = (Rindex+Lindex)//2
            if ItemIn[1] < ListIn[Mindex][1]:
                Lindex = Mindex
            else:
                Rindex = Mindex
        if Rindex - Lindex == 1:
            ListIn.insert(Rindex,ItemIn)
        else:
            #right an left indices are concurrent.  This can happen when ItemIn has an identical time to one of the
            #items in ListIn.  These are either the same object (in which case we don't insert), or we have a future combined event (with a simplex collapse and a triangle collapse being concurrent ... in which case we add in the new object)
            if not type(ItemIn[0]) == type(ListIn[Rindex][0]):
                ListIn.insert(Rindex,ItemIn)

#This is a stand-alone function that takes two Trajectory Time-slices, and outputs the trajectory time-slice that is a fraction of the way (halfway is the default) between the two in time (as a linear interpolation)
#It is assumed that the two time-slices are of the same size (and order), and that frac is in [0,1]
#it is also assumed that TS1 comes before TS2 in time
def FracTraj(TS1, TS2, frac = 0.5):
    OutTraj = []
    for i in range(0,len(TS1)):
        OutTraj.append([(TS2[i][0]-TS1[i][0])*frac + TS1[i][0], (TS2[i][1]-TS1[i][1])*frac + TS1[i][1]])
    return OutTraj
      
#this is a stand-alone function that takes a trajectory set, and outputs a trajectory set as a copy of the first, but with additional linearly interpolated time-slices between each adjacent time slice in the original (equally spaced in time)
def TrajInFill(Traj):
    TrajOut = []
    for i in range(0,len(Traj)-1):
        TrajOut.append(Traj[i])
        TrajOut.append(FracTraj(Traj[i],Traj[i+1]))
    TrajOut.append(Traj[-1])
    return TrajOut


#this is a stand-alone function that takes a test point and two ordered points, and outputs True if the test point is to the left of the line going through the two ordered points (with the first of these below the second).
#does this by testing for a positive cross-product: (p2-p1)x(testpt-p1) > 0
def IsPtLeft(testpt,p1,p2):
    if (p2[0]-p1[0])*(testpt[1]-p1[1]) - (p2[1]-p1[1])*(testpt[0]-p1[0]) >= 0:
        return True
    else:
        return False
    
#this is a stand-alone function that takes a test point and three ordered points (the three points of a triangle in ccw order), and returns True if the test point is interior to the triangle and False if it is outside.  This uses the IsPtLeft Function.
def TriContains(testpt,pts):
    if IsPtLeft(testpt,pts[0],pts[1]) and IsPtLeft(testpt,pts[1],pts[2]) and IsPtLeft(testpt,pts[2],pts[0]):
        return True
    else:
        return False

    
#this stand-alone function takes in two lines (each defined by two points) and outputs True if they intersect (between each of their point pairs)
#If the flag for time info is True, then we also output the time t1.  So Line 1 is one whose parameterized intersection time is returned.  The time is the fraction of the line from Line1[0] to Line1[1]
def IsIntersection(Line1,Line2,timeinfo = False):
    IsInt = False
    det = (Line2[1][1]-Line2[0][1])*(Line1[1][0]-Line1[0][0])-(Line1[1][1]-Line1[0][1])*(Line2[1][0]-Line2[0][0])
    t1 = 0
    t2 = 0
    if not det == 0:
        bx = (Line1[0][1]*Line1[1][0] - Line1[1][1]*Line1[0][0])
        by = (Line2[0][1]*Line2[1][0] - Line2[1][1]*Line2[0][0])
        xout = (bx*(Line2[1][0]-Line2[0][0]) - by*(Line1[1][0]-Line1[0][0]))/det
        yout = (bx*(Line2[1][1]-Line2[0][1]) - by*(Line1[1][1]-Line1[0][1]))/det
        
        if abs(Line1[1][0]-Line1[0][0]) > abs(Line1[1][1]-Line1[0][1]):
            t1 = (xout - Line1[0][0])/(Line1[1][0]-Line1[0][0])
        else:
            t1 = (yout - Line1[0][1])/(Line1[1][1]-Line1[0][1])
        if not (t1 > 1 or t1 < 0):
            if abs(Line2[1][0]-Line2[0][0]) > abs(Line2[1][1]-Line2[0][1]):
                t2 = (xout - Line2[0][0])/(Line2[1][0]-Line2[0][0])
            else:
                t2 = (yout - Line2[0][1])/(Line2[1][1]-Line2[0][1])

            if not (t2 > 1 or t2 < 0):
                IsInt = True
    if timeinfo:
        return [IsInt,t1]
    else:
        return IsInt
       
