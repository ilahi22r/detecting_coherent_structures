import EtecDualPeriodicBCv2 as Et
import numpy as np
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import math
import scipy.io as sio
import copy
from scipy.spatial import ConvexHull


# For the combineloops function below, we will need to reference the relative locations of points in the doubly periodic boundary.  This function outputs the number of x shifts (+- integer) and y shifts (+- integer) needed to get from pt1 to pt2.  [0,0] is the default of both point being in the fundamental domain
def GetRelShift(pt1,pt2,tri, first = True):
    #first, we determine if the two points are connected via an edge in the triangulation.  If so, then we can get the needed info directly from the triangulation object.
    SimpSet = tri.pointlist[pt1].SimpNeighbors(pt1)
    for simp in SimpSet:
        if pt2 in simp.points:
            local_pt1_index = simp.LocalID(pt1)
            local_pt2_index = simp.LocalID(pt2)
            if (local_pt1_index+1)%3 == local_pt2_index:
                vantage_point = (local_pt1_index+2)%3
                return(np.array(simp.relptregion[vantage_point]))
            else:
                vantage_point = (local_pt1_index+1)%3
                return(-1*np.array(simp.relptregion[vantage_point]))
            break  #not really needed
    
    #if not connected by an edge, we can look at which shifted domain (of the 9 immediate copies) has the closest version of pt2 to the pt1 in the FD
    pt1_FD_pos = np.array(tri.pointpos[pt1])
    pt2_FD_pos = np.array(tri.pointpos[pt2])
    Lx = tri.FDsizes[0]
    Ly = tri.FDsizes[1]
    Min_Dist = (Lx**2+Ly**2)**(0.5)
    rpp = None
    disp = [-1,0,1]
    for i in disp:
        for j in disp:
            pt2_trial_pos = pt2_FD_pos + np.array([i*Lx,j*Ly])
            vec_diff = pt2_trial_pos - pt1_FD_pos
            dist = (vec_diff[0]**2+vec_diff[1]**2)**(0.5)
            if dist < Min_Dist:
                Min_Dist = dist
                rpp = np.array([i,j])
    
    rpp_prev = None
    Min_Dist = (Lx**2+Ly**2)**(0.5)
    for i in disp:
        for j in disp:
            pt2_trial_pos = pt2_FD_pos + np.array([i*Lx,j*Ly])
            vec_diff = pt2_trial_pos - pt1_FD_pos
            dist = (vec_diff[0]**2+vec_diff[1]**2)**(0.5)
            if dist < Min_Dist and not (i == rpp[0] and j == rpp[1]):
                Min_Dist = dist
                rpp_prev = np.array([i,j])
                
    if first:
        return rpp
    else:
        return rpp_prev
    


#now we get the point positions of the points in a loop list relative to a given reference point as starting in the fundamental domain
def GetPtPos(loop,tri,ref_ind = 0, regular_loop = True):
    pt_pos_fd = [tri.pointpos[lp] for lp in loop]
    rel_to_prevpt = [GetRelShift(loop[(i-1)%len(loop)],loop[i],tri) for i in range(len(loop))]
    #sometimes rel_to_prevpt will have inproperly chosen elements (if the actual length of a segment in the loop is larger than the line connecting the points across a domain)
    if regular_loop:  #a regular loop has zero winding numbers for the two torus directions
        #now sum up the elements of rel_to_prevpt
        x_sum = 0
        y_sum = 0
        for rpp in rel_to_prevpt:
            x_sum += rpp[0]
            y_sum += rpp[1]
        if not (x_sum == 0 and y_sum == 0):
            #print("Sum of relative positions is not [0,0], it is = ", [x_sum,y_sum])
            fix_index = None
            for k in range(len(rel_to_prevpt)):
                if x_sum == rel_to_prevpt[k][0] and y_sum == rel_to_prevpt[k][1]:
                    fix_index = k
            
            #fix_index = rel_to_prevpt.index(np.array([x_sum,y_sum]))
            if not fix_index == None:
                rel_to_prevpt[fix_index] = GetRelShift(loop[(fix_index-1)%len(loop)],loop[fix_index],tri,first = False)
                x_sum2 = 0
                y_sum2 = 0
                for rpp in rel_to_prevpt:
                    x_sum2 += rpp[0]
                    y_sum2 += rpp[1]
                
                #print("Now the sum of the relative positions is = ", [x_sum2,y_sum2])
            else:
                print("a more complicated fix is needed")
                print("rel_to_prevpt = ",rel_to_prevpt)
                print("total = ", [x_sum,y_sum])
    
    #print("rel_to_prevpt = ", rel_to_prevpt)
    rel_accum = [None for i in range(len(loop))]
    rel_accum[ref_ind] = np.array([0,0])
    Lx = tri.FDsizes[0]
    Ly = tri.FDsizes[1]
    for i in range(1,len(loop)):
        rel_accum[(ref_ind+i)%len(loop)] = rel_accum[(ref_ind+i-1)%len(loop)] + rel_to_prevpt[(ref_ind+i)%len(loop)]
    #print("rel_accum = ", rel_accum)
    pt_pos = [pt_pos_fd[i]+np.array([Lx*rel_accum[i][0], Ly*rel_accum[i][1]])  for i in range(len(loop))]
    return pt_pos




#there are a few functions to write, and a few conventions to clear up before we write down the main function.

#first of all, we are assuming that the input loops are of the form of a list of point ids (the ids are how we reference these points in a triangulation2d object).  These points are listed in CCW order (how we traverse the loop).  Some of the points in this boundary list are not interior to the loop.

#first function takes in a loop and the triangulation, and outputs an equal length list with elements as True (for a convex point) and False (for a concave point).
def ConvexList(loop, tri):
    loop_len = len(loop)
    CVL = [True for i in range(loop_len)] #the list to be returned
    if loop_len > 2:
        PtPos = GetPtPos(loop,tri)
        for i in range(loop_len):
            current_pt_pos = PtPos[i]
            prev_pt_pos = PtPos[(i-1)%loop_len]
            next_pt_pos = PtPos[(i+1)%loop_len]
            vec_1 = [current_pt_pos[k] - prev_pt_pos[k] for k in range(2)]
            vec_2 = [next_pt_pos[k] - current_pt_pos[k] for k in range(2)]
            #vec_1 X vec_2 is > 0 for convex current point
            v1_X_v2 = vec_1[0]*vec_2[1] - vec_1[1]*vec_2[0]
            if v1_X_v2 < 0:
                CVL[i] = False
    return CVL

#Next, we need a function that will find the points interior to the loop.  We assume that each loop is not "folded" (i.e. each boundary point occurs only once in the boundary list - a point appearing twice, once as convex and once as concave, would indicate that the loop has been pinced/folded at this point)
#interior points include all the boundary points that are convex, and any other point that is interior to the bounding edges 
def GetInteriorPoints(loop, tri):
    interior_pts = []
    convex_pts = ConvexList(loop, tri)
    for i in range(len(loop)):
        if convex_pts[i]:
            interior_pts.append(loop[i])
    #now get the convex decomposition of the polygon formed from the boundary points
    CV_decomp = GetConvexDecomposition(loop, tri)
    for i in range(tri.ptnum):
        if not i in loop:
            for CVset in CV_decomp:
                if InteriorToConvexSet(i,CVset,tri):
                    interior_pts.append(i)
    return interior_pts
    
    
#determine whether a point is interior to a given convex boundary set (this is given in CCW order)
def InteriorToConvexSet(trial_pt, convex_set, tri, LocReturn = False):
    IsInterior = True
    trial_pt_pos = np.array(tri.pointpos[trial_pt])
    #we will check all 9 versions of this point (shifted from FD) for being interior to the loop (one single version of the loop)
    Shifted_trial_pts_pos = []
    Lx = tri.FDsizes[0]
    Ly = tri.FDsizes[1]
    disp = [-1,0,1]
    for i in disp:
        for j in disp:
            Shifted_trial_pts_pos.append(trial_pt_pos + np.array([i*Lx,j*Ly]))
            
    #now get the positions of the points in convext set
    cs_pos = GetPtPos(convex_set,tri)
    
    for tp in Shifted_trial_pts_pos:
        IsInterior = True
        for i in range(len(cs_pos)):
            vec_1 = [cs_pos[(i+1)%(len(cs_pos))][k] - cs_pos[i][k] for k in range(2)]
            vec_2 = [tp[k] - cs_pos[i][k] for k in range(2)]
            v1_X_v2 = vec_1[0]*vec_2[1] - vec_1[1]*vec_2[0]
            if v1_X_v2 < 0:
                IsInterior = False
                break
        if IsInterior:
            if LocReturn:
                return [IsInterior, tp]
            else:
                return IsInterior
    if LocReturn:
        return [IsInterior, None]
    else:
        return IsInterior

#determine whether a point is interior to a given convex boundary set (this is given in CCW order) - in this version, the input are point positions
#We can assume that the convex_set_pos points have already been modified to be in the appropriate copy of the FD.  Just need to try the different offsets of the trial point
def InteriorToConvexSet2(trial_pt_pos, convex_set_pos,tri, LocReturn = False):
    IsInterior = True
    Shifted_trial_pts_pos = []
    Lx = tri.FDsizes[0]
    Ly = tri.FDsizes[1]
    disp = [-1,0,1]
    for i in disp:
        for j in disp:
            Shifted_trial_pts_pos.append(np.array(trial_pt_pos) + np.array([i*Lx,j*Ly]))
    
    for tp in Shifted_trial_pts_pos:
        IsInterior = True
        for i in range(len(convex_set_pos)):
            vec_1 = [convex_set_pos[(i+1)%(len(convex_set_pos))][k] - convex_set_pos[i][k] for k in range(2)]
            vec_2 = [tp[k] - convex_set_pos[i][k] for k in range(2)]
            v1_X_v2 = vec_1[0]*vec_2[1] - vec_1[1]*vec_2[0]
            if v1_X_v2 < 0:
                IsInterior = False
                break
        if IsInterior:
            if LocReturn:
                return [IsInterior, tp]
            else:
                return IsInterior

    if LocReturn:
        return [IsInterior, None]
    else:
        return IsInterior



#In order to use the InteriorToConvexSet, we must take a general bounding set and break it down into a series of convex set.  We do this by breaking off triangles from the polygon just before concave points, and repeating
def GetConvexDecomposition(loop, tri):
    loop_temp = [x for x in loop]
    decomposed = False
    decomposition = []
    while not decomposed:
        convex_temp = ConvexList(loop_temp, tri)
        starting_point = None
        for i in range(len(loop_temp)):
            if convex_temp[i] and not convex_temp[(i+1)%len(convex_temp)]:
                starting_point = i
                break
        if starting_point == None:
            decomposed = True
            decomposition.append(loop_temp)
        else:
            decomposition.append([loop_temp[(starting_point+ len(loop_temp)-1+k)%len(loop_temp)] for k in range(3)])
            del loop_temp[starting_point]
    return decomposition
    
        
        
#before we merge two loops, we need at least one more function
#this takes two line segments and returns T/F for whether they cross between the endpoints and if so, also returns the crossing point
#the input are two lines - each: [[x_1,y_1],[x_2,y_2]]
def GetIntersection(Line1,Line2):
    L1A, L1B = Line1
    L2A, L2B = Line2
    L1Ax, L1Ay = L1A
    L1Bx, L1By = L1B
    L2Ax, L2Ay = L2A
    L2Bx, L2By = L2B
    line1len = ((L1Ax-L1Bx)**2 + (L1Ay-L1By)**2)**(0.5)
    line2len = ((L1Ax-L1Bx)**2 + (L1Ay-L1By)**2)**(0.5)
    epsilon = 1e-6
    epsilon1 = epsilon*line1len
    epsilon2 = epsilon*line2len
    M11 = L1By - L1Ay
    M12 = L1Ax - L1Bx
    M21 = L2By - L2Ay
    M22 = L2Ax - L2Bx
    C1 = L1Ax*L1By - L1Ay*L1Bx
    C2 = L2Ax*L2By - L2Ay*L2Bx
    detM = M11*M22 - M12*M21
    if detM == 0:
        #the lines are parallel
        return [False, None]
    else:
        x = (M22*C1 - M12*C2)/detM
        y = (M11*C2 - M21*C1)/detM
        t1 = None
        if not M12 == 0:
            t1 = (x-L1Ax)/(-M12)
        else:
            t1 = (y - L1Ay)/M11
        t2 = None
        if not M22 == 0:
            t2 = (x-L2Ax)/(-M22)
        else:
            t2 = (y - L2Ay)/M21
        
        if t1 > epsilon1 and t1 < (1.0-epsilon1) and t2 > epsilon2 and t2 < (1.0-epsilon2):
            #the line segments intersect
            return [True, [x,y]]
        else:
            #the lines intersect, but not the line segments
            return [False, [x,y]]
        

        
#now for the main function.  This takes two loops and returns a new loops which is the combination of these input loops.  
def CombineLoops(loop1, loop2, tri, Just_Largest = True, Verbose = False):
    #first, let's get the interior of both loops to verify that they have some overlap.
    if Verbose:
        print("Finding Loop Interiors")
    L1_interior = GetInteriorPoints(loop1,tri)
    L2_interior = GetInteriorPoints(loop2,tri)
    if Verbose:
        print("L1_interior = ",L1_interior)
        print("L2_interior = ",L2_interior)
    L1L2Intersection = list(set(L1_interior) & set(L2_interior))
    L1L2Union = list(set(L1_interior) | set(L2_interior))
    if Verbose:
        print("L1L2Intersection = ", L1L2Intersection)
        print("L1L2Union = ", L1L2Union)
        
        
    if len(L1L2Intersection) == 0:
        print("The intersection of the interior point lists of the two loops is empty (i.e. the loops don't intersect)!")
        return None
    else:
        #to prepare all the information we might need, we compute the convexity information for each loop.  We also wrap this info (and all other info about the two loops) in lists containing both, so we can toggle back and forth between the two.
        if Verbose:
            print("Getting Convexity Lists")
        L1_convexity = ConvexList(loop1, tri)
        L2_convexity = ConvexList(loop2, tri)
        loop_convexity = [L1_convexity,L2_convexity]
        loop_interior = [L1_interior,L2_interior]
        loop = [loop1,loop2]
        loop_len = [len(loop[0]), len(loop[1])]
        #Now let's get the positions of each point (only the points in the loops will have proper locations)
        pointpos = [None for i in range(tri.ptnum)]
        #find the points that are closest in the FD
        ptpair_closest = None
        Lx = tri.FDsizes[0]
        Ly = tri.FDsizes[1]
        Min_Dist = (Lx**2+Ly**2)**(0.5)
        for L1pt in loop1:
            for L2pt in loop2:
                pt1pos = np.array(tri.pointpos[L1pt])
                pt2pos = np.array(tri.pointpos[L2pt])
                dist = np.linalg.norm(pt1pos-pt2pos)
                if dist < Min_Dist:
                    Min_Dist = dist
                    ptpair_closest = [L1pt,L2pt]
        #print("ptpair_closest = ", ptpair_closest, "Min_Dist = ", Min_Dist)
        L1ind = loop1.index(ptpair_closest[0])
        L2ind = loop2.index(ptpair_closest[1])
        #print("L1ind = ", L1ind, ", L2ind = ", L2ind)
        #print("Getting the loop positions")
        loop1pos = GetPtPos(loop1,tri,L1ind)
        loop2pos = GetPtPos(loop2,tri,L2ind)
        for i in range(len(loop1)):
            pointpos[loop1[i]] = loop1pos[i]
        for i in range(len(loop2)):
            pointpos[loop2[i]] = loop2pos[i]
            
            
        #now we can use pointpos in place of tri.pointpos
        #print("loop1pos = ", loop1pos)
        #print("loop2pos = ", loop2pos)
        #print("pointpos = ", pointpos)
        
        #now let's make a pair of lists which encode which edges cross each other
        L1_crossings = [[] for i in range(loop_len[0])]
        L2_crossings = [[] for j in range(loop_len[1])]
        #now determine if there is a line crossing
        for i in range(loop_len[0]):
            line1 = [pointpos[loop[0][(i+k)%loop_len[0]]] for k in range(2)]
            for j in range(loop_len[1]):
                line2 = [pointpos[loop[1][(j+k)%loop_len[1]]] for k in range(2)]
                IJintersection = GetIntersection(line1,line2)
                if IJintersection[0]:
                    ipos = pointpos[loop[0][i]]
                    idist = ((ipos[0]-IJintersection[1][0])**2+ (ipos[1]-IJintersection[1][1])**2)**(0.5)
                    L1_crossings[i].append([j,IJintersection[1],idist])
                    jpos = pointpos[loop[1][j]]
                    jdist = ((jpos[0]-IJintersection[1][0])**2+ (jpos[1]-IJintersection[1][1])**2)**(0.5)
                    L2_crossings[j].append([i,IJintersection[1],jdist])
        #now go through and sort each sub list that contains more than one element by the dist data point (increasing)
        for i in range(loop_len[0]):
            if len(L1_crossings[i]) > 1:
                L1_crossings[i].sort(key=lambda x: x[2])
        for j in range(loop_len[1]):
            if len(L2_crossings[j]) > 1:
                L2_crossings[j].sort(key=lambda x: x[2])
        #and bundle this in the usual way        
        loop_crossings = [L1_crossings, L2_crossings]
        
        rough_new_loops = []  #this will have the loops that stitch together from the two initial loops and the crossings.  These loops will then need to be "relaxed"
        
        points_used = [[False for i in range(loop_len[k])] for k in range(2)] #this is how we keep track of where to start our loop stitching procedure for later loops
        
        exhausted_points = False
        switch_starting_loop = False
        #this is the big loop; each time through will generate a new rough_new_loop
        
        while not exhausted_points:
            if Verbose:
                print("rough_new_loops", rough_new_loops)
            current_loop = 0  #we start the current loop as loop1
            if switch_starting_loop:
                current_loop = 1
            current_index_default = next((i for i in range(loop_len[current_loop]) if not points_used[current_loop][i]),None)
            if Verbose:
                print("current_index_default = ", current_index_default)
            current_index = next((i for i in range(loop_len[current_loop]) if not points_used[current_loop][i] and not loop[current_loop][i] in loop[(current_loop+1)%2]), current_index_default) 
            if Verbose:
                print("current_index = ", current_index)
            #we start with the first point in the current loop that has not been used as part of a previous loop and is not shared with the other loop
            #print("points_used = ",points_used)
            #print("current_index = ", current_index)
            #print("current_loop = ", current_loop)
            starting_point = loop[current_loop][current_index]
            rough_loop = []  #this is where we will put the points
            #rough_loop = [[True, starting_point,False]]
            #each element will consist of a triple: [regular point - T/F, pt index or [x,y], to be removed - T/F]    non-regular point is one that we need to add in when lines cross (they will be removed, and instead of an index in the second slot, they have the x-y location of the point)
            current_point = None
            iterator = 0
            LRside = None  #this is used when we have parallel segments of the two loops.  It records whether the current loop is on the "Left" or "Right" side of the parallel segment (in the direction of the current loop). 
            
            #now we traverse around the two loops, stiching sections of them together to form a rough loop
            while not current_point == starting_point:
                if Verbose:
                    print("rough_loop = ", rough_loop)
                    print("current_loop = ", current_loop, ", current_point = ", current_point, ", starting_point = ", starting_point, ", step number = ", iterator, ",  LRside = ",  LRside)
                #if iterator < 200:
                #    print("current_loop = ", current_loop, ", current_point = ", current_point, ", starting_point = ", starting_point, ", step number = ", iterator, ",  LRside = ",  LRside)
                #    iterator += 1
                #else:
                #    break
                
                points_used[current_loop][current_index] = True
                #first, we need to determine if there is a line crossing, and then add this point to the list
                if len(loop_crossings[current_loop][current_index]) > 0:
                    
                    Line_intersection_continue = True
                    which_intersection = 0
                    while Line_intersection_continue:
                        int_info = loop_crossings[current_loop][current_index][which_intersection]
                        prev_index = current_index
                        rough_loop.append([False,int_info[1],True])
                        if Verbose:
                            print("line crossing: ", int_info[1])
                        current_loop = (current_loop+1)%2
                        current_index = int_info[0]
                        
                        if len(loop_crossings[current_loop][current_index]) == 1:
                            Line_intersection_continue = False
                            #print("only one line crossing")
                        else:
                            #get the local id of the crossing
                            other_indices = [x[0] for x in loop_crossings[current_loop][current_index]]
                            local_id = other_indices.index(prev_index)
                            if local_id == len(loop_crossings[current_loop][current_index])-1:
                                #this is at the end, and we can get out of the while loop
                                Line_intersection_continue = False
                            else:
                                
                                which_intersection = local_id+1
                        
                    
                    #now move the current point up and add to rough loop.  Also set the current_point appropriately
                    #current_index = (current_index+1)%loop_len[current_loop]
                    #current_point = loop[current_loop][current_index]
                    #print("adding in current point = ", current_point, ", in loop ", current_loop)
                    
                    #rough_loop.append([True, current_point, False])
                
                if not loop[current_loop][(current_index+1)% loop_len[current_loop]] in loop[(current_loop+1)%2]:
                    #there are no intersections of any type to deal with here, we just move up to the next point
                    if Verbose:
                        print("adding in current point = ", current_point, ", in loop ", current_loop)
                    current_index = (current_index+1)%loop_len[current_loop]
                    current_point = loop[current_loop][current_index]
                    rough_loop.append([True, current_point, False])
                else:
                    
                    current_index = (current_index+1)%loop_len[current_loop]
                    #the current point is shared between the two loops
                    current_point = loop[current_loop][current_index]
                    if Verbose:
                        print("Next Point is Shared ", current_point)
                    #we need to go through all the options individually
                    #first, let's get the surrounding points
                    CL_pp = loop[current_loop][(current_index-1)% loop_len[current_loop]]  #current loop, previous point
                    CL_np = loop[current_loop][(current_index+1)% loop_len[current_loop]]  #current loop, next point
                    other_index = loop[(current_loop+1)%2].index(current_point)
                    OL_pp = loop[(current_loop+1)%2][(other_index-1)% loop_len[(current_loop+1)%2]]  #other loop, previous point
                    OL_np = loop[(current_loop+1)%2][(other_index+1)% loop_len[(current_loop+1)%2]]  #other loop, next point
                    surrounding_points = [CL_pp,CL_np,OL_pp,OL_np]
                    red_surrounding_points = list(set(surrounding_points))
                    if Verbose:
                        print("Surrounding Points = ", surrounding_points)
                    
                    if len(red_surrounding_points) == 4:
                        if Verbose:
                            print("Regular case of 4 unique surrounding points")
                        #now go through the possibilities
                        #we first calculate the angles for each vector (from the current point to each of the surrounding points)
                        SP_vectors = [[pointpos[sp][k] - pointpos[current_point][k] for k in range(2)]  for sp in surrounding_points]
                        SP_angles = [GetAngle(SP_vectors[0], SP_vectors[k+1]) for k in range(3)]  #these angles are all based off of the CL_pp
                        #now get the permutation that puts these in order of increasing angle
                        sort_index = np.argsort(np.array(SP_angles))
                        #also, wrap up the Is convex info for both loops (will use 1st element for current loop info and 2nd for other loop info)
                        current_pt_convex = [loop_convexity[current_loop][current_index], loop_convexity[(current_loop+1)%2][other_index]]
                        
                        if Verbose:
                            print("Sort Index = ", sort_index, ", pt convex = ", current_pt_convex)
                        
                        
                        #now go throught the 6 options (each of which will have 4 sub options due to the convex/concave options for both loops at this point):
                        if sort_index[0] == 0:
                            if sort_index[1] == 1:  #[0,1,2] -> [CL_np,OL_pp,OL_np]
                                if current_pt_convex[0]:
                                    if current_pt_convex[1]: #[True,True] -> Convex, Convex #this combination shouldn't be possible
                                        print("This should not be able to occur: sort_index = ", sort_index, ", and current_pt_convex = ", current_pt_convex)
                                    else: #[True,False] -> Convex, Concave #no intersection.  Keep point, don't change current_loop
                                        rough_loop.append([True, current_point, False])
                                else:
                                    if current_pt_convex[1]: #[False,True] -> Concave, Convex #no intersection.  Keep point, don't change current_loop
                                        rough_loop.append([True, current_point, False])
                                    else: #[False,False] -> Concave, Concave #one intersection, the current point may or may not be removed (depending on the geometry), the current_loop will switch
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        cross_prod = SP_vectors[0][0]*SP_vectors[3][1] - SP_vectors[0][1]*SP_vectors[3][0]
                                        if cross_prod > 0:
                                            rough_loop.append([True, current_point, False])  #keep the point
                                        else:
                                            rough_loop.append([True, current_point, True])  #remove the point
                            else: #[0,2,1] -> [CL_np,OL_np,OL_pp]
                                if current_pt_convex[0]:
                                    if current_pt_convex[1]: #[True,True] -> Convex,Convex  #no intersection.  Keep point, don't change current_loop
                                        rough_loop.append([True, current_point, False])
                                    else:  #[True,False] -> Convex, Concave #not possible!
                                        print("This should not be able to occur: sort_index = ", sort_index, ", and current_pt_convex = ", current_pt_convex)
                                else:
                                    if current_pt_convex[1]:  #[False,True] -> Concave, Convex  #this is a double intersection, remove point, don't change current loop
                                        rough_loop.append([True, current_point, True])                                                                         
                                    else:  #[False,False] -> Concave, Concave #no intersection.  Keep point, don't change current_loop
                                        rough_loop.append([True, current_point, False])
                        elif sort_index[0] == 1:
                            if sort_index[1] == 0:  #[1,0,2] -> [OL_pp,CL_np,OL_np]
                                if current_pt_convex[0]:
                                    if current_pt_convex[1]: #[True,True] -> Convex, Convex #1 intersection, keep point, change loop
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, False])
                                    else:  #[True,False] -> Convex, Concave # 1 intersection, remove point, change loop
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, True]) 
                                else:
                                    if current_pt_convex[1]: #[False,True] -> Concave, Convex # 1 intersection, remove point, change loop
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, True]) 
                                    else:  #[False,False] -> Concave, Concave # 1 intersection, change loop, keep/remove point up to the geometry
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        cross_prod = SP_vectors[0][0]*SP_vectors[3][1] - SP_vectors[0][1]*SP_vectors[3][0]
                                        if cross_prod > 0:
                                            rough_loop.append([True, current_point, False])  #keep the point
                                        else:
                                            rough_loop.append([True, current_point, True])  #remove the point            
                            else:  #[1,2,0] -> [OL_pp,OL_np,CL_np]
                                if current_pt_convex[0]:
                                    if current_pt_convex[1]: #[True,True] -> Convex, Convex  #No intersections, keep point, don't change loop
                                        rough_loop.append([True, current_point, False])                                 
                                    else:  #[True,False] -> Convex, Concave #two intersections, remove point, don't change loop
                                        rough_loop.append([True, current_point, True])
                                else:
                                    if current_pt_convex[1]:   #[False,True] -> Concave, Convex #not possible!
                                        print("This should not be able to occur: sort_index = ", sort_index, ", and current_pt_convex = ", current_pt_convex)
                                    else:  #[False,False] -> Concave, Concave #No intersections, keep point, don't change loop
                                        rough_loop.append([True, current_point, False])                 
                        else:  #sort_index[0] == 2
                            if sort_index[1] == 0: #[2,0,1] -> [OL_np,CL_np,OL_pp]
                                if current_pt_convex[0]:
                                    if current_pt_convex[1]: #[True,True] -> Convex, Convex  # 1 intersection, change loop, keep/remove point up to the geometry
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        cross_prod = SP_vectors[0][0]*SP_vectors[3][1] - SP_vectors[0][1]*SP_vectors[3][0]
                                        if cross_prod > 0:
                                            rough_loop.append([True, current_point, True])  #remove the point
                                        else:
                                            rough_loop.append([True, current_point, False])  #keep the point         
                                    else:  #[True,False] -> Convex, Concave # 1 intersection, remove point, change loop
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, True])       
                                else:
                                    if current_pt_convex[1]:   #[False,True] -> Concave, Convex # 1 intersection, remove point, change loop
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, True])
                                    else:  #[False,False] -> Concave, Concave # 1 intersection, keep point, change loop
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, False])           
                            else:  #[2,1,0] -> [OL_np,OL_pp,CL_np]
                                if current_pt_convex[0]:
                                    if current_pt_convex[1]: #[True,True] -> Convex, Convex  # 2 intersections, use 1, change loop, keep/remove point up to the geometry
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        cross_prod = SP_vectors[0][0]*SP_vectors[3][1] - SP_vectors[0][1]*SP_vectors[3][0]
                                        if cross_prod > 0:
                                            rough_loop.append([True, current_point, True])  #remove the point
                                        else:
                                            rough_loop.append([True, current_point, False])  #keep the point
                                    else:  #[True,False] -> Convex, Concave # No intersections, keep point, don't change loop
                                        rough_loop.append([True, current_point, False])         
                                else:
                                    if current_pt_convex[1]:   #[False,True] -> Concave, Convex #no intersections, keep loop, keep point
                                        rough_loop.append([True, current_point, False])                                
                                    else:  #[False,False] -> Concave, Concave #not possible!
                                        print("This should not be able to occur: sort_index = ", sort_index, ", and current_pt_convex = ", current_pt_convex)
                        if Verbose:                
                            print("rough loop = ", rough_loop)               
                        
                    elif len(red_surrounding_points) == 3:
                        if Verbose:
                            print("Case of 3 unique surrounding points - i.e. some parallel lines")
                            print("LRside = ", LRside)
                        #We need to determine which case we are dealing with.  We first find the pair of surrounding_points that are the same.
                        same_points = []
                        for i in range(3):
                            for j in range(i+1,4):
                                if surrounding_points[i] == surrounding_points[j]:
                                    same_points = [i,j]
                                    break
                            if len(same_points) > 0:
                                break
                        #now we get the ordering of surrounding points
                        
                        SP_vectors = [[pointpos[sp][k] - pointpos[current_point][k] for k in range(2)]  for sp in surrounding_points]
                        SP_angles = [GetAngle(SP_vectors[0], SP_vectors[k+1]) for k in range(3)]  #these angles are all based off of the CL_pp
                        #now get the permutation that puts these in order of increasing angle
                        #print("SP_vectors = ", SP_vectors)
                        #print("SP_angles = ", SP_angles)
                        sort_index = np.argsort(np.array(SP_angles))
                        current_pt_convex = [loop_convexity[current_loop][current_index], loop_convexity[(current_loop+1)%2][other_index]]
                        
                        #print("Sort Index = ", sort_index, ", pt convex = ", current_pt_convex)
                        #print("same_points = ", same_points)
                        
                        if same_points[0] == 0:
                            
                            #we need to find the RLside data in the case that this is a first point to be considered (in this case RLside == None)
                            if LRside == None:
                                if len(loop[current_loop]) == 2:
                                    LRside = "Right"
                                else:
                                    #first find whether the current loop and other loop are going in the same or different directions on the parallel segment
                                    if CL_pp == OL_pp:
                                        #same direction
                                        c_p_temp = CL_pp
                                        c_index_temp = (current_index-1)%loop_len[current_loop]
                                        o_index_temp = loop[(current_loop+1)%2].index(c_p_temp)
                                        c_pt_temp_convex = [loop_convexity[current_loop][c_index_temp], loop_convexity[(current_loop+1)%2][o_index_temp]]
                                        CL_ppp = loop[current_loop][(c_index_temp-1)%loop_len[current_loop]]
                                        OL_ppp = loop[(current_loop+1)%2][(o_index_temp-1)%loop_len[(current_loop+1)%2]]
                                        #now find the three vectors to compare
                                        base_vec = [pointpos[current_point][k] - pointpos[c_p_temp][k] for k in range(2)]
                                        CL_ppp_vec = [pointpos[CL_ppp][k] - pointpos[c_p_temp][k] for k in range(2)]
                                        OL_ppp_vec = [pointpos[OL_ppp][k] - pointpos[c_p_temp][k] for k in range(2)]
                                        CL_ppp_ang = GetAngle(base_vec, CL_ppp_vec)
                                        OL_ppp_ang = GetAngle(base_vec, OL_ppp_vec)
                                        if CL_ppp_ang < OL_ppp_ang:
                                            if c_pt_temp_convex == [True, True] or c_pt_temp_convex == [False, False]:
                                                LRside = "Left"
                                            else:
                                                LRside = "Right"
                                        else:
                                            if c_pt_temp_convex == [True, True] or c_pt_temp_convex == [False, False]:
                                                LRside = "Right"
                                            else:
                                                LRside = "Left"

                                    else:
                                        #opposite direction
                                        c_p_temp = CL_pp
                                        c_index_temp = (current_index-1)%loop_len[current_loop]
                                        o_index_temp = loop[(current_loop+1)%2].index(c_p_temp)
                                        c_pt_temp_convex = [loop_convexity[current_loop][c_index_temp], loop_convexity[(current_loop+1)%2][o_index_temp]]
                                        CL_ppp = loop[current_loop][(c_index_temp-1)%loop_len[current_loop]]
                                        OL_nnp = loop[(current_loop+1)%2][(o_index_temp+1)%loop_len[(current_loop+1)%2]]
                                        #now find the three vectors to compare
                                        base_vec = [pointpos[current_point][k] - pointpos[c_p_temp][k] for k in range(2)]
                                        CL_ppp_vec = [pointpos[CL_ppp][k] - pointpos[c_p_temp][k] for k in range(2)]
                                        OL_nnp_vec = [pointpos[OL_nnp][k] - pointpos[c_p_temp][k] for k in range(2)]
                                        CL_ppp_ang = GetAngle(base_vec, CL_ppp_vec)
                                        OL_nnp_ang = GetAngle(base_vec, OL_nnp_vec)
                                        if CL_ppp_ang < OL_nnp_ang:
                                            if c_pt_temp_convex == [True, False] or c_pt_temp_convex == [False, True]:
                                                LRside = "Left"
                                            else:
                                                LRside = "Right"
                                        else:
                                            if c_pt_temp_convex == [True, False] or c_pt_temp_convex == [False, True]:
                                                LRside = "Right"
                                            else:
                                                LRside = "Left"

                            #####################################
                            
                            if same_points[1] == 2:
                                if sort_index[1] == 0:
                                    #[CL_pp = OL_pp,CL_np,OL_np]
                                    if current_pt_convex == [True,True] or current_pt_convex == [False,False]:   #CV,CV or CC, CC
                                        if LRside == "Right":
                                            rough_loop.append([True, current_point, False]) #No intersections, keep point, don't change loop
                                        elif LRside == "Left":
                                            current_loop = (current_loop+1)%2
                                            current_index = other_index
                                            rough_loop.append([True, current_point, False])  #intersection, keep point, change loop
                                                
                                    elif current_pt_convex == [False,True]: #CC,CV
                                        if LRside == "Right":
                                            rough_loop.append([True, current_point, True]) #2 intersections, RM point, don't change loop
                                        elif LRside == "Left":
                                            current_loop = (current_loop+1)%2
                                            current_index = other_index
                                            rough_loop.append([True, current_point, True])  #intersection, RM point, change loop
                                                
                                    else: #CV,CC
                                        print("Impossible Combination")
                                    LRside = None 
                                else:
                                    #[CL_pp = OL_pp,OL_np,CL_np]
                                    if current_pt_convex == [True,True] or current_pt_convex == [False,False]:   #CV,CV or CC, CC
                                        if LRside == "Right":
                                            current_loop = (current_loop+1)%2
                                            current_index = other_index
                                            rough_loop.append([True, current_point, False])  #intersection, keep point, change loop
                                        elif LRside == "Left":
                                            rough_loop.append([True, current_point, False]) #No intersections, keep point, don't change loop
                                        
                                    elif current_pt_convex == [True,False]: # CV, CC
                                        if LRside == "Right":
                                            current_loop = (current_loop+1)%2
                                            current_index = other_index
                                            rough_loop.append([True, current_point, True])  #intersection, RM point, change loop
                                        elif LRside == "Left":
                                            rough_loop.append([True, current_point, True]) #2 intersections, RM point, don't change loop
                                        
                                    else:  #CC, CV
                                        print("Impossible Combination")
                                    LRside = None 
                            elif same_points[1] == 3:
                                if sort_index[1] == 0:
                                    #[CL_pp = OL_np,CL_np,OL_pp]
                                    if current_pt_convex == [True, False] or current_pt_convex == [False, False]:
                                        if LRside == "Right":
                                            rough_loop.append([True, current_point, False]) #No intersections, keep point, don't change loop
                                            LRside = None 
                                        elif LRside == "Left":
                                            current_loop = (current_loop+1)%2
                                            current_index = other_index
                                            rough_loop.append([True, current_point, True])  #intersection, remove point, change loop
                                            #LRside is unchanged
                                                                
                                    elif current_pt_convex == [False, True]:
                                        
                                        if LRside == "Right":
                                            current_loop = (current_loop+1)%2
                                            current_index = other_index
                                            rough_loop.append([True, current_point, True])  #intersection, remove point, change loop
                                        elif LRside == "Left":
                                            current_loop = (current_loop+1)%2
                                            current_index = other_index
                                            rough_loop.append([True, current_point, False]) #intersections, keep point, change loop
                                        #LRside is unchanged
                                    else:  #CV, CV
                                        print("Impossible Combination") 
                                
                                else:
                                    #[CL_pp = OL_np,OL_pp,CL_np]
                                    if current_pt_convex == [True, False] or current_pt_convex == [False, True]:
                                        if LRside == "Right":
                                            current_loop = (current_loop+1)%2
                                            current_index = other_index
                                            rough_loop.append([True, current_point, True])  #intersection, remove point, change loop
                                            #LRside is unchanged
                                        elif LRside == "Left":
                                            rough_loop.append([True, current_point, False]) #keep point, keep loop
                                            LRside = None
                                    elif current_pt_convex == [True, True]:
                                        if LRside == "Right":
                                            current_loop = (current_loop+1)%2
                                            current_index = other_index
                                            rough_loop.append([True, current_point, False])  #intersection, keep point, change loop
                                        elif LRside == "Left":
                                            current_loop = (current_loop+1)%2
                                            current_index = other_index
                                            rough_loop.append([True, current_point, True])  #intersection, remove point, change loop
                                        #LRside is unchanged
                                    else:  #CC, CC
                                        print("Impossible Combination") 
                                        
                            else: #same_points[1] = 1
                                if sort_index[2] == 2:
                                    #[CL_pp = CL_np, OL_pp,  OL_np]
                                    if current_pt_convex[1]:
                                        rough_loop.append([True, current_point, False]) 
                                    else:
                                        rough_loop.append([True, current_point, True])
                                else:
                                    #[CL_pp = CL_np, OL_np,  OL_pp]
                                    if current_pt_convex[1]:
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        if SP_angles[2] < np.pi:
                                            rough_loop.append([True, current_point, True]) 
                                        else:
                                            rough_loop.append([True, current_point, False])
                                    else:
                                        rough_loop.append([True, current_point, False])
                                    
                                    
                            
                        elif same_points[0] == 1:
                            if same_points[1] == 2:
                                if sort_index[2] == 2:
                                    #[CL_pp,CL_np = OL_pp, OL_np]
                                    if current_pt_convex == [True, False] or current_pt_convex == [False, True]:
                                        rough_loop.append([True, current_point, False])  #no intersection, keep point, keep loop
                                        LRside = "Right"
                                    elif current_pt_convex == [False, False]:
                                        #swap loop
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        LRside = None
                                        #RM or Keep pt based on geometry
                                        if SP_angles[2] < np.pi:
                                            #keep
                                            rough_loop.append([True, current_point, False])
                                        else:
                                            #remove
                                            rough_loop.append([True, current_point, True])
                                    else:  #CV, CV
                                        print("Impossible Combination") 
                                    
                                
                                else:
                                    #[CL_pp,OL_np,CL_np = OL_pp]
                                    if current_pt_convex == [True, False] or current_pt_convex == [False, True]:
                                        rough_loop.append([True, current_point, False])  #no intersection, keep point, keep loop
                                        LRside = "Left"
                                    elif current_pt_convex == [True, True]:
                                        #swap loop
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        #RM or Keep pt based on geometry
                                        if SP_angles[2] < np.pi:
                                            #remove
                                            rough_loop.append([True, current_point, True])
                                        else:
                                            #keep
                                            rough_loop.append([True, current_point, False])
                                        LRside = None
                                    else:  #CC, CC
                                        print("Impossible Combination") 
                            
                            else:  #same_points[1] == 3:
                                if sort_index[2] == 1:
                                    #[CL_pp,CL_np = OL_np, OL_pp]
                                    if current_pt_convex == [True, True] or current_pt_convex == [False, False]:
                                        rough_loop.append([True, current_point, False])  #no intersection, keep point, keep loop
                                        LRside = "Right"
                                    elif current_pt_convex == [False, True]:
                                        #swap loop
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        #RM pt 
                                        rough_loop.append([True, current_point, True])
                                        LRside = "Right"
                                    else:  #CV, CC
                                        print("Impossible Combination") 
                                    
                                else:
                                    #[CL_pp,OL_pp,CL_np = OL_np]
                                    if current_pt_convex == [True, True] or current_pt_convex == [False, False]:
                                        rough_loop.append([True, current_point, False])  #no intersection, keep point, keep loop
                                        LRside = "Left"
                                    elif current_pt_convex == [True, False]:
                                        #swap loop
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        #RM pt 
                                        rough_loop.append([True, current_point, True])
                                        LRside = "Left"
                                    else:  #CC, CV
                                        print("Impossible Combination") 
                                        
                        else: #same_points[0] == 2
                            if sort_index[2] == 0:
                                #[CL_pp, OL_pp = OL_np, CL_np]
                                if current_pt_convex[0]:
                                    current_loop = (current_loop+1)%2
                                    current_index = other_index
                                    if SP_angles[2] < np.pi:
                                        rough_loop.append([True, current_point, True]) 
                                    else:
                                        rough_loop.append([True, current_point, False])
                                else:
                                    rough_loop.append([True, current_point, False])
                            else:
                                #[CL_pp, CL_np, OL_pp = OL_np]
                                if current_pt_convex[0]:
                                    rough_loop.append([True, current_point, False])
                                else:
                                    rough_loop.append([True, current_point, True])
                                    
                                    
                                    
                        
                    elif len(red_surrounding_points) == 2:
                        if Verbose:
                            print("Case of two unique surrounding points.   LRside = ",  LRside)
                        #keep Right/Left
                        #we need to explicitly deal with the case where both loops are 2-point loops
                        
                        
                        #if  LRside == None, then we must go back along the current loop until we get a case of three unique points.  We then find the correct L/R designation
                        if LRside == None:
                            if len(loop[current_loop]) == 2 or len(loop[(current_loop+1)%2]) == 2:
                                LRside = "Right"
                            else:
                            
                                step_back = 1
                                found_end = False
                                CL_pp_step = CL_np_step = OL_pp_step = OL_np_step = None
                                current_point_step = other_index_step = current_index_step = None
                                while not found_end:
                                    current_index_step = (current_index-step_back)%loop_len[current_loop]
                                    current_point_step = loop[current_loop][current_index_step]
                                    CL_pp_step = loop[current_loop][(current_index_step-1)% loop_len[current_loop]] 
                                    CL_np_step = loop[current_loop][(current_index_step+1)% loop_len[current_loop]]  
                                    other_index_step = loop[(current_loop+1)%2].index(current_point_step)
                                    OL_pp_step = loop[(current_loop+1)%2][(other_index_step-1)% loop_len[(current_loop+1)%2]]  #other loop, previous point
                                    OL_np_step = loop[(current_loop+1)%2][(other_index_step+1)% loop_len[(current_loop+1)%2]]  #other loop, next point
                                    surrounding_points_step = [CL_pp_step,CL_np_step,OL_pp_step,OL_np_step]
                                    red_surrounding_points_step = list(set(surrounding_points_step))
                                    if len(red_surrounding_points_step) == 3:
                                        #reached the end
                                        found_end = True
                                    else:
                                        step_back  += 1

                                c_pt_step_convex = [loop_convexity[current_loop][current_index_step], loop_convexity[(current_loop+1)%2][other_index_step]]

                                ################
                                if CL_pp == OL_pp:
                                    #same direction
                                    #now find the three vectors to compare
                                    base_vec = [pointpos[CL_np_step][k] - pointpos[current_point_step][k] for k in range(2)]
                                    CL_pp_vec = [pointpos[CL_pp_step][k] - pointpos[current_point_step][k] for k in range(2)]
                                    OL_pp_vec = [pointpos[OL_pp_step][k] - pointpos[current_point_step][k] for k in range(2)]
                                    CL_pp_ang = GetAngle(base_vec, CL_pp_vec)
                                    OL_pp_ang = GetAngle(base_vec, OL_pp_vec)
                                    if CL_pp_ang < OL_pp_ang:
                                        if c_pt_step_convex == [True, True] or c_pt_step_convex == [False, False]:
                                            LRside = "Left"
                                        else:
                                            LRside = "Right"
                                    else:
                                        if c_pt_step_convex == [True, True] or c_pt_step_convex == [False, False]:
                                            LRside = "Right"
                                        else:
                                            LRside = "Left"

                                else:
                                    ####there are more option if three points coincide********
                                    #opposite direction
                                    #now find the three vectors to compare
                                    base_vec = [pointpos[CL_np_step][k] - pointpos[current_point_step][k] for k in range(2)]
                                    CL_pp_vec = [pointpos[CL_pp_step][k] - pointpos[current_point_step][k] for k in range(2)]
                                    OL_np_vec = [pointpos[OL_np_step][k] - pointpos[current_point_step][k] for k in range(2)]
                                    CL_pp_ang = GetAngle(base_vec, CL_pp_vec)
                                    OL_np_ang = GetAngle(base_vec, OL_np_vec)
                                    if CL_pp_ang < OL_np_ang:
                                        if c_pt_step_convex == [True, False] or c_pt_step_convex == [False, True]:
                                            LRside = "Left"
                                        else:
                                            LRside = "Right"
                                    else:
                                        if c_pt_step_convex == [True, False] or c_pt_step_convex == [False, True]:
                                            LRside = "Right"
                                        else:
                                            LRside = "Left"
                        
                        #now for the options
                        if CL_pp == CL_np:
                            if OL_pp == OL_np:
                                #print("CL_pp = CL_np, OL_pp = OL_np")
                                #for now, we assume both sides are 2-pt loops (and the CCW orientation that goes with them) 
                                #need geometry
                                v1 = [pointpos[current_point][k] - pointpos[CL_pp][k] for k in range(2)]
                                v2 = [pointpos[OL_pp][k] - pointpos[current_point][k] for k in range(2)]
                                v1crossv2 = v1[0]*v2[1]-v1[1]*v2[0]
                                if v1crossv2 > 0: 
                                    rough_loop.append([True, current_point, False])
                                else:
                                    rough_loop.append([True, current_point, True])
                                current_loop = (current_loop+1)%2
                                current_index = other_index
                            elif OL_pp == CL_pp:
                                #print("CL_pp = CL_np = OL_pp, OL_np")
                                OL_conv = [loop_convexity[(current_loop+1)%2][other_index], loop_convexity[(current_loop+1)%2][(other_index-1)%loop_len[(current_loop+1)%2]]]
                                #print("OL_conv = ", OL_conv)
                                if OL_conv[0]:
                                    if OL_conv[1]:
                                        rough_loop.append([True, current_point, False])
                                    else:
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, False])
                                else:
                                    if OL_conv[1]:
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, True])
                                    else:
                                        rough_loop.append([True, current_point, False])
                                
                            else:  # OL_np == CL_pp:
                                #print("CL_pp = CL_np = OL_np, OL_pp")
                                OL_conv = [loop_convexity[(current_loop+1)%2][other_index], loop_convexity[(current_loop+1)%2][(other_index+1)%loop_len[(current_loop+1)%2]]]
                                #print("OL_conv = ", OL_conv)
                                if OL_conv[0]:
                                    if OL_conv[1]:
                                        rough_loop.append([True, current_point, False])
                                    else:
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, False])
                                else:
                                    if OL_conv[1]:
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, True])
                                    else:
                                        rough_loop.append([True, current_point, False])
                                
                                     
                        elif OL_pp == OL_np:
                            if CL_pp == OL_pp:
                                #print("CL_pp = OL_np = OL_pp, CL_np")
                                CL_conv = [loop_convexity[current_loop][current_index], loop_convexity[current_loop][(current_index-1)%loop_len[current_loop]]]
                                #print("CL_conv = ", CL_conv)
                                if CL_conv[0]:
                                    if CL_conv[1]:
                                        rough_loop.append([True, current_point, False])
                                    else:
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, False])
                                else:
                                    if CL_conv[1]:
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, True])
                                    else:
                                        rough_loop.append([True, current_point, False])
                                
                            else: #CL_np == OL_pp
                                #print("CL_pp, OL_np = OL_pp = CL_np")
                                CL_conv = [loop_convexity[current_loop][current_index], loop_convexity[current_loop][(current_index+1)%loop_len[current_loop]]]
                                #print("CL_conv = ", CL_conv)
                                if CL_conv[0]:
                                    if CL_conv[1]:
                                        rough_loop.append([True, current_point, False])
                                    else:
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, False])
                                else:
                                    if CL_conv[1]:
                                        current_loop = (current_loop+1)%2
                                        current_index = other_index
                                        rough_loop.append([True, current_point, True])
                                    else:
                                        rough_loop.append([True, current_point, False])
                            
                        else:
                            #include new point (Keep)
                            rough_loop.append([True, current_point, False])
                        
                    else:
                        print("Error: there should be no less than 2 unique points in the surrounding point set")
                        #there actually could be a valid case, but it involves a folded part of one loop (and a ptpair loop too), so we are going to ignore it for now.  ### revisit ******
            
            exhausted_points = True
            for TF in points_used[current_loop]:
                if not TF:
                    exhausted_points = False
                    break
            if exhausted_points:
                for TF in points_used[(current_loop+1)%2]:
                    if not TF:
                        exhausted_points = False
                        switch_starting_loop = True
                        break
            #now append rough loop to rough_new_loops
            rough_new_loops.append(rough_loop)
        
        
        #next we need to check each rough loop to see if any are CW.  If so, we then reverse their order to be CCW
        for i in range(len(rough_new_loops)):
            pos_list = []
            for j in range(len(rough_new_loops[i])):
                if rough_new_loops[i][j][0]:
                    pos_list.append(pointpos[rough_new_loops[i][j][1]])
                else:
                    pos_list.append(rough_new_loops[i][j][1])
            angle_accum = 0
            rnllen = len(rough_new_loops[i])
            for j in range(rnllen):
                v1 = [pos_list[(j+1)%rnllen][k] - pos_list[j][k] for k in range(2)]
                v2 = [pos_list[(j+2)%rnllen][k] - pos_list[(j+1)%rnllen][k] for k in range(2)]
                angle_accum += GetAngle(v1, v2, True)
            #print("for rough loop number ", i, ", the accumulated angle is ", angle_accum)
            if angle_accum < 0:
                #this loop is going the wrong way.  We just reverse its direction
                rough_new_loops[i].reverse()
        
        if Verbose:
            print(rough_new_loops)
        
        #Now we need to take each rough loop and relax it.  This removes each removable point and relaxes the local band structure
        
        if Verbose:
            print("Relaxing loops now")
        
        for rough_loop in rough_new_loops:
            if Verbose:
                print("relaxing this rough loop: ", rough_loop)
            all_relaxed = False
            while not all_relaxed:
                if Verbose:
                    print("Not all relaxed")
                    print("Current rough loop = ", rough_loop)
                all_relaxed = True
                for i in range(len(rough_loop)):
                    if rough_loop[i][2]:
                        all_relaxed = False
                        if Verbose:
                            print("considering ", rough_loop[i])
                        #the element at position i needs to be dealt with
                        #first determine convex vs concave for this point and the two that border it, and determine if either of the two bordering points is pinned (i.e. will be removed)
                        Is_Convex = [None, None, None]
                        point_locations = []
                        point_ids = []
                        for k in range(5):
                            index = (i-2+k)%len(rough_loop)
                            if rough_loop[index][0]:
                                point_locations.append(pointpos[rough_loop[index][1]])
                                if k in [1,2,3]:
                                    point_ids.append(rough_loop[index][1])
                            else:
                                point_locations.append(rough_loop[index][1])
                        temp_vectors = []
                        for k in range(4):
                            temp_vectors.append([point_locations[k+1][s] - point_locations[k][s] for s in range(2)])
                        for k in range(3):
                            #get the cross product
                            cross_prod = temp_vectors[k][0]*temp_vectors[k+1][1] - temp_vectors[k][1]*temp_vectors[k+1][0]
                            if cross_prod >= 0:
                                Is_Convex[k] = True
                            else:
                                Is_Convex[k] = False
                                
                        Is_Pinned = [rough_loop[(i-1+k)%len(rough_loop)][2] for k in range(3)]
                        if Is_Convex[1]:
                            #the central point is convex
                            triangle_point_list = [point_locations[1+k] for k in range(3)]
                            Interior_ids = []
                            Interior_pt_pos = []
                            for k in range(tri.ptnum):
                                if not k in point_ids:
                                    trial_pt_loc = tri.pointpos[k]
                                    IsInterior = InteriorToConvexSet2(trial_pt_loc, triangle_point_list,tri, True)
                                    if IsInterior[0]:
                                        Interior_ids.append(k)
                                        Interior_pt_pos.append(IsInterior[1])
                                        pointpos[k] = IsInterior[1]
                            
                            points_to_add = None
                            if len(Interior_ids) > 0:
                                cv_hull_trial_pts = [triangle_point_list[0]] + Interior_pt_pos + [triangle_point_list[2]]
                                cv_hull_trial_pt_ids = [point_ids[0]] + Interior_ids + [point_ids[2]]
                                CH_meta_ids = ConvexHull(cv_hull_trial_pts).vertices
                                CH_ids = [cv_hull_trial_pt_ids[k] for k in CH_meta_ids]
                                
                                while not CH_ids[0] == point_ids[0]:
                                    CH_ids.append(CH_ids.pop(0))
                                del CH_ids[0]
                                del CH_ids[-1]
                                points_to_add = CH_ids
                            
                            #now let's record the position in rough loop for the initial and final points
                            Initial_pt_loc = (i-1)%len(rough_loop)
                            Remove_pt_loc = i
                            Final_pt_loc = (i+1)%len(rough_loop)
                            
                            #first check the initial and final points to see if we need to mark them to be removed (if their convexity has changed)
                            if not Is_Pinned[0]:
                                #Is_Convex[0]
                                v1 = temp_vectors[0]
                                v2 = None
                                if points_to_add == None:
                                    v2 = [point_locations[-2][k] - point_locations[1][k] for k in range(2)]
                                else:
                                    #### revisit this ********
                                    v2 = [pointpos[points_to_add[0]][k] - point_locations[1][k] for k in range(2)]
                                cross_prod = v1[0]*v2[1] - v1[1]*v2[0]
                                new_conv = None
                                if cross_prod >= 0:
                                    new_conv = True
                                else:
                                    new_conv = False
                                
                                if not new_conv == Is_Convex[0]:
                                    rough_loop[Initial_pt_loc][2] = True
                            
                            if not Is_Pinned[2]:
                                #Is_Convex[0]
                                v1 = None
                                if points_to_add == None:
                                    v1 = [point_locations[-2][k] - point_locations[1][k] for k in range(2)]
                                else:
                                    v1 = [point_locations[-2][k] - pointpos[points_to_add[-1]][k] for k in range(2)]
                                v2 = temp_vectors[-1]
                                cross_prod = v1[0]*v2[1] - v1[1]*v2[0]
                                new_conv = None
                                if cross_prod >= 0:
                                    new_conv = True
                                else:
                                    new_conv = False
                                
                                if not new_conv == Is_Convex[-1]:
                                    rough_loop[Final_pt_loc][2] = True
                            
                            place_to_add = Remove_pt_loc
                            if not points_to_add == None:
                                for Pt_add in points_to_add:
                                    rough_loop.insert(place_to_add,[True,Pt_add,False])
                                    place_to_add += 1
                            Remove_pt_loc = place_to_add                             
                            del rough_loop[Remove_pt_loc]
                            
                        else:
                            #the central point is concave
                            triangle_point_list = [point_locations[3], point_locations[2], point_locations[1]]
                            Interior_ids = []
                            Interior_pt_pos = []
                            for k in range(tri.ptnum):
                                if not k in point_ids:
                                    trial_pt_loc = tri.pointpos[k]
                                    IsInterior = InteriorToConvexSet2(trial_pt_loc, triangle_point_list,tri, True)
                                    if IsInterior[0]:
                                        Interior_ids.append(k)
                                        Interior_pt_pos.append(IsInterior[1])
                                        pointpos[k] = IsInterior[1]
                            
                            points_to_add = None
                            if len(Interior_ids) > 0:
                                cv_hull_trial_pts = [triangle_point_list[0]] + Interior_pt_pos + [triangle_point_list[2]]
                                cv_hull_trial_pt_ids = [point_ids[2]] + Interior_ids + [point_ids[0]]
                                CH_meta_ids = ConvexHull(cv_hull_trial_pts).vertices
                                CH_ids = [cv_hull_trial_pt_ids[k] for k in CH_meta_ids]
                                
                                while not CH_ids[0] == point_ids[2]:
                                    CH_ids.append(CH_ids.pop(0))
                                del CH_ids[0]
                                del CH_ids[-1]
                                #reverse the list
                                CH_ids.reverse()
                                points_to_add = CH_ids
                            
                            #now let's record the position in rough loop for the initial and final points
                            Initial_pt_loc = (i-1)%len(rough_loop)
                            Remove_pt_loc = i
                            Final_pt_loc = (i+1)%len(rough_loop)
                            
                            #first check the initial and final points to see if we need to mark them to be removed (if their convexity has changed)
                            if not Is_Pinned[0]:
                                #Is_Convex[0]
                                v1 = temp_vectors[0]
                                v2 = None
                                if points_to_add == None:
                                    v2 = [point_locations[-2][k] - point_locations[1][k] for k in range(2)]
                                else:
                                    v2 = [pointpos[points_to_add[0]][k] - point_locations[1][k] for k in range(2)]
                                cross_prod = v1[0]*v2[1] - v1[1]*v2[0]
                                new_conv = None
                                if cross_prod >= 0:
                                    new_conv = True
                                else:
                                    new_conv = False
                                
                                if not new_conv == Is_Convex[0]:
                                    rough_loop[Initial_pt_loc][2] = True
                            
                            if not Is_Pinned[2]:
                                #Is_Convex[0]
                                v1 = None
                                if points_to_add == None:
                                    v1 = [point_locations[-2][k] - point_locations[1][k] for k in range(2)]
                                else:
                                    v1 = [point_locations[-2][k] - pointpos[points_to_add[-1]][k] for k in range(2)]
                                v2 = temp_vectors[-1]
                                cross_prod = v1[0]*v2[1] - v1[1]*v2[0]
                                new_conv = None
                                if cross_prod >= 0:
                                    new_conv = True
                                else:
                                    new_conv = False
                                
                                if not new_conv == Is_Convex[-1]:
                                    rough_loop[Final_pt_loc][2] = True                            
                            place_to_add = Remove_pt_loc
                            if not points_to_add == None:
                                for Pt_add in points_to_add:
                                    rough_loop.insert(place_to_add,[True,Pt_add,False])
                                    place_to_add += 1
                            Remove_pt_loc = place_to_add                             
                            del rough_loop[Remove_pt_loc]
        
                        break
            
        #will use: InteriorToConvexSet(trial_pt, convex_set, tri):
        if Verbose:
            print(rough_new_loops)
        new_loops = [[lp[i][1] for i in range(len(lp))] for lp in rough_new_loops]
        if Verbose:
            print("New Loops",new_loops)
        #finally, we check each loop for their interior points and compare against our expectations:  we might get some loops whose interior is some subset of the intersection of the original loop interiors (we discard these), we might get a loop whose interior is equal to the union of the original loop interiors (this is the one we want), or we might have that the difference between the interiors of two loops is the union of the original loop interiors (we want these two or more loops)
        output_loops = []
        L1IntL2 = set(L1L2Intersection)
        L1UnionL2 = set(L1L2Union)
        for n_loop in new_loops:
            IP = set(GetInteriorPoints(n_loop,tri))
            if IP == L1UnionL2:
                output_loops.append(n_loop)
                #print("surrounding loop = ", n_loop)
            else:
                if not IP <= L1IntL2:
                    output_loops.append(n_loop)
                    #print("interior bounding loop = ", n_loop)
        
        #then we return these loops
        #return output_loops
        #we are just returning the outer bounding loop for now
        if Just_Largest:
            max_loc = 0
            prev_len = 0
            for i in range(len(output_loops)):
                if len(output_loops[i]) > prev_len:
                    prev_len = len(output_loops[i])
                    max_loc = i
            return output_loops[max_loc]
        else:
            return output_loops
        
        
        
#quick function to get the angle [0,2*pi] between vec1 and vec2 (order matters)
#if Split == True, then we output [0,pi] for a positice v1crossv2, and [0,-pi] for a negative cross product
def GetAngle(v1, v2, Split = False):
    v1_hat = v1 / np.linalg.norm(v1)
    v2_hat = v2 / np.linalg.norm(v2)
    dot_p = np.dot(v1_hat,v2_hat)
    #v1dotv2 = v1[0]*v2[0] + v1[1]*v2[1]
    #v1crossv2 = v1[0]*v2[1] - v1[1]*v2[0]
    cross_p = np.cross(v1_hat,v2_hat)
    #v1_mag = (v1[0]**2+v1[1]**2)**(0.5)
    #v2_mag = (v2[0]**2+v2[1]**2)**(0.5)
    #print(v1dotv2, v1crossv2, v1_mag, v2_mag)
    #print(v1dotv2/(v1_mag*v2_mag))
    if dot_p > 1.0:
        dot_p = 1.0
    elif dot_p < -1.0:
        dot_p = -1.0
    theta = np.arccos(dot_p)
    if cross_p >= 0.0:
        return theta
    else:
        if not Split:
            return 2*np.pi - theta
        else:
            return -1*theta
  
        
