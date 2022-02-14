
##########
# CREDIT #
##########

# Written by: Vyshnavi Karra
# Edited by: Vyshnavi Karra 
# University: Northeastern University 
# Advisor: Francisco Hung 

# Last Updated: 01-05-2022 

# For any questions, please create an issue on GitHub

# Notes: 
# 1) This is based on my master's thesis MATLAB code (paper is here: https://rucore.libraries.rutgers.edu/rutgers-lib/55523/PDF/1/play/):
# 2) curvature: https://en.wikipedia.org/wiki/Curvature
# 3) torsion: https://en.wikipedia.org/wiki/Torsion_of_a_curve
# 4) more math stuff: https://math.gmu.edu/~rsachs/math215/textbook/Math215Ch2Sec3.pdf

##########
# SET UP # 
##########

import numpy as np 
import scipy as sp 
import math as m 
import matplotlib.pyplot as plt
import glob 
import matplotlib.patches as mpatches
import os
import itertools
import seaborn as sns
sns.set_style("white")

#############
# FUNCTIONS #
#############
def get_box(path):
    """
    Get the box size and timestep from the gro/g96 file. 
    """
    timestep = []
    box_size = []
    
    with open(path) as f:
        for line in f:
            if (line.lstrip().startswith('TIMESTEP')):
                timestep.append(next(f).split())
            elif (line.lstrip().startswith('BOX')):
                box_size.append(next(f).split())

    return timestep, box_size

def get_inner_pairs(n_rings, ring_beads): 
    """
    Get the pairs for the inner radius function. 
    """
    first_bead = [8,9,13,14,19,18]
    second_bead = [23,24,28,29,4,3]
    inner_first_bead = []
    inner_second_bead = []
    
    for i in range(len(first_bead)): 
        for j in range(n_rings): 
            inner_first_bead.append(first_bead[i]+(j*ring_beads))
            inner_second_bead.append(second_bead[i]+(j*ring_beads))
            
    return inner_first_bead, inner_second_bead

def get_outer_pairs(n_rings, ring_beads): 
    """
    Get the pairs for the outer radius function. 
    """
    first_bead = [21,25,26,15,16,20]
    second_bead = [6,10,11,30,1,5]
    outer_first_bead = []
    outer_second_bead = []
    
    for i in range(len(first_bead)): 
        for j in range(n_rings): 
            outer_first_bead.append(first_bead[i]+(j*ring_beads))
            outer_second_bead.append(second_bead[i]+(j*ring_beads))
            
    return outer_first_bead, outer_second_bead

def get_inner_radius(x, y, z, first, second, bead_size, x_box,y_box,z_box): 
    """
    ~~INNER RADIUS~~
    The pairs are beads 8--23, 9--24, 13--28, 14--29, 19--4, 18--3 for the 1st ring. 
    Following rings are (ring #-1)*30 + those pairs. 
    (eg. 2nd ring pairs would be: 38--53, 39--54, 43--58, 44--59, 49--34, 48--3.)
    """
    inner_radius = [] 
    for i in range(len(x)): 
        for j in range(len(first)): 
            inner_radius.append(float((np.sqrt((x[first[j]]-x[second[j]])**2 + (y[first[j]]-y[second[j]])**2 + (z[first[j]]-z[second[j]])**2))/2) - bead_size)    
    return inner_radius 

def get_outer_radius(x, y, z, first, second, bead_size, x_box,y_box,z_box):
    """
    ~~OUTER RADIUS~~ 
    The pairs are beads 21--6, 25--10, 26--11, 15--30, 16--1, 20--5 for the 1st ring. 
    Following rings follow the same pattern as with the inner radius. 
    """
    outer_radius = [] 
    for i in range(len(x)): 
        for j in range(len(first)):
            outer_radius.append(float((np.sqrt((x[first[j]]-x[second[j]])**2 + (y[first[j]]-y[second[j]])**2 + (z[first[j]]-z[second[j]])**2))/2) + bead_size)
    return outer_radius 

def get_axial_rise(x, y, z, first, second, x_box,y_box,z_box): 
    """
    Get the axial rise which is the distance between bead 5 in ring 1 and bead 37 in ring 2. 
    """
    
    axial_rise = []
    
    for i in range(len(x)): 
        for j in range(len(first)): 
            axial_rise.append(0.5*np.sqrt((x[first[j]-1]-x[second[j]-1])**2 + (y[first[j]-1]-y[second[j]-1])**2 + (z[first[j]-1]-z[second[j]-1])**2)/2)
             
    return axial_rise

###############
# USER INPUTS #
###############

n_rings = 10
ring_beads = 30

#############
# LOAD FILE #
#############

path = ''
timestep, box_size = get_box(path)
x = []
y = []
z = []


with open(path) as f, open('coordinates.txt','w') as fout:
    for line in f:
        it = itertools.dropwhile(lambda line: line.strip() != 'POSITIONRED', f) 
        if next(it, None) is None: 
            break
        fout.writelines(itertools.takewhile(lambda line: line.strip() != 'END', it))

with open('coordinates.txt') as f: 
    for line in f:
        x.append(float(line.split()[0]))
        y.append(float(line.split()[1]))
        z.append(float(line.split()[2]))
    f.close()

coordinates = np.array([x,y,z])
coordinates = np.resize(coordinates,[np.int(len(x)),3])

####################
# CALCULATE RADIUS # 
####################

# get box dimensions 
x_box = []
y_box = []
z_box = []
for i in range(len(box_size)):
    x_box.append(box_size[i][0])
    y_box.append(box_size[i][1])
    z_box.append(box_size[i][2])

# get the pairs 
inner_first_bead, inner_second_bead = get_inner_pairs(n_rings,ring_beads)
outer_first_bead, outer_second_bead = get_outer_pairs(n_rings,ring_beads)

# get radii 
bead_size = 0.34
inner_radius = get_inner_radius(np.array(x),np.array(y),np.array(z),first=inner_first_bead,
                                second=inner_second_bead,bead_size=bead_size,x_box=x_box,y_box=y_box,z_box=z_box)
outer_radius = get_outer_radius(np.array(x),np.array(y),np.array(z),first=outer_first_bead,
                                second=outer_second_bead,bead_size=bead_size,x_box=x_box,y_box=y_box,z_box=z_box)


avg_inner_radius = np.sum(inner_radius)/len(inner_radius)
avg_outer_radius = np.sum(outer_radius)/len(outer_radius)
print(avg_inner_radius)
print(avg_outer_radius)

######################
# CALCULATE CONSTANT #
######################

# get pairs 
axial_first_bead = [20,15,10,5,30,25,75,70,65,90,85,80,130,125,150,145,135,185,210,205,200,195,190,290,295,
                    300,275,280,285,235,240,215,220,225,230,180,155,160,165,170,175,95,100,105,110,115,120]
axial_second_bead = [75,70,65,90,85,80,130,125,150,145,135,185,210,205,200,195,190,270,265,260,255,250,245,
                     235,240,215,220,225,230,180,155,160,165,170,175,95,100,105,110,115,120,40,45,50,55,60,35]
assert len(axial_first_bead)==len(axial_second_bead)

# get the axial rise 
axial_rise = get_axial_rise(np.array(x), np.array(y), np.array(z), first=axial_first_bead, 
                            second=axial_second_bead,x_box=x_box,y_box=y_box,z_box=z_box)

# get the constant for the curvature
constant = []
for i in range(len(axial_rise)): 
    constant.append(float(axial_rise[i])*(1/(2*m.pi)))

avg_axial_rise = np.sum(axial_rise)/len(axial_rise)
print(avg_axial_rise)

#######################
# CALCULATE CURVATURE #
#######################

curvature = []
for i in range(len(constant)): 
    curvature.append(outer_radius[i]/((outer_radius[i]**2)+(constant[i]**2)))

avg_curvature = np.sum(curvature)/len(curvature)
print(avg_curvature)

#####################
# CALCULATE TORSION #
#####################

torsion = []
for i in range(len(constant)): 
    torsion.append(constant[i]/((outer_radius[i]**2)+(constant[i]**2)))

avg_torsion = np.sum(torsion)/len(torsion)
print(avg_torsion)







