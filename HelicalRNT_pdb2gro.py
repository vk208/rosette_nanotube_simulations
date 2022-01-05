
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
# 1) This is to convert the helical RNT .pdb file to a .gro file 
# 2) This can be adopted to any .pdb file if the forcefield atom types are known. 

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

#############
# FUNCTIONS #
#############

def bonded_parameters(path): 
    bonds = []
    angles = []
    dihedrals = []

    files = glob.glob(path)
    for name in files:
        with open(name) as f:
            for line in f:
                if (not (line.lstrip().startswith('HEADER')) and not ((line.lstrip().startswith('TITLE'))) and not ((line.lstrip().startswith('AUTHOR')))):
                    cols = line.split() 
                    if (cols[0]=='CONECT' and len(cols)==3): 
                        bonds.append(int(cols[1]))
                        bonds.append(int(cols[2]))
                    if (cols[0]=='CONECT' and len(cols)==4): 
                        angles.append(int(cols[1]))
                        angles.append(int(cols[2]))
                        angles.append(int(cols[3]))
                    if (cols[0]=='CONECT' and len(cols)==5): 
                        dihedrals.append(int(cols[1]))
                        dihedrals.append(int(cols[2]))
                        dihedrals.append(int(cols[3]))
                        dihedrals.append(int(cols[4]))
    
    bonds = np.reshape(np.array(bonds),[int(len(bonds)/2),2])
    angles = np.reshape(np.array(angles),[int(len(angles)/3),3])
    dihedrals = np.reshape(np.array(dihedrals),[int(len(dihedrals)/4),4])

    return bonds, angles, dihedrals
     
def coordinates(path): 
    index = []
    atom = [] 
    residue = []
    xcoord = []
    ycoord = []
    zcoord = []
    
    files = glob.glob(path)
    for name in files:
        with open(name) as f:
            for line in f:
                if (not (line.lstrip().startswith('HEADER')) and not ((line.lstrip().startswith('TITLE'))) and not ((line.lstrip().startswith('AUTHOR')))):
                    cols = line.split() 
                    if (cols[0]=='HETATM' and len(cols)==11): 
                        index.append(int(cols[1]))
                        atom.append(cols[2])
                        residue.append(cols[3])
                        xcoord.append(float(cols[5]))
                        ycoord.append(float(cols[6]))
                        zcoord.append(float(cols[7]))
    
    return index, atom, residue, xcoord, ycoord, zcoord

def mass(atom): 
    masses = [] 
    
    for i in range(len(atom)): 
        if 'O' in atom[i]: 
            masses.append(15.994)
        elif 'C' in atom[i]:
            masses.append(12.0107)
        elif 'N' in atom[i]: 
            masses.append(14.0067)
        elif 'H' in atom[i]:
            masses.append(1.0079)
            
    return masses

def bond_length(xcoord,ycoord,zcoord,bonds): 
    bond_length = []
    
    for i in range(len(bonds)):
        if i==0:
            x = (xcoord[bonds[i][1]]/10-xcoord[bonds[i][0]]/10)**2
            y = (ycoord[bonds[i][1]]/10-ycoord[bonds[i][0]]/10)**2
            z = (zcoord[bonds[i][1]]/10-zcoord[bonds[i][0]]/10)**2
            distance = "%.3f" % np.sqrt((x + y + z))
            bond_length.append(float(distance))
        elif i>0: 
            x = (xcoord[bonds[i][1]-1]/10-xcoord[bonds[i][0]-1]/10)**2
            y = (ycoord[bonds[i][1]-1]/10-ycoord[bonds[i][0]-1]/10)**2
            z = (zcoord[bonds[i][1]-1]/10-zcoord[bonds[i][0]-1]/10)**2
            distance = "%.3f" % np.sqrt((x + y + z))
            bond_length.append(float(distance))
        elif bonds[i][1]==1560: 
            x = (xcoord[bonds[-1][1]-1]/10-xcoord[bonds[-1][0]-1]/10)**2
            y = (ycoord[bonds[-1][1]-1]/10-ycoord[bonds[-1][0]-1]/10)**2
            z = (zcoord[bonds[-1][1]-1]/10-zcoord[bonds[-1][0]-1]/10)**2
            distance = "%.3f" % np.sqrt((x + y + z))
            bond_length.append(float(distance))
    return bond_length

def theta(xcoord,ycoord,zcoord,angles): 
    theta = [] 
    
    for i in range(len(angles)): 
        if i==0: 
            A = np.array([xcoord[angles[i][0]]/10,ycoord[angles[i][0]]/10,zcoord[angles[i][0]]/10])
            B = np.array([xcoord[angles[i][1]]/10,ycoord[angles[i][1]]/10,zcoord[angles[i][1]]/10])
            C = np.array([xcoord[angles[i][2]]/10,ycoord[angles[i][2]]/10,zcoord[angles[i][2]]/10])
            ba = A - B
            bc = C - B
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.degrees(np.arccos(cosine_angle))
            if angle>90: 
                theta.append(angle)
            else: 
                theta.append(angle+90)
        elif i>0: 
            A = np.array([xcoord[angles[i][0]-1]/10,ycoord[angles[i][0]-1]/10,zcoord[angles[i][0]-1]/10])
            B = np.array([xcoord[angles[i][1]-1]/10,ycoord[angles[i][1]-1]/10,zcoord[angles[i][1]-1]/10])
            C = np.array([xcoord[angles[i][2]-1]/10,ycoord[angles[i][2]-1]/10,zcoord[angles[i][2]-1]/10])
            ba = A - B
            bc = C - B
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.degrees(np.arccos(cosine_angle))
            if angle>90: 
                theta.append(angle)
            else: 
                theta.append(angle+90)
                
    return theta

def calculate_phi(dihedrals):
    phi = []
    
    for i in range(len(dihedrals)):
        
        # get atoms and their coordinates 
        a0 = np.array([xcoord[dihedrals[i][0]], ycoord[dihedrals[i][0]], zcoord[dihedrals[i][0]]])
        a1 = np.array([xcoord[dihedrals[i][1]], ycoord[dihedrals[i][1]], zcoord[dihedrals[i][1]]])
        a2 = np.array([xcoord[dihedrals[i][2]], ycoord[dihedrals[i][2]], zcoord[dihedrals[i][2]]])
        a3 = np.array([xcoord[dihedrals[i][3]], ycoord[dihedrals[i][3]], zcoord[dihedrals[i][3]]])
        
        # get vectors 
        v0 = -1.0*(a1 - a0)
        v1 = a2 - a1
        v2 = a3 - a2
        
        # normalize 
        v1 /= np.linalg.norm(v1)
        
        # vector rejections
        # U = projection of v0 onto plane perpendicular to v1
        #   = v0 minus component that aligns with v1
        # W = projection of v2 onto plane perpendicular to v1
        #   = v2 minus component that aligns with v1
        U = v0 - np.dot(v0, v1)*v1
        W = v2 - np.dot(v2, v1)*v1
        
        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x = np.dot(U, W)
        y = np.dot(np.cross(v1, U), W)
        
        phi.append(np.degrees(np.arctan2(y, x)))
    
    return phi

def atomtype(atom): 
    atomtypes = [] 
    
    i = 0
    for j in range(60): 
        atomtypes.append('opls_233')
        atomtypes.append('opls_333')
        atomtypes.append('opls_335')
        atomtypes.append('opls_238')
        atomtypes.append('opls_341')
        atomtypes.append('opls_750')
        atomtypes.append('opls_752')
        atomtypes.append('opls_235')
        atomtypes.append('opls_337')
        atomtypes.append('opls_338')
        atomtypes.append('opls_334')
        atomtypes.append('opls_336')
        atomtypes.append('opls_340')
        atomtypes.append('opls_303')
        atomtypes.append('opls_748')
        atomtypes.append('opls_135')
        
        i = i + 16
        
    for j in range(60):
        atomtypes.append('opls_240')
        atomtypes.append('opls_240')
        atomtypes.append('opls_240')
        atomtypes.append('opls_240')
        atomtypes.append('opls_140')
        atomtypes.append('opls_140')
        atomtypes.append('opls_140')
        atomtypes.append('opls_140')
        atomtypes.append('opls_140')
        atomtypes.append('opls_140')
        
        i = i + 10
            
        if i>1561:
            break;

    return atomtypes 

def charge(atom): 
    charges = [] 
    
    i = 0
    for j in range(60): 
        charges.append(-0.5)
        charges.append(-0.3025)
        charges.append(-0.54)
        charges.append(-0.4225)
        charges.append(-0.79)
        charges.append(-0.4815)
        charges.append(0.4565)
        charges.append(0.5)
        charges.append(0.04)
        charges.append(0.3225)
        charges.append(0.55)
        charges.append(0.46)
        charges.append(-0.48)
        charges.append(-0.604)
        charges.append(0.039)
        charges.append(-0.0575)
        
        i = i + 16
        
    for j in range(60):
        charges.append(0.3)
        charges.append(0.37)
        charges.append(0.37)
        charges.append(0.41)
        charges.append(0.06)
        charges.append(0.06)
        charges.append(0.06)
        charges.append(0.06)
        charges.append(0.06)
        charges.append(0.06)
        
        i = i + 10
            
        if i>1561:
            break;

    return charges 

def description(atom): 
    info = [] 
    
    i = 0
    for j in range(60): 
        info.append(';O: C=O in amide')
        info.append(';Cytosine N1 ')
        info.append(';Cytosine N3')
        info.append(';N: general amide overlapping with guanidinium	')
        info.append(';Cytosine N-C4	')
        info.append(';N1 of neutral ARG (HN=CZ)	')
        info.append(';CZ of neutral ARG	')
        info.append(';C: C=O in amide or unsaturated amide 	')
        info.append(';Cytosine C5')
        info.append(';Cytosine C6')
        info.append(';Cytosine C2')
        info.append(';Cytosine C4')
        info.append(';Cytosine O-C2	')
        info.append(';N: guanidinium NHR (changed from type 6002 from Arthur MAE file) ')
        info.append(';CD neutral guanidine/ARG')
        info.append(';C: alkanes ')
        
        i = i + 16
        
    for j in range(60):
        info.append(';H on neutral N ')
        info.append(';H on neutral N ')
        info.append(';H on neutral N ')
        info.append(';H on neutral N ')
        info.append(';H: alkanes')
        info.append(';H: alkanes')
        info.append(';H: alkanes')
        info.append(';H: alkanes')
        info.append(';H: alkanes')
        info.append(';H: alkanes')
        
        i = i + 10
            
        if i>1561:
            break;

    return info 

                             
# LHT RNT pdb file 
path = '***.pdb'

bonds, angles, dihedrals = bonded_parameters(path)
index, atom, residue, xcoord, ycoord, zcoord = coordinates(path)
bond_length = bond_length(xcoord,ycoord,zcoord,bonds)
theta = theta(xcoord,ycoord,zcoord,angles)
phi =  calculate_phi(dihedrals)


masses = mass(atom)
atomtypes = atomtype(atom)
charges = charge(atom)
info = description(atom)

##############
# CHECKPOINT #
##############

assert len(masses)==len(atom), 'Missing either a mass or atom!'
assert len(atomtypes)==len(atom), 'Missing either an atom type or atom!'
assert len(charges)==len(atom), 'Missing either an atom charge or atom!'
assert float('{:4f}'.format(float(sum(charges))))==0, 'Missing a charge, not a neutral molecule!'
assert len(info)==len(atom), 'Missing either an atom description or atom!'
assert len(dihedrals)==len(phi), 'Missing either a dihedral set of 4 atoms or the angle phi'

######################
# WRITE THE ITP FILE # 
######################

filename = ''
with open(filename,"w") as itp: 
    itp.write(";GCTP\n\n")
    itp.write("[ moleculetype ]\n")
    itp.write("RNT     3 \n\n")
    itp.write("[ atoms ]\n")
    for i in range(len(xcoord)): 
        line = "%5d %5s%5d%8s %5s%5d%8.4f%8.4f   %15s\n" % (i+1,           # atom number 
                                                   atomtypes[i],          # bead type
                                                   1,          # basically atom number again 
                                                   "RNT",                       # residue name 
                                                   atom[i],            # residue group name 
                                                   i+1,          # charge group number 
                                                   charges[i],               # charges 
                                                   masses[i],        #masses
                                                   info[i])
        itp.write(line)
    
    itp.write("\n")
    itp.write("[ bonds ]\n")
    for i in range(len(bonds)-1): 
        line = "%5d%5d%5d%8.3f%6d\n"% (bonds[i][0],        # atom i
                                       bonds[i][1],        # atom j 
                                       1,                  # bond type (1 is harmonic chemical, 6 is harmonic not chemical, 7 is FENE)
                                       bond_length[i],     # bond length
                                       5000)              # force constant 
        itp.write(line)
    
    itp.write("\n")
    itp.write("[ angles ]\n")
    for i in range(len(angles)-1): 
        line = "%5d%5d%5d%5d%8.3f%5d\n" % (angles[i][0],            # atom i 
                                           angles[i][1],            # atom j 
                                           angles[i][2],            # atom k 
                                           1,                       # angle type 
                                           theta[i],                # angle in degrees
                                           2500)                    # force constant
        itp.write(line)
     
    itp.write("\n")
    itp.write("[ dihedrals ]\n")
    if dihedrals != []: 
        for i in range(len(dihedrals)-1): 
            line = "%5d%5d%5d%5d%5d %8.3f%5d%5d\n" % (dihedrals[i][0],       # atom i 
                                                     dihedrals[i][1],       # atom j 
                                                     dihedrals[i][2],       # atom k 
                                                     dihedrals[i][3],       # atom l 
                                                     1,                     # proper dihedral type
                                                     phi[i],                 # proper dihedral angle
                                                     600,                   # force constant
                                                     2)                     # multiplicity
            itp.write(line)

    itp.write("\n")
    itp.close()



