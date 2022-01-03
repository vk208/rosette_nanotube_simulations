
##########
# CREDIT #
##########

# Written by: Vyshnavi Karra
# Edited by: Vyshnavi Karra 
# University: Northeastern University 
# Advisor: Francisco Hung 

# Last Updated: 01-03-2022 

# Notes: 
# 1) This code takes the OPLS all atom gro file as an input and converts it to the MARTINI coarse grained model. There are 2 classes: atom and bead. The atom class will go through the OPLS gro file. The bead class will take the atom class outputs and use them for the tiny beads parameters.
# 2) This is based on/similar to the martinize.py code! (Source: https://github.com/cgmartini/martinize.py)
# 3) The tiny bead version is the initial hypothesis of the CG structure of the G^C motif of the rosette nanotube. Therefore, that is what the majority of the simulations are based on.
# 4) However, the TP1 bead does *NOT* include the CH3 side chain in addition to the standard TP1 bead. 
# 5) So future simulations may need some tweaking of this CG model where the TP1 bead is the standard MARTINI P1 bead to include the CH3 side chain or longer side chains. 

# For any questions, please create an issue on GitHub

#############
# LIBRARIES #
#############

import numpy as np 
import math 
import scipy as sp
import glob 
import re
from collections import defaultdict

###########
# CLASSES #
###########

class Atom():
    
    masses = {"O":15.99, 
          "C":12.01,
          "N":14.01,
          "H":1.01
        }
    
    def __init__(self,label=None,symbol=None,index=None,position=None):
        
        
        assert symbol.upper() in ('C','H','O',"N")
        self.symbol = symbol.upper()
        self.index = int(index)
        self.position = position
        self.mass = masses[self.symbol]
        self.label = label
        
    def __repr__(self):
        
        return "<Atom {}_{}>".format(self.label,self.index)
    
    def __getitem__(self, i):
        return self.index[i]
    

class Bead():
    
    def __init__(self,atoms=None,label=None,index=None, res=None):
        for atom in atoms:
            assert isinstance(atom,Atom)
        self.atoms = atoms #atoms 
        self.label = label #bead type, based on atoms present 
        self.index = int(index) #atom number
        self.com = self.get_com() #get the center of mass coordinates 
        self.radius = 0.16 #tiny bead parameter in MARTINI, in nm 
        self.res = res #residue name
        self.charges = self.get_charges() #get the partial charges 
        
    def __repr__(self):
        
        label = "<Bead {}".format(self.label)
        for atom in self.atoms:
            label += " " + atom.label
        label += ">"
        return label
    
    def __getitem__(self, i):
        return self.index[i]
        
    def get_com(self):
        """
        This will get the Center Of Mass (com) for the OPLS atoms to lump together into a MARTINI bead. 
        """
        #3 decimal places is enough for Gromacs 
        %precision 3
        
        # get the center of mass coordinates of each atom 
        sum_of_masses = sum([atom.mass for atom in self.atoms])
        xyz = [0,0,0]
        for i in (0,1,2):
            xyz[i] = sum([atom.mass*atom.position[i] for atom in self.atoms])/sum_of_masses
        return xyz
    
    def get_charges(self):
        """
        This will set the charges for the MARTINI CG beads, specifically TP1 and TP2. 
        """
        #3 decimal places is enough for Gromacs 
        %precision 3 
        
        # THIS WON'T CHANGE! 
        n_motifs = 6 #the number of motifs in a rosette ring is 6, for the stacked ring conformation
        
        # THIS MAY CHANGE!
        n_rings = 2 #the number of rings in the nanotube 
        
        # THIS MAY CHANGE!
        # for partial charges
        # IF they're the same values with different signs
        qTP = 0.35 
        if qTP != []:
            if qTP != 0: 
                motif_charges = [(0),(1*qTP),(-1*qTP),(1*qTP),(-1*qTP)]
            else: 
                motif_charges = [(qTP),(qTP),(qTP),(qTP),(qTP)]
        # ELSE they're different values 
        else: 
            qTP1 = 0 
            qTP2 = 0
            if qTP1 != 0:
                if qTP2 != 0: 
                    motif_charges = [(0),(1*qTP2),(-1*qTP1),(1*qTP2),(-1*qTP1)]
        
        charges = []
        if motif_charges != []:
            for i in range(n_motifs):
                for j in range(n_rings):
                    charges.append(float(motif_charges[0]))
                    charges.append(float(motif_charges[1]))
                    charges.append(float(motif_charges[2]))
                    charges.append(float(motif_charges[3]))
                    charges.append(float(motif_charges[4]))
        return charges
    
    
##########################
# TINY BEADS INFORMATION #
##########################

# Remember, tiny beads are smaller than the standard DNA beads to account for the DNA hydrogen bonding. 

# Tip: Go through your OPLS gro file, and number your atoms to create differences between the different
# atoms of the same element. 

Bead_info = {"bead1": ("C10 N2 C23"), 
             "bead2": ("O16 C11 N3"), 
             "bead3": ("C12 N5 C9"),
             "bead4": ("O1 C8 N4"),
             "bead5": ("N6 C7 N17")
            }


############################### 
# OPEN OPLS ALL ATOM GRO FILE #
############################### 

# path to your file 
path = ''

# clean up data 
digit_search = re.compile('\d+')
symbol_search =  re.compile('[a-zA-Z]+')

# open up file 
lines = open(path,'r').readlines()

# initialize
atoms = []

#set masses 
masses = {"O":15.99, 
          "C":12.01,
          "N":14.01,
          "H":1.01
        }
    
# read the file in your path   
for line in lines:
    _,label,index,x,y,z = line.split()
    symbol = re.search(symbol_search,label).group()
    atom = Atom(label=label,symbol=symbol,index=int(index),position=(float(x),float(y),float(z)))
    atoms.append(atom)
    
###########################
# CREATE ATOMS DICTIONARY # 
###########################

atomtypes = defaultdict(list)
for atom in atoms:
    atomtypes[atom.label].append(atom)

#####################
# TINY BEADS SET UP #
#####################

# set up empty lists 
bead1 = []
bead2 = []
bead3 = []
bead4 = []
bead5 = []

# counter 
counter = 4

# number of MARTINI CG beads in 1 G^C motif in this model is 5 
n_beads = 5 

# the number of G^C motifs in a rosette ring is always 6 for the stacked ring conformation 
n_motifs = 6 

# change this when there's more than 2 rings in the nanotube 
n_rings = 10

# tiny bead 1 = TN0 
for i,b in enumerate(zip(atomtypes["C10"],atomtypes["N2"],atomtypes["C23"])):
    if i !=0: 
        j = i+counter*i+1
        bead1.append(Bead(atoms=b,label='TN0',index=j, res="CSC"))
    else: 
        j = i+1 
        bead1.append(Bead(atoms=b,label='TN0',index=j, res="CSC"))

# tiny bead 2 = TP2 
for i,b in enumerate(zip(atomtypes["O16"],atomtypes["C11"],atomtypes["N3"])):
    if i !=0: 
        j = i+counter*i+2
        bead2.append(Bead(atoms=b,label='TP2',index=j, res="CSC"))
    else: 
        j = i+2 
        bead2.append(Bead(atoms=b,label='TP2', index=j, res="CSC"))
        
# tiny bead 3 = TP1 
for i,b in enumerate(zip(atomtypes["C12"],atomtypes["N5"],atomtypes["C9"])):
    if i !=0: 
        j = i+counter*i+3
        bead3.append(Bead(atoms=b,label='TP1',index=j, res="CSC"))
    else: 
        j = i+3
        bead3.append(Bead(atoms=b,label='TP1',index=j, res="CSC"))
        
# tiny bead 4 = TP2
for i,b in enumerate(zip(atomtypes["O1"],atomtypes["C8"],atomtypes["N4"])):
    if i !=0: 
        j = i+ counter*i + 4
        bead4.append(Bead(atoms=b,label='TP2',index=j, res="GSC"))
    else: 
        j = i+4
        bead4.append(Bead(atoms=b,label='TP2',index=j, res="GSC"))
        
# tiny bead 5 = TP1 
for i,b in enumerate(zip(atomtypes["N6"],atomtypes["C7"],atomtypes["N17"])):
    if i !=0: 
        j = i+counter*i+5
        bead5.append(Bead(atoms=b,label='TP1',index=j, res="GSC"))
    else: 
        j = i+5 
        bead5.append(Bead(atoms=b,label='TP1',index=j, res="GSC"))

# zip up all of the beads into 1 list 
rosette = list(zip(bead1,bead2,bead3,bead4,bead5))

# get the partial charges into a list 
charges = list()
for i in range(len(bead1)):
    charges.append(rosette[i][0].charges)

# make the partial charges list into an array 
charges = np.array(charges)

# get all of the center of mass coordinates into a list 
com = list()
for i in range(len(rosette)):
    for j in range(5): 
        com.append(rosette[i][j].com)

# make the center of mass coordinates list into an array 
com = np.array(com)

###############
# FIRST CHECK # 
###############

assert len(com)==(n_motifs*n_rings*5), "Something is wrong with your code! - Vyshnavi Karra"

#######################
# GET DUMMY BEAD INFO # 
#######################

# 1 "ring" is 6 motifs with 5 beads each. 
# This will pair bead i with bead j opposite it on the other side of the ring. 
pairs = [(1,16),(2,17),(3,18),(4,19),(5,20),(6,21),(7,22),(8,23),
         (9,24),(10,25),(11,26),(12,27),(13,28),(14,29),(15,30)]

#check the coordinates 
motif_pairs = []
counter = 30
for k in range(n_rings): 
    for j in range(len(pairs)): 
        motif_pairs.append((pairs[j][0]+30*k, pairs[j][1]+30*k))
        
####################################
# DUMMY BEAD FORCEFIELD PARAMETERS # 
####################################

all_beads = ["P2","P3","AC2","P1","P4","P5","Nda","Na","Qda","Nd","Q0","Qa","C3","AC1","Qd","N0","C2","C1",
             "C5","C4","SNd","SQ0","SNda","SNa","SP1","SP2","SP3","SP4","SP5","SQda","SN0","SC5","SC4","SQa",
             "SQd","SC1","SC3","SC2","TQda","TY2","TY3","TQd","TN0","TP5","TP4","TP3","TP2","TP1","TNda","TQ0",
             "TG2","TG3","TA2","TA3","TT3","TT2","TC4","TC5","TC2","TC3","TC1","TQa","TNd","TNa","vP5","vP4",
             "vNd","vP1","vNda","vP3","vP2","vAC2","vQ0","vAC1","vQda","vC1","vC2","vC3","vC4","vC5","vN0",
             "vQa","vNa","vQd","vSC4","vSC5","vSQda","vSN0","vSC1","vSC2","vSC3","vSQd","vSQ0","vSNa","vSQa",
             "vSNd","vSP5","vSP4","vSP1","vSNda","vSP3","vSP2","POL","D","VK"]

nonbonded_params = []
for i in range(len(all_beads)): 
    if all_beads[i] != "VK": 
        nonbonded_params.append(["VK", all_beads[i], 1, "4.700000e-01", "0.000000e+00"])
    else: 
        nonbonded_params.append(["VK", all_beads[i], 1, "1.000000e+00", "0.000000e+00"])


##########################
# DUMMY BEAD COORDINATES # 
##########################

# have to automate this somehow... 
dummy_coords = []

####################
# DUMMY BEAD BONDS # 
####################

distance = []
midpoint = []
dummy_coords = []
for i in range(len(motif_pairs)): 
    distance.append([motif_pairs[i][0], motif_pairs[i][1], ((com[motif_pairs[i][1]-1][0]-com[motif_pairs[i][0]-1][0])**2 + 
                (com[motif_pairs[i][1]-1][1]-com[motif_pairs[i][0]-1][1])**2 + 
                (com[motif_pairs[i][1]-1][2]-com[motif_pairs[i][0]-1][2])**2)**0.5])
    midpoint.append([(com[motif_pairs[i][1]-1][0]+com[motif_pairs[i][0]-1][0])/2, 
                (com[motif_pairs[i][1]-1][1]+com[motif_pairs[i][0]-1][1])/2, 
                (com[motif_pairs[i][1]-1][2]+com[motif_pairs[i][0]-1][2])/2])
    if motif_pairs[i][1] % 20 == 0: 
        dummy_coords.append([midpoint[i][0], midpoint[i][1], midpoint[i][2]])

dummy_bond = []
counter = 1
for i in range(len(distance)): 
    dummy_bond.append([distance[i][0],counter+(n_motifs*n_beads*n_rings),distance[i][-1]/2])
    dummy_bond.append([distance[i][1],counter+(n_motifs*n_beads*n_rings),distance[i][-1]/2]) 
    if distance[i][1] % 30 == 0: 
        counter = counter + 1

###################
# TOPOLOGY SET UP # 
###################

### for PMFs of rings, add harmonic bonds for h-bonds and improper dihedrals for planarity
hbond_bonds = [(1,10),(2,9),(6,15),(7,14),(11,20),(12,19),(16,25),(17,24),(21,30),(22,29),(26,5),(27,4)]
self_dummy_bonds = [(301,302),(302,303),(303,304),(304,305),(305,306),(306,307),(307,308),(308,309),(309,310)]
hbond_impropers = [(1,2,9,10),(6,7,14,15),(11,12,19,20),(16,17,24,25),(21,22,29,30),(26,27,4,5)]
total_hbonds = len(hbond_bonds)*n_rings

### if no changes to h-bonds and impropers, uncomment these and comment out the ones above 
#hbond_bonds = []
#hbond_impropers = []



# Remember, these are the bonded parameters of the G^C motif of the current CG model. 

motif_bonds = [(1,2),(2,3),(3,1),(3,4),(4,5),(5,1)]
motif_angles = [(1,2,3),(2,3,1),(2,1,3),(1,3,4),(3,4,5),(4,5,1),(5,1,3)]
motif_dihedrals = []
motif_impropers = [(1,2,3,4),(3,4,5,1),(2,1,5,4)]

# for reshaping the arrays for writing the itp files 
total_bonds = (n_rings*n_motifs*len(motif_bonds))
total_angles = (n_rings*n_motifs*len(motif_angles))
total_dihedrals = (n_rings*n_motifs*len(motif_dihedrals))
total_impropers = (n_rings*n_motifs*len(motif_impropers)+(n_rings*len(hbond_impropers)))


#########
# BONDS # 
#########

bonds = []
if motif_bonds != []: 
    counter = 5
    for i in range(len(motif_bonds)):
        for j in range(n_motifs*n_rings): 
            if j !=0: 
                bonds.append(int(motif_bonds[i][0]+(j*counter)))
                bonds.append(int(motif_bonds[i][1]+(j*counter)))
            else: 
                bonds.append(int(motif_bonds[i][0]))
                bonds.append(int(motif_bonds[i][1]))
    bonds = np.reshape(np.array(bonds),(int(total_bonds),2))
    print("Got all bond connectivities.")
else: 
    print("No bonds described for the G^C motif. Please make sure this is correct! - Vyshnavi Karra")

##########
# HBONDS # 
##########
                
hbonds = []                
if hbond_bonds != []: 
    hb_counter = 30
    for i in range(len(hbond_bonds)):
        for j in range(n_rings):
            if j!=0: 
                hbonds.append(int(hbond_bonds[i][0]+(j*hb_counter)))
                hbonds.append(int(hbond_bonds[i][1]+(j*hb_counter))) 
            else: 
                hbonds.append(int(hbond_bonds[i][0]))
                hbonds.append(int(hbond_bonds[i][1]))
                
    hbonds = np.reshape(np.array(hbonds),(int(total_hbonds),2))
    print("Got all hydrogen bond connectivities.")
else: 
    print("No explicit hydrogen bond described for the G^C motif. Please make sure this is correct! - Vyshnavi Karra")

###############
# CONSTRAINTS # 
###############

constraints = []
if motif_bonds != []: 
    counter = 5
    hb_counter = 3 
    for i in range(len(motif_bonds)):
        for j in range(n_motifs*n_rings): 
            if j !=0: 
                constraints.append(int(motif_bonds[i][0]+(j*counter)))
                constraints.append(int(motif_bonds[i][1]+(j*counter)))
            else: 
                constraints.append(int(motif_bonds[i][0]))
                constraints.append(int(motif_bonds[i][1]))
    constraints = np.reshape(np.array(constraints),(int(total_bonds),2))
    print("Got all constraints.")
else: 
    print("No constraints described for the G^C motif. Please make sure this is correct! - Vyshnavi Karra")



##########
# ANGLES # 
##########
    
angles = []
if motif_angles != []: 
    counter = 5
    for i in range(len(motif_angles)): 
        for j in range(n_motifs*n_rings): 
            if j != 0: 
                angles.append(motif_angles[i][0]+(j*counter))
                angles.append(motif_angles[i][1]+(j*counter))
                angles.append(motif_angles[i][2]+(j*counter))
            else: 
                angles.append(motif_angles[i][0])
                angles.append(motif_angles[i][1])
                angles.append(motif_angles[i][2])
                
    angles = np.reshape(np.array(angles),(int(total_angles),3))
    print("Got all angle connectivities.")
else: 
    print("No angles described for G^C motif. Please make sure this is correct! - Vyshnavi Karra ")

#############
# DIHEDRALS # 
#############

dihedrals = []
if motif_dihedrals != []: 
    counter = 5
    for i in range(len(motif_dihedrals)): 
        for j in range(n_motifs*n_rings): 
            if j != 0: 
                dihedrals.append(motif_dihedrals[i][0]+(j*counter))
                dihedrals.append(motif_dihedrals[i][1]+(j*counter))
                dihedrals.append(motif_dihedrals[i][2]+(j*counter))
                dihedrals.append(motif_dihedrals[i][3]+(j*counter))
            else: 
                dihedrals.append(motif_dihedrals[i][0])
                dihedrals.append(motif_dihedrals[i][1])
                dihedrals.append(motif_dihedrals[i][2])
                dihedrals.append(motif_dihedrals[i][3])
                
    #dihedrals = np.reshape(np.array(dihedrals),(int(total_dihedrals/4),4))
    print("Got all dihedral connectivities.")
else: 
    print("No dihedrals described for G^C motif. Please make sure this is correct! - Vyshnavi Karra")

#############
# IMPROPERS # 
#############   
    
impropers = []
if motif_impropers != []: 
    counter = 5
    hb_counter = 3
    for i in range(len(motif_impropers)): 
        for j in range(n_motifs*n_rings):
            if j != 0: 
                impropers.append(motif_impropers[i][0]+(j*counter))
                impropers.append(motif_impropers[i][1]+(j*counter))
                impropers.append(motif_impropers[i][2]+(j*counter))
                impropers.append(motif_impropers[i][3]+(j*counter))
            else: 
                impropers.append(motif_impropers[i][0])
                impropers.append(motif_impropers[i][1])
                impropers.append(motif_impropers[i][2])
                impropers.append(motif_impropers[i][3])
    for i in range(len(hbond_impropers)):
        for j in range(n_rings):
            if j!=0: 
                impropers.append(int(hbond_impropers[i][0]+(j*hb_counter)))
                impropers.append(int(hbond_impropers[i][1]+(j*hb_counter))) 
                impropers.append(int(hbond_impropers[i][2]+(j*hb_counter)))
                impropers.append(int(hbond_impropers[i][3]+(j*hb_counter)))
            else: 
                impropers.append(int(hbond_impropers[i][0]))
                impropers.append(int(hbond_impropers[i][1]))
                impropers.append(int(hbond_impropers[i][2]))
                impropers.append(int(hbond_impropers[i][3]))
                
    impropers = np.reshape(np.array(impropers),(int(total_impropers),4))
    print("Got all improper connectivities.")
else: 
    print("No dihedrals described for G^C motif. Please make sure this is correct! - Vyshnavi Karra ")

####################
# GET BOND LENGTHS # 
####################

# set up 
p1 = list()
p2 = list()

# get the indexes's com
for i in range(len(bonds)):
    for index,value in enumerate(com): 
        if (bonds[i][0]-1) == index: 
            p1.append(com[index])
        elif (bonds[i][1]-1) == index: 
            p2.append(com[index])
            
# calculate the bond length based on the com's 
bond_length = list()
for i in range(len(p1)):       
    bond_length.append(np.sqrt(((p1[i][0]-p2[i][0])**2)+((p1[i][1]-p2[i][1])**2)+((p1[i][2]-p2[i][2])**2)))
    
    
####################
# GET HBOND LENGTHS # 
####################

# set up 
h1 = list()
h2 = list()

# get the indexes's com
for i in range(len(hbonds)):
    for index,value in enumerate(com): 
        if (hbonds[i][0]-1) == index: 
            h1.append(com[index])
        elif (hbonds[i][1]-1) == index: 
            h2.append(com[index])
            
# calculate the bond length based on the com's 
hbond_length = list()
for i in range(len(h1)):       
    hbond_length.append(np.sqrt(((h1[i][0]-h2[i][0])**2)+((h1[i][1]-h2[i][1])**2)+((h1[i][2]-h2[i][2])**2)))

##########################
# GET CONSTRAINT LENGTHS # 
##########################

constraint_length = []

    
####################
# GET ANGLE THETAS # 
####################    

# set up 
p1 = list()
p2 = list()
p3 = list()

# get the indexes's com 
for i in range(len(angles)):
    for index,value in enumerate(com): 
        if (angles[i][0]-1) == index: 
            p1.append(com[index])
        elif (angles[i][1]-1) == index: 
            p2.append(com[index])
        elif (angles[i][2]-1) == index: 
            p3.append(com[index])
            
# calculate the theta's using the com 
p1 = np.array(p1)
p2 = np.array(p2)
p3 = np.array(p3)
theta = list()
for i in range(len(p1)):
    ba = p1[i]-p2[i]
    bc = p2[i]-p3[i]
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    theta.append(np.degrees(angle))
    

#####################
# GET DIHEDRAL PHIS # 
#####################  

# This model does not use dihedrals currently 

###################
# ELASTIC NETWORK # 
###################

# Please use gmx genrestr to make a distance restraint file first. 
# It is the basis for the elastic network in this model. 

# path to your file 
path = ''

# open up file 
lines = open(path,'r').readlines()

# read the file in your path   
disre = []
for line in lines:
    ai,aj,question,label,function,low,up1,up2,weight = line.split()
    disre.append([int(ai),int(aj),int(question),int(label),int(function),float(low),float(up1),float(up2),int(weight)])
    

################
# SECOND CHECK # 
################

if dummy_coords != []: 
    assert len(dummy_coords)==n_rings, "Mismatch between number of dummy beads and number of rings"
assert len(bonds)==len(bond_length), "Missing a bond length!"
assert len(angles)==len(theta), "Missing an angle theta!"
# assert len(dihedrals)==len(phi), "Missing a dihedral phi!"

######################
# WRITE THE GRO FILE # 
######################

filename = ''
with open(filename,"w") as gro: 
    gro.write("GCTP\n")
    gro.write("%d \n" % 310)                                                        # total number of residues 
    for i in range(len(rosette)): 
        for j in range(5):
            line = "    %5s%6s%5s%8.3f%8.3f%8.3f\n" % ("1GCTP",                     # residue name
                                                    rosette[i][j].res,              # bead type
                                                    rosette[i][j].index,            # atom number
                                                    float(rosette[i][j].com[0]),    # x coordinate
                                                    float(rosette[i][j].com[1]),    # y coordinate
                                                    float(rosette[i][j].com[2]))    # z coordinate
            gro.write(line)
    for i in range(len(dummy_coords)):
        dummyline = "    %5s%6s%5s%8.3f%8.3f%8.3f\n" % ("1GCTP",                    # residue name
                                                        "VK",                       # bead type 
                                                        len(rosette)*5+i+1,         # atom number
                                                        float(dummy_coords[i][0]),  # x coordinate
                                                        float(dummy_coords[i][1]),  # y coordinate
                                                        float(dummy_coords[i][2]))  # z coordinate 
        gro.write(dummyline)
    gro.write("10.0000 10.000 10.000\n")                                            # box dimensions
    
    gro.close()
        
######################
# WRITE THE ITP FILE # 
######################

filename = ''
with open(filename,"w") as itp: 
    itp.write(";GCTP\n\n")
    itp.write("[ moleculetype ]\n")
    itp.write("GCTP     3 \n\n")
    itp.write("[ atoms ]\n")
    for i in range(len(rosette)): 
        for j in range(5): 
            line = "%5d%5s%5d%8s%5s%5d%8.3f\n" % (rosette[i][j].index,           # atom number 
                                                   rosette[i][j].label,          # bead type
                                                   rosette[i][j].index,          # basically atom number again 
                                                   "GCTP",                       # residue name 
                                                   rosette[i][j].res,            # residue group name 
                                                   rosette[i][j].index,          # charge group number 
                                                   charges[0][j])                # charges 
            itp.write(line)
    for i in range(len(dummy_coords)):
        dummyline = "%5d%5s%5d%8s%5s%5d%8.3f\n" % (len(rosette)*5+i+1,           # atom number 
                                                   "VK",                         # bead type 
                                                   len(rosette)*5+i+1,           # basically atom numbera again
                                                   "GCTP",                       # residue name 
                                                   "VK",                         # residue group name 
                                                   len(rosette)*5+i+1,           # charge group number 
                                                   0.0)                          # charges 
        itp.write(dummyline)
    
    itp.write("\n")
    itp.write("[ bonds ]\n")
    for i in range(len(bonds)): 
        line = "%5d%5d%5d%8.3f%6d\n"% (bonds[i][0],        # atom i
                                       bonds[i][1],        # atom j 
                                       1,                  # bond type (1 is harmonic chemical, 6 is harmonic not chemical, 7 is FENE)
                                       bond_length[i],     # bond length
                                       10000)              # force constant 
        itp.write(line)
    if dummy_bond !=[]: 
        itp.write(";dummy bond\n")
        for i in range(len(dummy_bond)): 
            line = "%5d%5d%5d%8.3f%8d\n"% (dummy_bond[i][0],    # atom i  
                                           dummy_bond[i][1],    # atom j
                                           6,                   # bond type (1 is harmonic chemical, 6 is harmonic not chemical, 7 is FENE)
                                           dummy_bond[i][-1],   # bond length 
                                           25000)              # force constant
            itp.write(line) 
            
    if self_dummy_bonds !=[]: 
        itp.write(";self dummy bonds\n")
        for i in range(len(self_dummy_bonds)): 
            line = "%5d%5d%5d%8.3f%8d\n"% (self_dummy_bonds[i][0],    # atom i  
                                           self_dummy_bonds[i][1],    # atom j
                                           6,                   # bond type (1 is harmonic chemical, 6 is harmonic not chemical, 7 is FENE)
                                           0.4,   # bond length 
                                           20000)              # force constant
            itp.write(line)   
        
    #if hbonds != []: 
    #    itp.write(";harmonic bond hydrogen bonds\n")
    #    for i in range(len(hbonds)): 
    #        line = "%5d%5d%5d%8.3f%8d\n"% (hbonds[i][0],        # atom i 
    #                                       hbonds[i][1],        # atom j 
    #                                       6,                   # bond type (1 is harmonic chemical, 6 is harmonic not chemical, 7 is FENE)
    #                                       hbond_length[i],     # bond length 
    #                                       100000)              # force constant 
    #        itp.write(line)
    #
    ##if disre != []:
    #    itp.write(";elastic network\n")
    #    itp.write("#ifdef RUBBER_BANDS\n")
    #    itp.write("#ifndef RUBBER_FC\n")
    #    itp.write("#define RUBBER_FC 500.000000\n")
    #    itp.write("#endif\n")
    #    for i in range(len(disre)): 
    #        line = "%5d%5d%5d%8.3f  %s\n"% (disre[i][0],           # atom i 
    #                                        disre[i][1],           # atom j 
    #                                        1,                     # bond type (1 is harmonic chemical, 6 is harmonic not chemical, 7 is FENE)
    #                                        disre[i][5],           # elastic network bond length 
    #                                        "RUBBER_FC*1.0000")    # elastic network force * the weight 
    #        itp.write(line)
    
    if constraints != []: 
        if constraint_length != []:
            itp.write("\n")
            itp.write("[ constraints ]\n")
            for i in range(len(constraints)): 
                line = "%5d%5d%5d%8.3f%6d\n"% (constraints[i][0],         # atom i 
                                               constraints[i][1],         # atom j 
                                               1,                         # bond type (1 is harmonic chemical, 6 is harmonic not chemical, 7 is FENE)
                                               constraint_length[i])      # constraint length 
                itp.write(line)
    
    itp.write("\n")
    itp.write("[ angles ]\n")
    for i in range(len(angles)): 
        line = "%5d%5d%5d%5d%8.1f%5d\n" % (angles[i][0],            # atom i 
                                           angles[i][1],            # atom j 
                                           angles[i][2],            # atom k 
                                           1,                       # angle type 
                                           theta[i],                # angle in degrees
                                           2500)                    # force constant
        itp.write(line)
     
    itp.write("\n")
    itp.write("[ dihedrals ]\n")
    if dihedrals != [] and phi != []: 
        for i in range(len(dihedrals)): 
            line = "%5d%5d%5d%5d%5d%8.1f%5d%5d\n" % (dihedrals[i][0],       # atom i 
                                                     dihedrals[i][1],       # atom j 
                                                     dihedrals[i][2],       # atom k 
                                                     dihedrals[i][3],       # atom l 
                                                     1,                     # proper dihedral type
                                                     phi[i],                # proper dihedral angle
                                                     600,                   # force constant
                                                     2)                     # multiplicity
            itp.write(line)
    
    if impropers != []: 
        for i in range(len(impropers)): 
            line = "%5d%5d%5d%5d%5d%8.1f%5d%5d\n" % (impropers[i][0],       # atom i 
                                                     impropers[i][1],       # atom j 
                                                     impropers[i][2],       # atom k 
                                                     impropers[i][3],       # atom l 
                                                     1,                     # improper dihedral type
                                                     180,                   # improper dihedral angle
                                                     600,                   # force constant
                                                     2)                     # multiplicity
            itp.write(line)
    itp.write("\n")
    itp.close()
    








