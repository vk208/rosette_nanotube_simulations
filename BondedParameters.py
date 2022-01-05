
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
# 1) For each set of data files (AA, CG systems), the bonded distributions would need to be done prior to this. 
# 2) Each system's .xvg files would have to be individually called using the functions. 
# 3) Each system's histograms need to be specified in the seaborn plotting section. 

##########
# SET UP #
##########

#set up notebook 
import numpy as np 
import scipy as sp 
import math as m 
import matplotlib.pyplot as plt
import glob 
import matplotlib.patches as mpatches
import os
import seaborn as sns
sns.set_style("white")

#time scale  
timestep = 0.0001 
num_steps = 20000000
sampling_freq = 100 
picoseconds = timestep*num_steps 
sampling_tot = num_steps/sampling_freq
time = np.linspace(0,np.int(picoseconds),np.int(sampling_tot+1))

#############
# FUNCTIONS #
#############

def get_bonds(path): 
    bonds = []
    
    files = glob.glob(path)
    for name in files:
        with open(name) as f:
            for line in f:
                if (not (line.lstrip().startswith('#')) and not ((line.lstrip().startswith('@')))):
                    cols = line.split() 
                    if len(cols)==2: 
                        bonds.append(float(cols[1]))
            
    #resize bonds array 
    bonds = np.resize(bonds,[np.int(len(files)),np.int((len(bonds)/len(files)))])
    overall_avg = np.mean(bonds, axis=1)
    
    #truncate to 3 floats 
    for i in range(len(overall_avg)): 
        overall_avg[i] = "%.3f" % overall_avg[i]
    
    n, bins = np.histogram(overall_avg, bins='auto')
    mids = 0.5*(bins[1:] + bins[:-1])
    mean = np.average(mids, weights=n)
    var = np.average((mids - mean)**2, weights=n)
    std = np.sqrt(var)
    
    return bonds, n, bins, mids, mean, var, std 


def get_angles(path):
    afiles2 = [f for f in os.listdir(path) if f.endswith('.xvg')]
    apath2 = path
    agroups_dict = dict()
    for file in afiles2:
        aname = file.split('.')[0]
        with open(apath2 + str(file)) as f:
            angles = list()
            freqs = list()
            for line in f:
                if (not (line.lstrip().startswith('#')) and not ((line.lstrip().startswith('@')))):
                    cols = line.split() 
                    if len(cols)==2:
                        angles.append(float(cols[0]))
                        freqs.append(float(cols[1]))
        agroups_dict[aname]=(angles,freqs)
    
    agroups_dict_avg ={}
    for name,value in agroups_dict.items():
        agroups_dict_avg[name] = sum(value[1])/(float(len(value[1])))
    akeys, avalues = list(zip(*agroups_dict_avg.items()))
    
    avalues = list(avalues)
    
    #truncate to 3 floats  
    for i in range(len(avalues)): 
        avalues[i] = float("%.3f" % avalues[i])
        
        
    a, abins = np.histogram(avalues, bins='auto')
    amids = 0.5*(abins[1:] + abins[:-1])
    amean = np.average(amids, weights=a)
    avar = np.average((amids - amean)**2, weights=a)
    astd = np.sqrt(avar)
    
    return avalues, a, abins, amids, amean, avar, astd


def get_impropers(path):
    files2 = [f for f in os.listdir(path) if f.endswith('.xvg')]
    path2 = path
    groups_dict2 = dict()
    for file in files2:
        name2 = file.split('.')[0]
        with open(path2 + str(file)) as f:
            angles = list()
            freqs = list()
            for line in f:
                if (not (line.lstrip().startswith('#')) and not ((line.lstrip().startswith('@')))):
                    cols = line.split() 
                    if len(cols)==2:
                        angles.append(float(cols[0]))
                        freqs.append(float(cols[1]))
        groups_dict2[name2]=(angles,freqs)
    
    groups_dict2_avg ={}
    for name,value in groups_dict2.items():
        groups_dict2_avg[name] = sum(value[1])/(float(len(value[1])))
    ikeys, ivalues = list(zip(*groups_dict2_avg.items()))
    
    ivalues = list(ivalues)
    
    #truncate to 3 floats 
    for i in range(len(ivalues)): 
        ivalues[i] = float("%.3f" % ivalues[i])
    
    i, ibins = np.histogram(ivalues, bins='auto')
    imids = 0.5*(ibins[1:] + ibins[:-1])
    imean = np.average(imids, weights=i)
    ivar = np.average((imids - imean)**2, weights=i)
    istd = np.sqrt(ivar)
    
    return ivalues, i, ibins, imids, imean, ivar, istd
    
###################
# READ DATA FILES #
###################

#~~~~~~~~
# BONDS #
#~~~~~~~~

path = './*.xvg' # path to folder
bonds, n, bins, mids, mean, var, std = get_bonds(path)

#~~~~~~~~~
# ANGLES # 
#~~~~~~~~~

path = '' # path to folder 
avalues, a, abins, amids, amean, avar, astd = get_angles(path)

#~~~~~~~~~~~~
# IMPROPERS #
#~~~~~~~~~~~~

path = '' # path to folder 
ivalues, i, ibins, imids, imean, ivar, istd = get_impropers(path)

###########
# SEABORN # 
###########

#~~~~~~~~
# BONDS #
#~~~~~~~~

fig = plt.figure(figsize=[21,14])

sns.distplot(np.mean(bonds, axis=1), color="black", label='System 1', 
             kde_kws={"color": "k", "lw": 5, "label": "KDE"},
             hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": "black"})
patch1 = mpatches.Patch(color='black', label='System 1')


plt.title('Average bond distribution', fontsize=25)
plt.xlabel('Average bond lengths', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18) 
#plt.legend(handles=[patch1], loc='upper left', fontsize=20)
plt.show()
fig.savefig('bonds.png', bbox_inches='tight')

# Notes: 
# 1) divided the CG bonds by 2 because the way it's coarse grained makes it approx. double of the avg OPLS bond length

#~~~~~~~~~
# ANGLES # 
#~~~~~~~~~

fig = plt.figure(figsize=[21,14])
sns.distplot(avalues, color="black", label='OPLS 5 rings', 
             kde_kws={"color": "black", "lw": 5, "label": "KDE"},
             hist_kws={"histtype": "step", "linewidth": 1,"alpha": 1, "color": "black"})
patch1 = mpatches.Patch(color='black', label='OPLS 5 rings')

plt.title('Average angle distribution', fontsize=25)
plt.xlabel('Average angles', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18) 
plt.legend(handles=[patch1], loc='upper left', fontsize=20)
plt.show()
fig.savefig('angles.png', bbox_inches='tight')

#~~~~~~~~~~~~
# IMPROPERS # 
#~~~~~~~~~~~~

fig = plt.figure(figsize=[21,14])
sns.distplot(ivalues, color="black", label='OPLS 5 rings', 
             kde_kws={"color": "black", "lw": 5, "label": "KDE"},
             hist_kws={"histtype": "step", "linewidth": 1,"alpha": 1, "color": "black"})
patch1 = mpatches.Patch(color='black', label='OPLS 5 rings')

plt.title('Average improper dihedral distribution', fontsize=25)
plt.xlabel('Average angles', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18) 
plt.legend(handles=[patch1], loc='upper left', fontsize=20)
plt.show()
fig.savefig('impropers.png', bbox_inches='tight')


