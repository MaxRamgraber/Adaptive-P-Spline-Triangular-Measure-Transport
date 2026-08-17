import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib
import copy

use_latex   = False

if use_latex:
    
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    titlesize   = 14
    labelsize   = 12
    addendum    = "_latex"
    
else:
    
    matplotlib.style.use('default')
    titlesize   = 12
    labelsize   = 10
    addendum    = ""

# Clear previous figures and pickles
root_directory = os.path.dirname(os.path.realpath(__file__))

colors  = ['xkcd:cerulean','xkcd:grass green','xkcd:goldenrod','xkcd:orangish red','xkcd:orangish red']
marker  = ["","x","+","v","","^"]
altmarker  = ["","x","*","o","","s"]
orders          = [[1,1],[2,1],[2,2],[3,1],[3,3],[5,1],[5,5]]

random_seeds = [0,1,2,3,4,5,6,7,8,9]

dct = pickle.load(open("EnTF_results.p","rb"))
raw_DoFs = pickle.load(open("raw_DoFs.p","rb"))

Ns = dct["Ns"]

RMSEs = {N: [] for N in Ns}

RMSEs_quantiles = {N: {} for N in Ns}

dofs = {N: np.zeros((10,1000,3,4)) for N in Ns}

for N in Ns:
    
    for seed in random_seeds:
        
        os.chdir(root_directory+"/"+"memory_RS="+str(seed).zfill(2))
    
        res_dct = pickle.load(open("TM_filter_N="+str(N).zfill(4)+"_RS="+str(seed)+".p","rb"))
        
        RMSEs[N].append(np.mean(res_dct["RMSE_list"]))
        
        dofs[N][seed,:,:,:] = copy.copy(res_dct["dofs"])
        
    RMSEs_quantiles[N] = {
        "min"   : np.min(RMSEs[N]),
        "median": np.median(RMSEs[N]),
        "max"   : np.max(RMSEs[N])}
    
    
    RMSEs[N] = np.mean(RMSEs[N])
    
    dofs[N] = np.nanmedian(dofs[N],axis=2) # Average over obs permutations
    dofs[N] = np.nanmedian(dofs[N],axis=1) # Average over time
    dofs[N] = np.nanmean(dofs[N],axis=0) # Average over random seeds

#%%

# Create figure
plt.figure(figsize=(12,8))

# Create first subplot: RMSEs
plt.subplot(2,1,1)

plt.title(r"$\mathbf{A}$: averaged ensemble mean RMSEs for Lorenz-63",loc="left")
plt.ylabel("time-averaged ensemble mean RMSE", fontsize = labelsize)

# Plot the RMSEs of the Ensemble transport smoothing paper
for order in [1,3,5]: 
    
    if order == 1:
        
        plt.plot(
            Ns,
            dct["RMSEval_f_"+str(order)],
            label = 'EnTF (order '+str(order)+') → EnKF',
            marker  = marker[order],
            color=colors[order-1],ls=':')
        
    else:
    
        plt.plot(
            Ns,
            dct["RMSEval_f_"+str(order)],
            label = 'EnTF (order '+str(order)+')',
            marker  = marker[order],
            color=colors[order-1],ls=':')

plt.gca().set_xticks(Ns),
plt.gca().set_xticklabels(Ns, fontsize = labelsize)
plt.yticks(fontsize=labelsize)

RMSEvals_PSpline = [RMSEs[N] for N in Ns]

# Plot the RMSEs of the adaptive P-Spline
plt.plot(
    Ns,
    RMSEvals_PSpline,
    label = 'EnTF (adaptive P-Spline)',
    marker = "s",
    color="xkcd:dark grey")

plt.legend(frameon=False,ncol = 4, prop={"size":8}, fontsize = labelsize, loc = "upper center")

# Plot the DOF development
plt.subplot(2,1,2)

plt.title(r"$\mathbf{B}$: fraction of effective degrees of freedom to origin degrees of freedom",loc="left")

plt.xlabel("ensemble size", fontsize = labelsize)
plt.ylabel("fraction effective DoF to raw DoF", fontsize = labelsize)

plt.gca().set_xticks(Ns),
plt.gca().set_xticklabels(Ns, fontsize = labelsize)
plt.yticks(fontsize=labelsize)

Dofcolors = ["k","#999","#555","#111"]
for idx,color in enumerate(Dofcolors):
    
    if idx != 0:
    
        plt.plot(
            Ns,
            [dofs[n][idx]/raw_DoFs[n][idx] for n in Ns],
            color = color,
            ls = "--",
            label = "$S_{"+str(idx+1)+"}$")
            
plt.legend(frameon=False,ncol = 3, prop={"size":8}, fontsize = labelsize, loc = "lower center")

os.chdir(root_directory)
plt.savefig("results_L63.png",dpi=600,bbox_inches="tight")
plt.savefig("results_L63.pdf",dpi=600,bbox_inches="tight")