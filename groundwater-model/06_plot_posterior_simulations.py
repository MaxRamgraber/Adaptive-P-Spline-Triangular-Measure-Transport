import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import copy
import os
import pickle
from matplotlib.gridspec import GridSpec

root_directory = os.path.dirname(os.path.realpath(__file__))

# Find all results dictionaries
results = []
for file in list(os.listdir(root_directory)):
    if file.startswith("results_dict"):
        results.append(file)
np.sort(results)

observed_index = 50

N = 100
nrows = 51
ncols = 51

Cholesky_dictionary = pickle.load(open("Cholesky_dictionary.p","rb"))
X_prior    = np.delete(Cholesky_dictionary["log_hk"], observed_index, axis=0)
X_prior_h  = np.delete(Cholesky_dictionary["X"], observed_index, axis=0)

obs_indices = Cholesky_dictionary["obs_indices"]
num_obs = len(obs_indices)

S = Cholesky_dictionary["S"]
reverse_order = np.argsort(S)

X_star = pickle.load(open("X_star.p","rb"))
X_star_EnKS = pickle.load(open("X_star_EnKS.p","rb"))

X_star = np.zeros((N,int(nrows*ncols)+num_obs))
X_star_EnKS = np.zeros((N,int(nrows*ncols)+num_obs))

residuals = []
residuals_EnKS = []

residuals_conc = []
residuals_conc_EnKS = []

truth = pickle.load(open("models"+"\\"+"results_"+str(observed_index).zfill(4)+".p","rb"))

conc_star = np.zeros((N,nrows,ncols))
conc_star_EnKS = np.zeros((N,nrows,ncols))

for n in range(N):
    
    dct = pickle.load(open(root_directory+"\\"+"models_posterior"+"\\"+"posterior_simulation_EnKS_"+str(n).zfill(4)+".p","rb"))
    X_star_EnKS[n,num_obs:] = dct["h"].flatten()[S]
    
    residuals_EnKS.append(dct["h"] - truth["h"])
    
    residuals_conc_EnKS.append(dct["conc"] - truth["conc"])
    
    conc_star_EnKS[n,...] = copy.copy(dct["conc"])
    
    dct = pickle.load(open(root_directory+"\\"+"models_posterior"+"\\"+"posterior_simulation_"+str(n).zfill(4)+".p","rb"))
    X_star[n,num_obs:] = dct["h"].flatten()[S]
    residuals.append(dct["h"] - truth["h"])
    
    residuals_conc.append(dct["conc"] - truth["conc"])
    
    conc_star[n,...] = copy.copy(dct["conc"])
    
RMSE = [np.sqrt(np.mean(np.asarray(entry)**2)) for entry in residuals]
RMSE_EnKS = [np.sqrt(np.mean(np.asarray(entry)**2)) for entry in residuals_EnKS]

print("mean RMSE for the EnTS: {}".format(np.mean(RMSE)))
print("mean RMSE for the EnKS: {}".format(np.mean(RMSE_EnKS)))

RMSE_conc = [np.sqrt(np.mean(np.asarray(entry)**2)) for entry in residuals_conc]
RMSE_conc_EnKS = [np.sqrt(np.mean(np.asarray(entry)**2)) for entry in residuals_conc_EnKS]

print("mean RMSE (conc) for the EnTS: {}".format(np.mean(RMSE_conc)))
print("mean RMSE (conc) for the EnKS: {}".format(np.mean(RMSE_conc_EnKS)))

#%%

# Revert the vectorized ordering
X_star = X_star#[:,reverse_order]
X_star_EnKS = X_star_EnKS#[:,reverse_order]

X_star_grid = X_star[:,num_obs:][:,reverse_order].reshape((N,nrows,ncols))
X_star_EnKS_grid = X_star_EnKS[:,num_obs:][:,reverse_order].reshape((N,nrows,ncols))
X_prior_grid = X_prior.reshape((N,nrows,ncols))
X_prior_h_grid = X_prior_h.reshape((N,nrows,ncols))

fig = plt.figure(figsize = (12,6))

gs = GridSpec(
    nrows   = 2,
    ncols   = 4,
    hspace  = 0.5,
    wspace  = 0.5)

for col,X in enumerate([X_prior_h_grid,X_star_grid,X_star_EnKS_grid]):
    
    ax = plt.subplot(gs[0,col])
    im = ax.imshow(np.mean(X,axis=0), cmap = "turbo")
    # plt.colorbar()
    
    for r,c in obs_indices:
        plt.scatter(
            c,
            r,
            color = "k",
            marker = "+")
    
    if col == 0:
        plt.title("Prior mean of $h$"+"\n"+"(both)",loc="left")
    elif col == 1:
        plt.title("Posterior mean of $h$"+"\n"+"(P-Spline EnTS)",loc="left")
    else:
        plt.title("Posterior mean of $h$"+"\n"+"(EnKS)",loc="left")
        
    if col != 0:
        plt.gca().set_yticklabels([])
        
    plt.xlabel("grid columns")
    if col == 0:
        plt.ylabel("grid rows")
    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # ax = plt.gca()
    cax = inset_axes(
        ax,              # parent axes
        width="6%",      # 3 % of the parent’s width
        height="100%",   # exactly the parent’s height
        loc="lower left",
        bbox_to_anchor=(1.05, 0, 1, 1),   # move it just outside the image
        bbox_transform=ax.transAxes,
        borderpad=0)
    cbar = plt.colorbar(im, cax = cax)
    
    ax = plt.subplot(gs[1,col])
    im = ax.imshow(np.std(X,axis=0), cmap = "turbo")
    
    for r,c in obs_indices:
        plt.scatter(
            c,
            r,
            color = "k",
            marker = "+")
        
    if col == 0:
        plt.title("Prior std of $h$ [m]"+"\n"+"(both)",loc="left")
    elif col == 1:
        plt.title("Posterior std of $h$ [m]"+"\n"+"(P-Spline EnTS)",loc="left")
    else:
        plt.title("Posterior std of $h$ [m]"+"\n"+"(EnKS)",loc="left")
        
    if col != 0:
        plt.gca().set_yticklabels([])
        
    plt.xlabel("grid columns")
    if col == 0:
        plt.ylabel("grid rows")

    cax = inset_axes(
        ax,
        width="6%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0)
    cbar = plt.colorbar(im, cax = cax)
        
ax = plt.subplot(gs[0,-1])
im = ax.imshow(Cholesky_dictionary["log_hk"][observed_index,:].reshape((51,51)), cmap = "turbo")
plt.title("true field $K$")
for r,c in obs_indices:
    plt.scatter(
        c,
        r,
        color = "k",
        marker = "+")
    
plt.gca().set_yticklabels([])
plt.xlabel("grid columns")

cax = inset_axes(
    ax, 
    width="6%",
    height="100%", 
    loc="lower left",
    bbox_to_anchor=(1.05, 0, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0)
cbar = plt.colorbar(im, cax = cax)
    
ax = plt.subplot(gs[1,-1])
im = ax.imshow(truth["h"], cmap = "turbo")
plt.title("true field $h$ [m]")
for r,c in obs_indices:
    plt.scatter(
        c,
        r,
        color = "k",
        marker = "+")
    
plt.gca().set_yticklabels([])
plt.xlabel("grid columns")

cax = inset_axes(
    ax, 
    width="6%",
    height="100%", 
    loc="lower left",
    bbox_to_anchor=(1.05, 0, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0)
cbar = plt.colorbar(im, cax = cax)

fig.subplots_adjust(
    left=0.05,
    right=0.925,
    bottom = 0.075,
    top = 0.945,
    wspace=0.5,
    hspace=0.25
)

plt.draw()
plt.savefig("Darcy_heads.pdf",dpi=600)
plt.savefig("Darcy_heads.png",dpi=600)


#%%
    
plt.figure(figsize = (12,6))

for col,X in enumerate([conc_star,conc_star_EnKS]):
    
    plt.subplot(2,3,1+col)
    plt.imshow(np.mean(X,axis=0), cmap = "turbo")
    plt.colorbar()
    
    for r,c in obs_indices:
        plt.scatter(
            c,
            r,
            color = "r",
            marker = "+")
    
    if col == 0:
        plt.title("P-Spline EnTS")
    elif col == 1:
        plt.title("EnKS")
    else:
        plt.title("truth")
    
    plt.subplot(2,3,4+col)
    plt.imshow(np.std(X,axis=0), cmap = "turbo")
    plt.colorbar()
    
    for r,c in obs_indices:
        plt.scatter(
            c,
            r,
            color = "r",
            marker = "+")
        
plt.subplot(2,3,3)
plt.imshow(truth["conc"], cmap = "turbo")
plt.colorbar()
plt.title("true field")
for r,c in obs_indices:
    plt.scatter(
        c,
        r,
        color = "r",
        marker = "+")

    
#%%

residuals_conc = np.asarray(residuals_conc)
residuals_conc_EnKS = np.asarray(residuals_conc_EnKS)
  
plt.figure(figsize = (12,6))

for col,X in enumerate([residuals_conc,residuals_conc_EnKS]):
    
    plt.subplot(2,3,1+col)
    plt.imshow(np.mean(X,axis=0), cmap = "turbo")
    plt.colorbar()
    
    for r,c in obs_indices:
        plt.scatter(
            c,
            r,
            color = "r",
            marker = "+")
    
    if col == 0:
        plt.title("P-Spline EnTS")
    elif col == 1:
        plt.title("EnKS")
    else:
        plt.title("truth")
    
    plt.subplot(2,3,4+col)
    plt.imshow(np.std(X,axis=0), cmap = "turbo")
    plt.colorbar()
    
    for r,c in obs_indices:
        plt.scatter(
            c,
            r,
            color = "r",
            marker = "+")
        
plt.subplot(2,3,3)
plt.imshow(truth["conc"], cmap = "turbo")
plt.colorbar()
plt.title("true field")
for r,c in obs_indices:
    plt.scatter(
        c,
        r,
        color = "r",
        marker = "+")
    
#%%
    
Ns = [0,10,25,50,75,90]
plt.figure(figsize=(14,4))
for idx,N in enumerate(Ns):
    
    plt.subplot(2,len(Ns),1+idx)
    grid = X_star_EnKS[N,num_obs:][reverse_order].reshape((nrows,ncols))
    plt.imshow(grid, cmap = "turbo")
    plt.title("particle "+str(N))
    if idx == 0: plt.ylabel("EnKS")
    plt.axis("equal")
    plt.colorbar()
    
    plt.subplot(2,len(Ns),len(Ns)+1+idx)
    grid = X_star[N,num_obs:][reverse_order].reshape((nrows,ncols))
    plt.imshow(grid, cmap = "turbo")
    if idx == 0: plt.ylabel("P-Spline EnTS")
    plt.axis("equal")
    plt.colorbar()
    
focus_cell = (15,25)

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

plt.figure(figsize=(14,4))
gs = GridSpec(
    nrows = 2,
    ncols = len(obs_indices))

for idx,obs_idx in enumerate(obs_indices):
    
    subplotgs = GridSpecFromSubplotSpec(
        nrows   = 1, 
        ncols   = 2, 
        width_ratios = [1,0.2], 
        subplot_spec = gs[0,idx],
        wspace = 0)
    
    plt.subplot(subplotgs[0])
    
    plt.scatter(
        X_prior_grid[:,obs_idx[0],obs_idx[1]],
        X_prior_grid[:,focus_cell[0],focus_cell[1]],
        color = "b")
    
    plt.scatter(
        X_star_EnKS[:,idx+num_obs],
        X_star_EnKS_grid[:,focus_cell[0],focus_cell[1]],
        color = "xkcd:orange")
    
    ylims = plt.gca().get_ylim()
    
    plt.subplot(subplotgs[1])
    
    plt.hist(
        X_star_EnKS_grid[:,focus_cell[0],focus_cell[1]],
        orientation = "horizontal",
        color = "xkcd:orange",
        alpha = 0.75)
    
    
    plt.gca().set_ylim(ylims)
    
    subplotgs = GridSpecFromSubplotSpec(
        nrows   = 1, 
        ncols   = 2, 
        width_ratios = [1,0.2], 
        subplot_spec = gs[1,idx],
        wspace = 0)
    
    plt.subplot(subplotgs[0])
    
    plt.scatter(
        X_prior_grid[:,obs_idx[0],obs_idx[1]],
        X_prior_grid[:,focus_cell[0],focus_cell[1]],
        color = "b")
    
    plt.scatter(
        X_star[:,idx+num_obs],
        X_star_grid[:,focus_cell[0],focus_cell[1]],
        color = "xkcd:green",
        marker = "x")
    ylims = plt.gca().get_ylim()
    
    plt.subplot(subplotgs[1])

    plt.hist(
        X_star_grid[:,focus_cell[0],focus_cell[1]],
        orientation = "horizontal",
        color = "xkcd:green",
        alpha = 0.75)
    
    plt.gca().set_ylim(ylims)
