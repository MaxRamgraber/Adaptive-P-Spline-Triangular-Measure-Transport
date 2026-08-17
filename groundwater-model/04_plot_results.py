import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

X_star = pickle.load(open("X_star.p","rb"))
X_star_linear = pickle.load(open("X_star_linear.p","rb"))
X_star_EnKS = pickle.load(open("X_star_EnKS.p","rb"))
Cholesky_dictionary = pickle.load(open("Cholesky_dictionary.p","rb"))

X_prior    = np.delete(Cholesky_dictionary["log_hk"], observed_index, axis=0)

obs_indices = Cholesky_dictionary["obs_indices"]
num_obs = len(obs_indices)

# Get the map ordering
S = Cholesky_dictionary["S"]
reverse_order = np.argsort(S)

# Revert the vectorized ordering
X_star = X_star
X_star_EnKS = X_star_EnKS

X_star_grid = X_star[:,num_obs:][:,reverse_order].reshape((N,nrows,ncols))
X_star_linear_grid = X_star_linear[:,num_obs:][:,reverse_order].reshape((N,nrows,ncols))
X_star_EnKS_grid = X_star_EnKS[:,num_obs:][:,reverse_order].reshape((N,nrows,ncols))
X_prior_grid = X_prior.reshape((N,nrows,ncols))

fig = plt.figure(figsize = (12,6))

from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec

gs = GridSpec(
    nrows = 2,
    ncols = 4,
    wspace = 0.25)

for col,X in enumerate([X_prior_grid,X_star_grid,X_star_EnKS_grid]):
    
    ax = plt.subplot(gs[0,col])
    im = ax.imshow(np.mean(X,axis=0), cmap = "turbo")
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
        plt.title("Prior mean $\log_{10}K$"+"\n"+"(both)",loc="left")
    elif col == 1:
        plt.title("Posterior mean $\log_{10}K$"+"\n"+"(P-Spline EnTS)",loc="left")
    else:
        plt.title("Posterior mean $\log_{10}K$"+"\n"+"(EnKS)",loc="left")
    for r,c in obs_indices:
        plt.scatter(
            c,
            r,
            color = "r",
            marker = "+")
    cax = inset_axes(
        ax,
        width="6%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0)
    cbar = plt.colorbar(im, cax = cax)
    cbar.ax.yaxis.label.set_size(12)
    cbar.ax.tick_params(labelsize=12)
    
    ax = plt.subplot(gs[1,col])
    if col != 1:
        im = ax.imshow(np.std(X,axis=0), cmap = "turbo")
    else:
        im = ax.imshow(np.std(X,axis=0), cmap = "turbo",vmax=0.75) # -marked-
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
        plt.title("Prior std $\log_{10}K$"+"\n"+"(both)",loc="left")
    elif col == 1:
        plt.title("Posterior std $\log_{10}K$"+"\n"+"(P-Spline EnTS)",loc="left")
    else:
        plt.title("Posterior std $\log_{10}K$"+"\n"+"(EnKS)",loc="left")
    for r,c in obs_indices:
        plt.scatter(
            c,
            r,
            color = "r",
            marker = "+")
    cax = inset_axes(
        ax, 
        width="6%", 
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0, 1, 1), 
        bbox_transform=ax.transAxes,
        borderpad=0)
    cbar = plt.colorbar(im, cax = cax)
    cbar.ax.yaxis.label.set_size(12)
    cbar.ax.tick_params(labelsize=12)
    
ax = plt.subplot(gs[0,3])

im = ax.imshow(Cholesky_dictionary["log_hk"][observed_index,:].reshape((51,51)), cmap = "turbo")
ax.set_xticks([])
ax.set_yticks([])
for r,c in obs_indices:
    ax.scatter(
        c,
        r,
        color = "r",
        marker = "+")
plt.title("true field $\log_{10}K$",loc="left")
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cax = inset_axes(
    ax,
    width="6%",
    height="100%",
    loc="lower left",
    bbox_to_anchor=(1.05, 0, 1, 1), 
    bbox_transform=ax.transAxes,
    borderpad=0)
cbar = plt.colorbar(im, cax = cax)
cbar.ax.yaxis.label.set_size(12)
cbar.ax.tick_params(labelsize=12)

fig.subplots_adjust(
    left=0.0,
    right=0.95,
    wspace=0.5,
    hspace=0.25
)

#%%

Ns = [0,10,25,50,75,90]
fig = plt.figure(figsize=(14,6))

gs = GridSpec(
    nrows = 3,
    ncols = 6,
    wspace = 0.5)

for idx,N in enumerate(Ns):
    
    ax = plt.subplot(gs[0,idx])
    grid = X_prior[N,:].reshape((nrows,ncols))
    im = ax.imshow(grid, cmap = "turbo")
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.title("particle "+str(N))
    if idx == 0: ax.set_ylabel("EnKS prior", fontsize = 12)
    for r,c in obs_indices:
        ax.scatter(
            c,
            r,
            color = "r",
            marker = "+")
    cax = inset_axes(
        ax,
        width="6%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0, 1, 1), 
        bbox_transform=ax.transAxes,
        borderpad=0)
    
    label = None
    if idx == len(Ns)-1:
        label = "$\log_{10}K$ [$\log_{10}$ m/s]"
    cbar = plt.colorbar(im, cax = cax, label = label)
    cbar.ax.yaxis.label.set_size(12)
    cbar.ax.tick_params(labelsize=12)
    
    
    ax = plt.subplot(gs[1,idx])
    grid = X_star_EnKS[N,num_obs:][reverse_order].reshape((nrows,ncols))
    im = ax.imshow(grid, cmap = "turbo")
    ax.set_xticks([])
    ax.set_yticks([])
    
    if idx == 0: ax.set_ylabel("EnKS posterior", fontsize = 12)
    for r,c in obs_indices:
        ax.scatter(
            c,
            r,
            color = "r",
            marker = "+")
    cax = inset_axes(
        ax,
        width="6%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0, 1, 1), 
        bbox_transform=ax.transAxes,
        borderpad=0)
    label = None
    if idx == len(Ns)-1:
        label = "$\log_{10}K$ [$\log_{10}$ m/s]"
    cbar = plt.colorbar(im, cax = cax, label = label)
    cbar.ax.yaxis.label.set_size(12)
    cbar.ax.tick_params(labelsize=12)
    
    ax = plt.subplot(gs[2,idx])
    grid = X_star[N,num_obs:][reverse_order].reshape((nrows,ncols))
    im = ax.imshow(grid, cmap = "turbo")
    ax.set_xticks([])
    ax.set_yticks([])
    
    if idx == 0: ax.set_ylabel("EnTS posterior", fontsize = 12)
    for r,c in obs_indices:
        ax.scatter(
            c,
            r,
            color = "r",
            marker = "+")
    cax = inset_axes(
        ax,
        width="6%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0, 1, 1), 
        bbox_transform=ax.transAxes,
        borderpad=0)
    label = None
    if idx == len(Ns)-1:
        label = "$\log_{10}K$ [$\log_{10}$ m/s]"
    cbar = plt.colorbar(im, cax = cax, label = label)
    cbar.ax.yaxis.label.set_size(12)
    cbar.ax.tick_params(labelsize=12)
    
fig.subplots_adjust(
    bottom = 0.01,
    top = 0.985,
    left=0.025,
    right=0.925,
    wspace=0.5,
)
    
plt.savefig("posterior_samples_Darcy.png",dpi=300)
plt.savefig("posterior_samples_Darcy.pdf",dpi=300) 

#%%

focus_cell = (15,25)

fig = plt.figure(figsize=(12,5.5))
gs = GridSpec(
    nrows = 3,
    ncols = len(obs_indices),
    wspace = 0.4,
    hspace = 0.4)

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.markers import MarkerStyle

for idx,obs_idx in enumerate(obs_indices):
    
    ax = plt.subplot(gs[0,idx])
    
    ax.set_title("cell {}".format((int(obs_idx[0]),int(obs_idx[1]))), fontsize = 10)
    
    xrange = [
        np.min(X_prior_grid[:,obs_idx[0],obs_idx[1]]),
        np.max(X_prior_grid[:,obs_idx[0],obs_idx[1]])]
    xrange = np.asarray(xrange)
    xrange = [xrange[0] - np.diff(xrange)*0.1, xrange[1] + np.diff(xrange)*0.1]
    
    yrange = [
        np.min(X_prior_grid[:,focus_cell[0],focus_cell[1]]),
        np.max(X_prior_grid[:,focus_cell[0],focus_cell[1]])]
    yrange = np.asarray(yrange)
    yrange = [yrange[0] - np.diff(yrange)*0.1, yrange[1] + np.diff(yrange)*0.1]

    ax.scatter(
        X_prior_grid[:,obs_idx[0],obs_idx[1]], 
        X_prior_grid[:,focus_cell[0],focus_cell[1]], 
        c=plt.get_cmap("turbo")((X_prior_grid[:,obs_idx[0],obs_idx[1]] - xrange[0])/(xrange[1]-xrange[0])), 
        edgecolor="k", 
        lw = 0.25,
        marker=MarkerStyle("o", fillstyle="right"))
    
    ax.scatter(
        X_prior_grid[:,obs_idx[0],obs_idx[1]], 
        X_prior_grid[:,focus_cell[0],focus_cell[1]], 
        c=plt.get_cmap("turbo")((X_prior_grid[:,focus_cell[0],focus_cell[1]] - xrange[0])/(xrange[1]-xrange[0])), 
        edgecolor="k", 
        lw = 0.25,
        marker=MarkerStyle("o", fillstyle="left"))
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if idx == 0:
        plt.ylabel("prior"+"\n"+"cell {}".format(focus_cell), fontsize = 12)
    
    ylims = plt.gca().get_ylim()
    
    
    norm = mcolors.Normalize(vmin=yrange[0], vmax=yrange[1])
    sm   = cm.ScalarMappable(norm=norm, cmap="turbo")
    sm.set_array([])        # required, even if empty, to satisfy .colorbar()
    
    cax = inset_axes(
        ax,
        width="6%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0, 1, 1), 
        bbox_transform=ax.transAxes,
        borderpad=0)
    label = None
    if idx == len(obs_indices)-1:
        label = "$\log_{10}K$ [$\log_{10}$ m/s]"
    cbar = plt.colorbar(sm, cax = cax,label=label)
    cbar.ax.yaxis.label.set_size(12)
    cbar.ax.tick_params(labelsize=12)
    
    norm2 = mcolors.Normalize(vmin=xrange[0], vmax=xrange[1])
    sm2   = cm.ScalarMappable(norm=norm2, cmap="turbo")
    sm2.set_array([])        # required, even if empty, to satisfy .colorbar()
    
    cax2 = inset_axes(
        ax,
        width="100%", 
        height="6%",
        loc="lower left",
        bbox_to_anchor=(0.0, -0.11, 1, 1), 
        bbox_transform=ax.transAxes,
        borderpad=0)
    cbar2 = plt.colorbar(sm2, cax = cax2, orientation = "horizontal")
    cbar2.ax.xaxis.label.set_size(12)
    cbar2.ax.tick_params(labelsize=12)

    
    

    ax = plt.subplot(gs[1,idx])
    
    xrange = [
        np.min(X_star_EnKS_grid[:,obs_idx[0],obs_idx[1]]),
        np.max(X_star_EnKS_grid[:,obs_idx[0],obs_idx[1]])]
    xrange = np.asarray(xrange)
    xrange = [xrange[0] - np.diff(xrange)*0.1, xrange[1] + np.diff(xrange)*0.1]
    
    yrange = [
        np.min(X_star_EnKS_grid[:,focus_cell[0],focus_cell[1]]),
        np.max(X_star_EnKS_grid[:,focus_cell[0],focus_cell[1]])]
    yrange = np.asarray(yrange)
    yrange = [yrange[0] - np.diff(yrange)*0.1, yrange[1] + np.diff(yrange)*0.1]
    
    ax.scatter(
        X_star_EnKS_grid[:,obs_idx[0],obs_idx[1]], 
        X_star_EnKS_grid[:,focus_cell[0],focus_cell[1]], 
        c=plt.get_cmap("turbo")((X_star_EnKS_grid[:,obs_idx[0],obs_idx[1]] - xrange[0])/(xrange[1]-xrange[0])), 
        edgecolor="k", 
        lw = 0.25,
        marker=MarkerStyle("o", fillstyle="right"))
    
    ax.scatter(
        X_star_EnKS_grid[:,obs_idx[0],obs_idx[1]], 
        X_star_EnKS_grid[:,focus_cell[0],focus_cell[1]], 
        c=plt.get_cmap("turbo")((X_star_EnKS_grid[:,focus_cell[0],focus_cell[1]] - xrange[0])/(xrange[1]-xrange[0])), 
        edgecolor="k", 
        lw = 0.25,
        marker=MarkerStyle("o", fillstyle="left"))
    
    ax.scatter(
        X_prior_grid[:,obs_idx[0],obs_idx[1]], 
        X_prior_grid[:,focus_cell[0],focus_cell[1]], 
        c="xkcd:grey", 
        edgecolor="k", 
        zorder = -10,
        lw = 0.25)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if idx == 0:
        plt.ylabel("EnKS"+"\n"+"posterior"+"\n"+"cell {}".format(focus_cell), fontsize = 12)
    
    ylims = plt.gca().get_ylim()
    
    norm = mcolors.Normalize(vmin=yrange[0], vmax=yrange[1])
    sm   = cm.ScalarMappable(norm=norm, cmap="turbo")
    sm.set_array([]) 
    
    cax = inset_axes(
        ax,
        width="6%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0, 1, 1), 
        bbox_transform=ax.transAxes,
        borderpad=0)
    label = None
    if idx == len(obs_indices)-1:
        label = "$\log_{10}K$ [$\log_{10}$ m/s]"
    cbar = plt.colorbar(sm, cax = cax,label=label)
    cbar.ax.yaxis.label.set_size(12)
    cbar.ax.tick_params(labelsize=12)
    
    norm2 = mcolors.Normalize(vmin=xrange[0], vmax=xrange[1])
    sm2   = cm.ScalarMappable(norm=norm2, cmap="turbo")
    sm2.set_array([])
    
    cax2 = inset_axes(
        ax, 
        width="100%",
        height="6%",
        loc="lower left",
        bbox_to_anchor=(0.0, -0.11, 1, 1), 
        bbox_transform=ax.transAxes,
        borderpad=0)
    
    cbar2 = plt.colorbar(sm2, cax = cax2, orientation = "horizontal")
    cbar2.ax.xaxis.label.set_size(12)
    cbar2.ax.tick_params(labelsize=12)

    
    
    
    ax = plt.subplot(gs[2,idx])
    
    xrange = [
        np.min(X_star_grid[:,obs_idx[0],obs_idx[1]]),
        np.max(X_star_grid[:,obs_idx[0],obs_idx[1]])]
    xrange = np.asarray(xrange)
    xrange = [xrange[0] - np.diff(xrange)*0.1, xrange[1] + np.diff(xrange)*0.1]
    
    yrange = [
        np.min(X_star_grid[:,focus_cell[0],focus_cell[1]]),
        np.max(X_star_grid[:,focus_cell[0],focus_cell[1]])]
    yrange = np.asarray(yrange)
    yrange = [yrange[0] - np.diff(yrange)*0.1, yrange[1] + np.diff(yrange)*0.1]
    
    ax.scatter(
        X_star_grid[:,obs_idx[0],obs_idx[1]], 
        X_star_grid[:,focus_cell[0],focus_cell[1]], 
        c=plt.get_cmap("turbo")((X_star_grid[:,obs_idx[0],obs_idx[1]] - xrange[0])/(xrange[1]-xrange[0])), 
        edgecolor="k", 
        lw = 0.25,
        marker=MarkerStyle("o", fillstyle="right"))
    
    ax.scatter(
        X_star_grid[:,obs_idx[0],obs_idx[1]], 
        X_star_grid[:,focus_cell[0],focus_cell[1]], 
        c=plt.get_cmap("turbo")((X_star_grid[:,focus_cell[0],focus_cell[1]] - xrange[0])/(xrange[1]-xrange[0])), 
        edgecolor="k", 
        lw = 0.25,
        marker=MarkerStyle("o", fillstyle="left"))
    
    ax.scatter(
        X_prior_grid[:,obs_idx[0],obs_idx[1]], 
        X_prior_grid[:,focus_cell[0],focus_cell[1]], 
        c="xkcd:grey", 
        edgecolor="k", 
        zorder = -10,
        lw = 0.25)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if idx == 0:
        plt.ylabel("P-Spline EnTS"+"\n"+"posterior"+"\n"+"cell {}".format(focus_cell), fontsize = 12)
    
    ylims = plt.gca().get_ylim()
    
    norm = mcolors.Normalize(vmin=yrange[0], vmax=yrange[1])
    sm   = cm.ScalarMappable(norm=norm, cmap="turbo")
    sm.set_array([]) 
    
    cax = inset_axes(
        ax,
        width="6%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0)
    label = None
    if idx == len(obs_indices)-1:
        label = "$\log_{10}K$ [$\log_{10}$ m/s]"
    cbar = plt.colorbar(sm, cax = cax,label=label)
    cbar.ax.xaxis.label.set_size(12)
    cbar.ax.tick_params(labelsize=12)
    
    norm2 = mcolors.Normalize(vmin=xrange[0], vmax=xrange[1])
    sm2   = cm.ScalarMappable(norm=norm2, cmap="turbo")
    sm2.set_array([])
    
    cax2 = inset_axes(
        ax, 
        width="100%",
        height="6%",
        loc="lower left",
        bbox_to_anchor=(0.0, -0.11, 1, 1), 
        bbox_transform=ax.transAxes,
        borderpad=0)
    cbar2 = plt.colorbar(sm2, cax = cax2, orientation = "horizontal", label = "$\log_{10}K$ [$\log_{10}$ m/s]")
    cbar2.ax.xaxis.label.set_size(12)
    cbar2.ax.tick_params(labelsize=12)
    
fig.subplots_adjust(
    bottom = 0.125,
    top = 0.95,
    left=0.05,
    right=0.925,
    wspace=0.5,
)
    
    
plt.savefig("bimodal_marginals_Darcy.png",dpi=300)
plt.savefig("bimodal_marginals_Darcy.pdf",dpi=300) 

def bound_violation_statistics(X, lower=-7.0, upper=-5.0):
    violations = (X < lower) | (X > upper)

    violation_fraction = np.mean(violations)

    violation_magnitude = np.mean(
        np.maximum(lower - X, 0.0)
        + np.maximum(X - upper, 0.0)
    )

    return violation_fraction, violation_magnitude


v_EnKS, e_EnKS = bound_violation_statistics(X_star_EnKS_grid)
v_EnTS, e_EnTS = bound_violation_statistics(X_star_grid)

print(f"EnKS: violation fraction = {100*v_EnKS:.1f}%")
print(f"EnKS: mean violation magnitude = {e_EnKS:.4f}")
print(f"P-Spline EnTS: violation fraction = {100*v_EnTS:.1f}%")
print(f"P-Spline EnTS: mean violation magnitude = {e_EnTS:.4f}")