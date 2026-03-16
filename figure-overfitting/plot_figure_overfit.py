import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Load in the transport map class
from transport_map import transport_map

# Find the current working directory
root_directory = os.path.dirname(os.path.realpath(__file__))

def draw_multivariate_Gaussian_samples(size):
    
    seeds = np.random.uniform(size=size,low=0,high=1)
    
    samples = np.zeros(size)
    
    below = np.where(seeds <= 0.25)[0]
    above = np.where(seeds > 0.25)[0]
    
    samples[below] = scipy.stats.norm.rvs(
        loc = -1,
        scale = 0.5,
        size = len(below))
    
    samples[above] = scipy.stats.norm.rvs(
        loc = 1,
        scale = 0.5,
        size = len(above))
    
    
    return samples

def multivariate_Gaussian_pdf(x):
    
    return scipy.stats.norm.pdf(
        loc = -1,
        scale = 0.5,
        x = x)/4 + scipy.stats.norm.pdf(
        loc = 1,
        scale = 0.5,
        x = x)/4*3

# Set a random seed
np.random.seed(42)

# Close all open figures
plt.close('all')

N = 25
D = 1

X = draw_multivariate_Gaussian_samples(N)

X_extra = draw_multivariate_Gaussian_samples(10000)

Z = scipy.stats.norm.rvs(
    size = 10000)[:,None]

z = np.linspace(-3,3,1001)

plt.figure(figsize=(12,6))

gs = GridSpec(
    nrows = 3,
    ncols = 3,
    width_ratios = [1,3,1],
    hspace = 0.3,
    wspace = 0.3)

subgs = GridSpecFromSubplotSpec(
    nrows   = 3, 
    ncols   = 3, 
    subplot_spec = gs[:,1],
    hspace = 0.3,
    wspace = 0.3)

plt.subplot(gs[1,0])

plt.plot(
    multivariate_Gaussian_pdf(z),
    z,
    color = "#666",
)

plt.scatter(
    np.zeros(N),
    X,
    marker = "x",
    color = "xkcd:orangish red")

plt.hist(
    X_extra,
    bins=30, 
    orientation="horizontal",
    color = "xkcd:orangish red",
    alpha = 0.6,
    density=True, 
    zorder = -10)

plt.gca().set_ylim([-3,3])

plt.gca().xaxis.set_inverted(True)

plt.title(r"Target distribution",loc="left")
plt.gca().set_xticks([])

plt.ylabel("$x$")


plt.subplot(gs[1,-1])

plt.plot(
    scipy.stats.norm.pdf(z),
    z,
    color = "#666",
)

plt.hist(
    Z,
    bins=30, 
    orientation="horizontal",
    color = "xkcd:grass green",
    alpha = 0.3,
    density=True, 
    zorder = -10)

plt.scatter(
    np.zeros(N),
    Z[:N,0],
    color = "xkcd:grass green",
    marker="x")

plt.ylabel("$z$")

plt.title(r"Reference distribution",loc="left")
plt.gca().set_xticks([])

#%%

# =============================================================================
# Linear complexity
# =============================================================================

monotone = [
  [[0]] ]
nonmonotone = [
  [ [] ] ]


tm     = transport_map(
    monotone                = monotone,
    nonmonotone             = nonmonotone,
    X                       = X[:,None], # Dummy input
    monotonicity            = "separable monotonicity",
    standardize_samples     = False,
    ST_scale_mode           = "dynamic",
    verbose                 = False)

tm.optimize()

print(tm.special_terms)


X_ret = tm.inverse_map(Z)

Z_out = tm.map(X[:,None])[:,0]


plt.subplot(subgs[1,0])

plt.plot(
    tm.map(z[:,None])[:,0],
    z,
    color = "#666",)

plt.xlabel(r"$z$",labelpad=-2.5)
plt.ylabel(r"$x$")

xlims = [-3,3] #plt.gca().get_xlim()
ylims = [-3,3] #plt.gca().get_ylim()
plt.gca().set_xlim(xlims)
plt.gca().set_ylim(ylims)

for idx,x in enumerate(X):

    plt.plot(
        [-0.3,(Z_out[idx] - ylims[0])/(ylims[1] - ylims[0]),(Z_out[idx] - ylims[0])/(ylims[1] - ylims[0])],
        [(x - ylims[0])/(ylims[1] - ylims[0]), (x - ylims[0])/(ylims[1] - ylims[0]), 1.1],
        transform = plt.gca().transAxes,
        color = "xkcd:orangish red",
        lw = 1,
        alpha = 0.3)
    
plt.scatter(
    np.ones(N)*xlims[0],
    X,
    color = "xkcd:orangish red",
    marker="x")

plt.scatter(
    Z_out,
    np.ones(N)*ylims[1],
    color = "xkcd:orangish red",
    marker="+")

plt.plot(
    z,
    tm.map(z[:,None])[:,0],
    color = "#666",
    ls = "--")

xlims = [-3,3]
ylims = [-3,3]
plt.gca().set_xlim(xlims)
plt.gca().set_ylim(ylims)

for idx,x in enumerate(X):
    
    plt.plot(
        [(X_ret[idx,0] - ylims[0])/(ylims[1] - ylims[0]), (X_ret[idx,0] - ylims[0])/(ylims[1] - ylims[0]),1],
        [xlims[0],(Z[idx,0] - ylims[0])/(ylims[1] - ylims[0]),(Z[idx,0] - ylims[0])/(ylims[1] - ylims[0])],
        transform = plt.gca().transAxes,
        color = "xkcd:grass green",
        lw = 1,
        alpha = 0.3)
    
plt.scatter(
    np.ones(N)*xlims[1],
    Z[:N,0],
    color = "xkcd:grass green",
    marker="x")

plt.scatter(
    X_ret[:N,0],
    np.ones(N)*ylims[0],
    color = "xkcd:grass green",
    marker="+")


plt.subplot(subgs[2,0])

plt.scatter(
    X_ret[:N,0],
    np.zeros(N),
    marker = "+",
    color = "xkcd:grass green")

plt.hist(X_ret,zorder=-10,density=True,bins = 100,color="xkcd:grass green",alpha=0.6)

plt.plot(
    z,
    multivariate_Gaussian_pdf(z),
    color = "#666"
)

plt.xlabel("$x$",labelpad=-2.5)
plt.ylabel(r"$S^{\#}\eta (x)$")
plt.xlim(-3.,3.)

plt.subplot(subgs[0,0])

Z_push = tm.map(X_extra[:,None])[:,0]

plt.hist(Z_push,zorder=-10,density=True,bins = 100,color="xkcd:orangish red",alpha=0.6)

plt.plot(
    z,
    scipy.stats.norm.pdf(z),
    color = "#666"
)

plt.scatter(
    Z_out,
    np.zeros(N),
    color = "xkcd:orangish red",
    marker="+")

plt.title(r"$\mathbf{A}$: Linear map",loc="left")
plt.xlabel("$z$",labelpad=-2.5)
plt.ylabel(r"$S_{\#}\pi (z)$")
plt.xlim(-3.,3.)


#%%

# =============================================================================
# Moderate complexity
# =============================================================================

number_terms = 10

monotone = [
  [[0]]+['iRBF 0']*number_terms ]
nonmonotone = [
  [ [] ] ]


tm     = transport_map(
    monotone                = monotone,
    nonmonotone             = nonmonotone,
    X                       = X[:,None], # Dummy input
    monotonicity            = "separable monotonicity",
    standardize_samples     = False,
    ST_scale_mode           = "dynamic",
    verbose                 = False)

tm.special_terms[0][0]["centers"] = np.linspace(-3,3,number_terms)

tm.function_constructor_alternative()

tm.precalculate()

tm.optimize()

X_ret = tm.inverse_map(Z)

Z_out = tm.map(X[:,None])[:,0]

plt.subplot(subgs[1,1])

plt.plot(
    tm.map(z[:,None])[:,0],
    z,
    color = "#666")

xlims = [-3,3] 
ylims = [-3,3]
plt.gca().set_xlim(xlims)
plt.gca().set_ylim(ylims)

for idx,x in enumerate(X):

    plt.plot(
        [-0.3,(Z_out[idx] - ylims[0])/(ylims[1] - ylims[0]),(Z_out[idx] - ylims[0])/(ylims[1] - ylims[0])],
        [(x - ylims[0])/(ylims[1] - ylims[0]), (x - ylims[0])/(ylims[1] - ylims[0]), 1.1],
        transform = plt.gca().transAxes,
        color = "xkcd:orangish red",
        lw = 1,
        alpha = 0.3)

plt.scatter(
    np.ones(N)*xlims[0],
    X,
    color = "xkcd:orangish red",
    marker="x")

plt.scatter(
    Z_out,
    np.ones(N)*ylims[1],
    color = "xkcd:orangish red",
    marker="+")

plt.plot(
    z,
    tm.map(z[:,None])[:,0],
    color = "#666",
    ls = "--")

xlims = [-3,3] 
ylims = [-3,3] 
plt.gca().set_xlim(xlims)
plt.gca().set_ylim(ylims)

for idx,x in enumerate(X):
    
    plt.plot(
        [(X_ret[idx,0] - ylims[0])/(ylims[1] - ylims[0]), (X_ret[idx,0] - ylims[0])/(ylims[1] - ylims[0]),1],
        [xlims[0],(Z[idx,0] - ylims[0])/(ylims[1] - ylims[0]),(Z[idx,0] - ylims[0])/(ylims[1] - ylims[0])],
        transform = plt.gca().transAxes,
        color = "xkcd:grass green",
        lw = 1,
        alpha = 0.3)
    
plt.scatter(
    np.ones(N)*xlims[1],
    Z[:N,0],
    color = "xkcd:grass green",
    marker="x")

plt.scatter(
    X_ret[:N,0],
    np.ones(N)*ylims[0],
    color = "xkcd:grass green",
    marker="+")

plt.subplot(subgs[2,1])

plt.scatter(
    X_ret[:N,0],
    np.zeros(N),
    marker = "+",
    color = "xkcd:grass green")

plt.hist(X_ret,zorder=-10,density=True,bins = 100,color="xkcd:grass green",alpha=0.6)

plt.plot(
    z,
    multivariate_Gaussian_pdf(z),
    color = "#666"
)

plt.xlabel("$x$",labelpad=-2.5)
plt.xlim(-3.,3.)


plt.subplot(subgs[0,1])

Z_push = tm.map(X_extra[:,None])[:,0]

plt.hist(Z_push,zorder=-10,density=True,bins = 100,color="xkcd:orangish red",alpha=0.6)

plt.plot(
    z,
    scipy.stats.norm.pdf(z),
    color = "#666"
)

plt.scatter(
    Z_out,
    np.zeros(N),
    color = "xkcd:orangish red",
    marker="+")

plt.title(r"$\mathbf{B}$: Moderate map",loc="left")
plt.xlabel("$z$",labelpad=-2.5)
plt.xlim(-3.,3.)

#%%

# =============================================================================
# Extreme complexity
# =============================================================================

number_terms = 1000

monotone = [
  [[0]]+['iRBF 0']*number_terms ]
nonmonotone = [
  [ [] ] ]


tm     = transport_map(
    monotone                = monotone,
    nonmonotone             = nonmonotone,
    X                       = X[:,None], # Dummy input
    monotonicity            = "separable monotonicity",
    standardize_samples     = False,
    ST_scale_mode           = "dynamic",
    verbose                 = False)

tm.special_terms[0][0]["centers"] = np.linspace(-3,3,number_terms)

tm.function_constructor_alternative()

tm.precalculate()

tm.optimize()

X_ret = tm.inverse_map(Z)

Z_out = tm.map(X[:,None])[:,0]


plt.subplot(subgs[1,2])

plt.plot(
    tm.map(z[:,None])[:,0],
    z,
    color = "#666")


xlims = [-3,3] 
ylims = [-3,3] 
plt.gca().set_xlim(xlims)
plt.gca().set_ylim(ylims)

for idx,x in enumerate(X):

    plt.plot(
        [-0.3,(Z_out[idx] - ylims[0])/(ylims[1] - ylims[0]),(Z_out[idx] - ylims[0])/(ylims[1] - ylims[0])],
        [(x - ylims[0])/(ylims[1] - ylims[0]), (x - ylims[0])/(ylims[1] - ylims[0]), 1.1],
        transform = plt.gca().transAxes,
        color = "xkcd:orangish red",
        lw = 1,
        alpha = 0.3)
    
plt.scatter(
    np.ones(N)*xlims[0],
    X,
    color = "xkcd:orangish red",
    marker="x")

plt.scatter(
    Z_out,
    np.ones(N)*ylims[1],
    color = "xkcd:orangish red",
    marker="+")

plt.plot(
    z,
    tm.map(z[:,None])[:,0],
    color = "#666",
    ls = "--")

plt.xlabel(r"$z$")

xlims = [-3,3]
ylims = [-3,3]
plt.gca().set_xlim(xlims)
plt.gca().set_ylim(ylims)

for idx,x in enumerate(X):
    
    plt.plot(
        [(X_ret[idx,0] - ylims[0])/(ylims[1] - ylims[0]), (X_ret[idx,0] - ylims[0])/(ylims[1] - ylims[0]),1],
        [xlims[0],(Z[idx,0] - ylims[0])/(ylims[1] - ylims[0]),(Z[idx,0] - ylims[0])/(ylims[1] - ylims[0])],
        transform = plt.gca().transAxes,
        color = "xkcd:grass green",
        lw = 1,
        alpha = 0.3)
    
plt.scatter(
    np.ones(N)*xlims[1],
    Z[:N,0],
    color = "xkcd:grass green",
    marker="x")

plt.scatter(
    X_ret[:N,0],
    np.ones(N)*ylims[0],
    color = "xkcd:grass green",
    marker="+")

plt.subplot(subgs[2,2])

plt.scatter(
    X_ret[:N,0],
    np.zeros(N),
    marker = "+",
    color = "xkcd:grass green")

plt.hist(X_ret,zorder=-10,density=True,bins = 100,color="xkcd:grass green",alpha=0.6)

plt.plot(
    z,
    multivariate_Gaussian_pdf(z),
    color = "#666"
)

plt.xlabel("$x$",labelpad=-2.5)
plt.xlim(-3.,3.)

plt.subplot(subgs[0,2])

Z_push = tm.map(X_extra[:,None])[:,0]

plt.hist(Z_push,zorder=-10,density=True,bins = 100,color="xkcd:orangish red",alpha=0.6)

plt.plot(
    z,
    scipy.stats.norm.pdf(z),
    color = "#666"
)

plt.scatter(
    Z_out,
    np.zeros(N),
    color = "xkcd:orangish red",
    marker="+")

plt.title(r"$\mathbf{C}$: Complex map",loc="left")
plt.xlabel("$z$",labelpad=-2.5)
plt.xlim(-3.,3.)

plt.savefig("overfitting.pdf",bbox_inches="tight")