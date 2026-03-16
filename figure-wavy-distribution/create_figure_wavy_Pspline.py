import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os
from PSpline_triangular_transport import adaptive_spline_transport
from matplotlib import gridspec
import pickle
from matplotlib.ticker import MultipleLocator

np.random.seed(42)

# Clear previous figures and pickles
root_directory = os.path.dirname(os.path.realpath(__file__))
results = []
results += [each for each in os.listdir(root_directory) if each.endswith('.png')]
for fl in range(len(results)):
    os.remove(root_directory+'\\'+results[fl])

def density_wavy_distribution(X):
    
    import scipy.stats
    import numpy as np
    import copy
    
    # Scale X
    X       = copy.copy(X)
    X[:,0]  *= 1.5
    X[:,1]  /= 1.5
    
    X[:,1]  -= np.sin(X[:,0]*1.2)
    
    locX    = (X[:,0]/3+1)/2
    locX[np.where(locX < 0.000001)] = 0.000001
    locX[np.where(locX > 0.999999)] = 0.999999
    
    logpdf  = np.zeros(X.shape[0])
    logpdf  += scipy.stats.beta.logpdf(
        x       = locX,
        a       = 2,
        b       = 2)
    logpdf  += scipy.stats.norm.logpdf(
        x       = X[:,1],
        scale   = 1/6)
    
    logpdf += scipy.stats.beta.logpdf(locX, a=2, b=2) - np.log(6)
    
    return logpdf

def sample_wavy_distribution(size):

    X       = np.zeros((size,2))
    X[:,0]  = (scipy.stats.beta.rvs(
        a       = 2,
        b       = 2,
        size    = size)*2-1)*3
    X[:,1]  = scipy.stats.norm.rvs(
        scale   = 1/6,
        size    = size)
    
    X[:,1]  += np.sin(X[:,0]*1.2)
    
    X[:,0] /= 1.5
    X[:,1] *= 1.5
    
    return X

Xg,Yg = np.meshgrid(
    np.linspace(-3,3,101),
    np.linspace(-3,3,101))
Xg = Xg.flatten()
Yg = Yg.flatten()

XY = np.column_stack((Xg,Yg))

# Draw samples
N = 30
X = sample_wavy_distribution(N)

Z = scipy.stats.norm.rvs(size=(1000,2))

X_validation = sample_wavy_distribution(1000)

# Create a figure
fig = plt.figure(figsize=(12,7))

gs = gridspec.GridSpec(
    nrows = 3, 
    ncols = 5,
    hspace = 0.4)

lambdas = np.linspace(-10,10,101)

plot_lambdas = [0,25,48,75,100]

dofs = []
nlls = []
aics = []

true_density = []
true_density_gaussian = []

beta_initial = None

for idx, lmbda in enumerate(lambdas):
    
    print(idx)
    
    # Create and train the transport map
    transport_map = adaptive_spline_transport(
        k = 50, # 10
        skip_dimensions = 0)
    
    if "results_dict_idx="+str(idx).zfill(4)+".p" not in os.listdir(root_directory):
    
        transport_map.train_map(
            X = X, 
            lambda_initial = {
                0:  np.array([10]),
                1:  np.array([lmbda,10])},
            optimize_lambdas = False,
            beta_initial = beta_initial)

        # Save everything needed to apply the map later without re-optimizing
        results_dict = {
            "k"                   : transport_map.k,
            "degree"              : transport_map.degree,
            "D"                   : transport_map.D,
            "lambdas"             : transport_map.lambdas,
            "betas"               : transport_map.betas,
            "scaler"              : transport_map.scaler,
            "Bx"                  : transport_map.Bx,
            "nbases"              : transport_map.nbases,
            "sparsity"            : transport_map.sparsity,
            "training_data_range" : transport_map.training_data_range,
            "dofs"                : transport_map.dofs,
            "aics"                : transport_map.aics,
            "nlls"                : transport_map.nlls
            }
        
        beta_initial = transport_map.betas
        
        pickle.dump(results_dict,open("results_dict_idx="+str(idx).zfill(4)+".p","wb"))
        
    else:
        
        results_dict = pickle.load(open("results_dict_idx="+str(idx).zfill(4)+".p","rb"))
    
        # Restore trained component state
        transport_map.k                   = results_dict["k"]
        transport_map.degree              = results_dict["degree"]
        transport_map.D                   = results_dict["D"]
        transport_map.lambdas             = results_dict["lambdas"]
        transport_map.betas               = results_dict["betas"]
        transport_map.scaler              = results_dict["scaler"]
        transport_map.Bx                  = results_dict["Bx"]
        transport_map.nbases              = results_dict["nbases"]
        transport_map.sparsity            = results_dict["sparsity"]
        transport_map.training_data_range = results_dict["training_data_range"]
        transport_map.dofs                = results_dict["dofs"]
        transport_map.aics                = results_dict["aics"]
        transport_map.nlls                = results_dict["nlls"]
    
    pullback = transport_map.apply_inverse_map(Z)
    
    pushforward = transport_map.apply_forward_map(X_validation)
    
    print(f"dofs: {transport_map.dofs}")
    dofs.append(transport_map.dofs[1])
    print(f"aics: {transport_map.aics}")
    aics.append(transport_map.aics[1])
    print(f"nlls: {transport_map.nlls}")
    nlls.append(transport_map.nlls[1])
    print(f"lambdas: {transport_map.lambdas}")
    print(f"betas: {transport_map.betas}")

    if idx in plot_lambdas:

        plt.subplot(gs[0,plot_lambdas.index(idx)])
        
        plt.contour(
            Xg.reshape((101,101)),
            Yg.reshape((101,101)),
            np.exp(density_wavy_distribution(XY)).reshape((101,101)),
            cmap = "Greys")
   
        plt.scatter(
            pullback[:,0],
            pullback[:,1],
            s = 0.25,
            color = "xkcd:orangish red",
            zorder = 10)
        
        plt.gca().xaxis.set_major_locator(MultipleLocator(2))
        plt.gca().yaxis.set_major_locator(MultipleLocator(2))
        
        plt.title("$\log\lambda = {:.1f}$".format(lmbda))
        
        plt.xlabel("$x_1$")
        if idx == plot_lambdas[0]:
            plt.ylabel("$x_2$")
        
        plt.subplot(gs[1,plot_lambdas.index(idx)])
        
        plt.contour(
            Xg.reshape((101,101)),
            Yg.reshape((101,101)),
            scipy.stats.multivariate_normal.pdf(mean=np.zeros(2),cov=np.identity(2),x=XY).reshape((101,101)),
            cmap = "Greys")
        
        plt.scatter(
            pushforward[:,0],
            pushforward[:,1],
            s = 0.5,
            zorder = 10)
        
        plt.gca().xaxis.set_major_locator(MultipleLocator(3))
        plt.gca().yaxis.set_major_locator(MultipleLocator(3))
 
        plt.xlabel("$z_1$")
        if idx == plot_lambdas[0]:
            plt.ylabel("$z_2$")
        
        plt.annotate(
            "",
            xy=(0.5, 1.0),
            xycoords=("axes fraction", "axes fraction"),
            xytext=(0.5, 1.4),
            textcoords=("axes fraction", "axes fraction"),
            arrowprops=dict(arrowstyle="-", lw=1.2, color="#999",ls="--"),
            zorder = -10,
            clip_on=False,
        )
    
aics = np.asarray(aics)
dofs = np.asarray(dofs)

plt.subplot(gs[2,:])    
ax_left = plt.gca()

# First axis (left)
ax_left.plot(lambdas, aics,
             color="xkcd:cerulean",
             label="AIC")
ax_left.set_ylabel("AICc", color="xkcd:cerulean")
ax_left.set_xlabel("smoothing penalty coefficient $\log \lambda$")

xlims = ax_left.get_xlim()
ylims = ax_left.get_ylim()

ax_left.set_xlim(xlims)
ax_left.set_ylim(ylims)

subplot_locs = [0.09,0.295,0.5,0.705,0.91]

for i,idx in enumerate(plot_lambdas):
    
    x = lambdas[idx]
    y = ax_left.get_ylim()[0]

    ax_left.annotate(
        "",
        xy=(x, y), 
        xycoords="data",
        xytext=(x, 1.0),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="-", lw=1.2, color="#999",ls="--"),
        zorder = -10,
        clip_on=False,
    )
    
    ax_left.annotate(
        "",
        xy=(x, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(subplot_locs[i], 1.4),
        textcoords=("axes fraction", "axes fraction"),
        arrowprops=dict(arrowstyle="-", lw=1.2, color="#999",ls="--"),
        zorder = -10,
        clip_on=False,
    )

# Second axis (right)
ax_right = ax_left.twinx()
ax_right.plot(lambdas, dofs,
              color="xkcd:orangish red",
              label="DoF")
ax_right.set_ylabel("effective DoF",
                    color="xkcd:orangish red")

# Third axis
ax_third = ax_left.twinx()
ax_third.spines["right"].set_position(("outward", -10))
ax_third.spines["right"].set_visible(True)
ax_third.spines["left"].set_visible(False)
ax_third.yaxis.set_label_position("right")
ax_third.yaxis.set_ticks_position("right")

# Make ticks point inward
ax_third.tick_params(axis='y', direction='in', pad=-5)
for tick in ax_third.get_yticklabels():
    tick.set_horizontalalignment("right")

ax_third.yaxis.set_label_coords(0.925, 0.5)

ax_third.plot(
    lambdas, 
    nlls,
    color="xkcd:grass green",
    label="NLL")
ax_third.set_ylabel(
    "NLL",
   color="xkcd:grass green")

ax_right.set_ylim([0, ax_right.get_ylim()[1]])

plt.savefig("AIC_and_DOF.pdf", bbox_inches="tight")