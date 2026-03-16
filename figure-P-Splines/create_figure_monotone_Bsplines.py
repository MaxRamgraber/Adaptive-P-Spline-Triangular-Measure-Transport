import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

cmap = plt.get_cmap("turbo")

# Define knot vector and degree
degree = 3  # Quadratic
knots = np.array([0]*degree+[0, 1, 2, 3, 4, 5, 6]+[6]*degree)
n_basis = len(knots) - (degree + 1)

# Define x values for plotting
x = np.linspace(knots[0], knots[-1], 1000)

B = {}

# degree‑0 first --------------------------------------------------
n_basis0 = len(knots) - 1               # one less knot than entries
for i in range(n_basis0):
    B[(0, i)] = ((x >= knots[i]) & (x < knots[i+1])).astype(float)

# higher degrees --------------------------------------------------
for p in range(1, degree + 1):
    n_basis_p = len(knots) - p - 1
    for i in range(n_basis_p):

        # term 1 --------------------------------------------------
        term1 = np.zeros(len(x))
        denom1 = knots[i+p] - knots[i]
        if denom1 != 0:
            term1 = ((x - knots[i]) / denom1) * B[(p-1, i)]

        # term 2 --------------------------------------------------
        term2 = np.zeros(len(x))
        denom2 = knots[i+p+1] - knots[i+1]
        if denom2 != 0:
            term2 = ((knots[i+p+1] - x) / denom2) * B[(p-1, i+1)]

        B[(p, i)] = term1 + term2
        
# Make up increasing coefficients
c = [0.3,0.4,0.8,1.0,1.8,1.9,2.2,2.8,2.9]

plt.figure(figsize=(12,6))
gs = GridSpec(
    nrows   = 3,
    ncols   = 1,
    height_ratios = [0.1,1,1],
    hspace = 0.35)

import matplotlib

plt.subplot(gs[0,0])

cmap = matplotlib.cm.get_cmap('turbo')
import matplotlib
norm = matplotlib.colors.Normalize(vmin=0, vmax=8)
cb1 = matplotlib.colorbar.ColorbarBase(
    plt.gca(), 
    cmap        = cmap,
    ticks       = [0,1,2,3,4,5,6,7,8],
    norm        = norm,
    orientation = 'horizontal')

cb1.set_label("basis function", labelpad=10, fontsize = 12)

plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
cb1.ax.set_xticklabels(["$B_{"+str(i)+",d}$" for i in range(n_basis)], fontsize = 12)  # horizontal colorbar

import matplotlib.patheffects as pe

plt.subplot(gs[1,0])
for i in range(n_basis):
    plt.plot(
        x[:-1],
        B[(degree,i)][:-1]*c[i],
        color = cmap(i/(n_basis-1)))
    plt.fill_between(
        x = x[:-1],
        y1 = B[(degree,i)][:-1]*c[i],
        color = cmap(i/(n_basis-1)),
        alpha = 0.5)
    max_val = np.max(B[(degree,i)][:-1])
    max_x = x[np.where(B[(degree,i)][:-1] == max_val)][0]
    txt = plt.text(
        max_x,
        max_val*c[i] + 0.2,
        "$c_{"+str(i)+"} = "+str(c[i])+"$",
        ha = "center",
        color = "k")
    
    txt.set_path_effects([pe.Stroke(linewidth=0.5, foreground=cmap(i/(n_basis-1))),
                          pe.Normal()])

plt.gca().set_xticks([0,1,2,3,4,5,6])
plt.gca().set_xticklabels([])
plt.ylabel('Degree '+str(degree)+' B-splines $B_{(i,'+str(degree)+')}$')

ylims = plt.gca().get_ylim()
plt.gca().set_ylim([0, ylims[1]*1.2])

xlims = plt.gca().get_xlim()
plt.gca().set_xlim([xlims[0]-0.5, xlims[1]+0.5])

plt.title(r"$\boldsymbol{A}$: B-spline basis functions with monotonously increasing coefficients",loc="left")

# Now plot the cumulative function
plt.subplot(gs[2,0])

# Original basis evaluations
evals = np.zeros(len(x))[:-1]

for i in range(n_basis):
    
    plt.fill_between(
        x = x[:-1],
        y1 = evals + B[(degree,i)][:-1]*c[i],
        y2 = evals,
        color = cmap(i/(n_basis-1)),
        alpha = 0.5)
    
    if i == n_basis-1:
        plt.plot(
            x[:-1],
            evals + B[(degree,i)][:-1]*c[i],
            color = "xkcd:dark grey")
    
    evals += B[(degree,i)][:-1]*c[i]
    
plt.gca().set_xticks([0,1,2,3,4,5,6])
if p != degree:
    plt.gca().set_xticklabels([])
else:
    plt.xlabel('knot positions')
    plt.gca().set_xticklabels(["$t_{0}$ - $t_{3}$","$t_{4}$","$t_{5}$","$t_{6}$","$t_{7}$","$t_{8}$","$t_{9}$ - $t_{12}$"])
plt.ylabel("monotone superposition")

ylims = plt.gca().get_ylim()
plt.gca().set_ylim([0, ylims[1]*1.2])

s_left = (evals[1] - evals[0])/(x[1] - x[0])

plt.plot(
    [-1,0],
    [-1*s_left + evals[0],evals[0]],
    color = "xkcd:dark grey",
    ls = "--")

s_right = (evals[-2] - evals[-3])/(x[-2] - x[-3])

plt.plot(
    [x[-2],x[-2]+1],
    [evals[-2],evals[-2] + 1*s_right],
    color = "xkcd:dark grey",
    ls = "--")

plt.gca().set_xlim([xlims[0]-0.5, xlims[1]+0.5])

ylims = plt.gca().get_ylim()

plt.plot(
    [0,0],
    ylims,
    color = "xkcd:grey",
    ls = ":",
    zorder = -1)

plt.plot(
    [6,6],
    ylims,
    color = "xkcd:grey",
    ls = ":",
    zorder = -1)

plt.fill_between(
    x[:-1],
    y1 = evals,
    y2 = ylims[1],
    color = "xkcd:grey",
    alpha = 0.25)

plt.text(
    0.5,
    0.9,
    "B-spline support",
    ha = "center",
    va = "center",
    color = "xkcd:dark grey",
    transform = plt.gca().transAxes)


plt.text(
    0.05,
    0.55,
    "linear extrapolation",
    rotation = 90,
    ha = "center",
    va = "center",
    color = "xkcd:dark grey",
    transform = plt.gca().transAxes,
    zorder = -2)
plt.text(
    0.95,
    0.425,
    "linear extrapolation",
    rotation = 90,
    ha = "center",
    va = "center",
    color = "xkcd:dark grey",
    transform = plt.gca().transAxes,
    zorder = -2)

plt.title(r"$\boldsymbol{B}$: Resulting monotone function",loc="left")

plt.savefig("B_splines_monotone.png",dpi=300,bbox_inches="tight")
plt.savefig("B_splines_monotone.pdf",dpi=300,bbox_inches="tight")