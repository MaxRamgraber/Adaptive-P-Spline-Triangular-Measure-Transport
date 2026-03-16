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
        
plt.figure(figsize=(12,12))
gs = GridSpec(
    nrows   = 5,
    ncols   = 1,
    height_ratios = [0.1,1,1,1,1])

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

for p in range(degree+1):
    
    plt.subplot(gs[1+p,0])
    for i in range(n_basis):
        plt.plot(
            x[:-1],
            B[(p,i)][:-1],
            color = cmap(i/(n_basis-1)))
        plt.fill_between(
            x = x[:-1],
            y1 = B[(p,i)][:-1],
            color = cmap(i/(n_basis-1)),
            alpha = 0.5)
        
    plt.gca().set_xticks([0,1,2,3,4,5,6])
    if p != degree:
        plt.gca().set_xticklabels([])
    else:
        plt.xlabel('knot positions')
        plt.gca().set_xticklabels(["$t_{0}$ - $t_{3}$","$t_{4}$","$t_{5}$","$t_{6}$","$t_{7}$","$t_{8}$","$t_{9}$ - $t_{12}$"])
    plt.ylabel('Degree '+str(p)+' B-splines $B_{(i,'+str(p)+')}$')

plt.savefig("B_splines.png",dpi=300,bbox_inches="tight")
plt.savefig("B_splines.pdf",dpi=300,bbox_inches="tight")