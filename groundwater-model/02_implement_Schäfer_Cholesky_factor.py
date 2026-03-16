import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import pickle
import os

root_directory = os.path.dirname(os.path.realpath(__file__))

def boundary_dist(row, col, nrows, ncols):
    """
    Distance from cell (r,c) to the boundary of a nrows x ncols grid.
    If the grid is [0..nrows-1] x [0..ncols-1],
    the distance to boundary is the minimum distance to any edge.
    """
    return min(row, nrows - 1 - row, col, ncols - 1 - col) + 1

def get_grid_dictionary(nrows, ncols):
    """
    Returns an array 'points' of shape (N,2) giving the (r,c) coords
    of each cell in row-major order. Also returns a dict 'rc_to_i'
    mapping (r,c) -> flat_index.
    """
    vec_to_mat = {}
    mat_to_vec = {}
    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            vec_to_mat[idx] = (row,col)
            mat_to_vec[(row,col)] = idx
            idx += 1
    return vec_to_mat, mat_to_vec

def maximin_selection(nrows, ncols, obs_indices=None):
    
    """
    Select cells on an nrows × ncols grid so that at every step the next cell
    maximises the minimum of
        – its distance to the current selection S
        – its distance to the grid boundary
    (i.e. it stays as far as possible from both).

    If `obs_indices` (row/col pairs) are supplied, the algorithm first consumes
    those cells (in the optimal order above) and then continues with the
    remaining cells.

    Returns
    -------
    S : list[int]
        Vectorised indices in the order chosen.
    min_scores : list[float]
        The score (= min{dist-to-S, dist-to-boundary}) at the moment each cell
        was added.
    points : ndarray (N, 2)
        Row/col coordinates for every vectorised index.
    """
    
    def boundary_dist(row, col):
        return min(row, nrows - 1 - row, col, ncols - 1 - col) + 1

    vec_to_mat, mat_to_vec = get_grid_dictionary(nrows, ncols)
    C = nrows * ncols
    points = np.asarray([vec_to_mat[i] for i in range(C)])
    bdist = np.fromiter(
        (boundary_dist(r, c) for r, c in points), dtype=float, count=C
    )

    # -------------------------------------------------------------------------
    # 1. choose first cell
    if obs_indices is None or len(obs_indices) == 0:
        first_idx = int(np.argmax(bdist))
        S = [first_idx]
        min_scores = [bdist[first_idx]]
    else:
        S = []
        min_scores = []
    
    def pick_next(candidates, tree_S):
        # distance to already-selected cells
        if len(S) == 0:
            d_to_S = np.full(len(candidates), np.inf)
        else:
            d_to_S, _ = tree_S.query(points[candidates], k=1)
        scores = np.minimum(d_to_S, bdist[candidates])
        best_pos = int(np.argmax(scores))
        return candidates[best_pos], scores[best_pos]

    # -------------------------------------------------------------------------
    # 2. phase 1 – only the observed cells (if any)
    remaining_obs = (
        []
        if obs_indices is None
        else [mat_to_vec[tuple(rc)] for rc in obs_indices if mat_to_vec[tuple(rc)] not in S]
    )

    tree_S = KDTree(points[S])

    while remaining_obs:
        cand = np.asarray(remaining_obs)
        best_idx, best_score = pick_next(cand, tree_S)
        S.append(best_idx)
        min_scores.append(best_score)
        remaining_obs.remove(best_idx)
        tree_S = KDTree(points[S])  # update tree

    # -------------------------------------------------------------------------
    # 3. phase 2 – all other cells
    remaining = [i for i in range(C) if i not in S]
    while remaining:
        cand = np.asarray(remaining)
        best_idx, best_score = pick_next(cand, tree_S)
        S.append(best_idx)
        min_scores.append(best_score)
        remaining.remove(best_idx)
        tree_S = KDTree(points[S])

    return S, min_scores, points


def collect_neighbors(S, distvals, grid_coords, rho=1.0):
    
    """
    This function processes the list S backwards, one cell at a time.
    For each last cell in S:
      1) Let threshold = distvals[ (current length of S)-1 ] * rho
      2) Among the cells "above" it in S (i.e. all but the last),
         find those whose distance is <= threshold
      3) Append those "above" neighbors to the *start* of a local list
         for that cell
      4) Remove the last cell from S
    Continue until S is empty.

    Args:
      S: list of chosen cell indices in some order (length N)
      distvals: list of length N, distvals[i] is the "distance" associated w/
                S[i] (same indexing)
      grid_coords: array of shape (total_cells, 2), grid_coords[i] gives row,col
                   for cell index i
      rho: scalar to scale distvals

    Returns:
      all_neighbors: A list of length N of neighbor-lists. 
         all_neighbors[0] is for the *initial* last cell in S,
         all_neighbors[1] is for the next last cell, etc.
         So if you popped 5 from S first, the neighbors of cell 5 end up in
         all_neighbors[0].
    """
    # Make a copy so original S isn't destroyed
    S_copy = S[:]

    all_neighbors = []

    while S_copy:
        
        # The last cell in the current S_copy
        current_idx = S_copy[-1]
        
        # The corresponding entry in distvals is distvals[len(S_copy)-1]
        threshold = distvals[len(S_copy)-1] * rho

        # Coordinates of the current cell
        r_cur, c_cur = grid_coords[current_idx]

        # We'll collect neighbors in a local list
        # (appending them to the *front* as specified)
        current_neighbors = []

        # Explore the "above" cells, i.e. everything except the last in S_copy
        above_cells = S_copy[:-1]

        for above_idx in above_cells:
            r_a, c_a = grid_coords[above_idx]
            dist = np.hypot(r_a - r_cur, c_a - c_cur)
            if dist <= threshold:
                # Insert at the *start* of the list
                current_neighbors.insert(0, above_idx)

        # Store this cell's neighbor list in all_neighbors
        all_neighbors.append(current_neighbors)

        # Remove the last cell in S_copy
        S_copy.pop()
        
    # Okay, the order I created is the wrong way around
    all_neighbors.reverse()

    return all_neighbors

def find_Cholesky_of_precision_matrix(X, S, neighbors):
    
    # Create a copy of X
    X = copy.copy(X)
    
    S_reverse = copy.copy(S)
    S_reverse.reverse()
    
    assert len(X.shape) == 2, "X should be an array of shape N-by-D, where N is the ensemble size and D is the number of cells."
    
    # Get the dimensions of the prior samples
    D       = X.shape[1]
    
    # Pre-allocate space for the Cholesky of the precision matrix
    L = np.zeros((D,D))
    
    for idx,node in tqdm(enumerate(S), desc="Evaluating Cholesky"):
        
        # Which nodes are relevant
        node_set = [node] + neighbors[idx]
        
        # Get the relevant X entries
        X_local = X[:,node_set]
        
        # Get the precision of the node_set
        cov_local = np.cov(X_local.T)
        if X_local.shape[1] == 1:
            cov_local = np.ones((1,1))*cov_local
        prec_local = np.linalg.inv(cov_local)
        
        # Get the e vector
        e       = np.zeros((len(node_set),1))
        e[0,0]  = 1
    
        # Use theorem 2.1
        L_vec = prec_local@e / np.sqrt(e.T@prec_local@e)
        
        # Columns in Cholesky
        row_in_Cholesky = [S_reverse.index(subnode) for subnode in node_set]
        
        # Save that as a lower-triangular matrix
        L[row_in_Cholesky,S_reverse.index(node)] = L_vec[:,0]
    
    return L

N = 101

h       = []
conc    = []
log_hk  = []

for n in range(N):
    
    dct     = pickle.load(open(root_directory+"\\"+"models"+"\\"+"results_"+str(n).zfill(4)+".p","rb"))
    
    h       .append(dct["h"])
    conc    .append(dct["conc"])
    log_hk  .append(dct["log_hk"])
    
h       = np.asarray(h)
conc    = np.asarray(conc)
log_hk  = np.asarray(log_hk)

print("Finished loading simulation results.")

plt.figure()

obs_indices = [
    [10,10],
    [25,10],
    [40,10],
    [10,40],
    [25,40],
    [40,40] ]



#%%

print("Set up maximin selection")

# How many rows and columns do we have?    
nrows, ncols = h.shape[1], h.shape[2]

S, distvals, grid_coords = maximin_selection(nrows, ncols, obs_indices = obs_indices)

print("Collect all neighbours")
neighbors = collect_neighbors(S, distvals, grid_coords, rho=2.0)

# Get the correctly ordered obs_indices
obs_indices = [list(grid_coords[idx,:]) for idx in S[:len(obs_indices)]]

print("Reshape ensemble array")

# Stack the layers we are updating
X = h.reshape((N,int(nrows*ncols)))
log_hk = log_hk.reshape((N,int(nrows*ncols)))

Cholesky_dictionary = {
    "S" : S,
    "neighbors" : neighbors,
    "X"     : X,
    "log_hk" : log_hk,
    "obs_indices" : obs_indices,
    "distvals" : distvals,
    "grid_coords" : grid_coords
    }

pickle.dump(Cholesky_dictionary,open("Cholesky_dictionary.p","wb"))