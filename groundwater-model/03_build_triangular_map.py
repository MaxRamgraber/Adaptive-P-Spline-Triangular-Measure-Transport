import numpy as np
import scipy.stats
import copy
import os
import pickle
import scipy.special
from PSpline_triangular_transport import adaptive_spline_transport
import jax
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================================================================
# Parallel worker: train one map component and write results_dict_k=....
# =====================================================================

def _optimize_one_component(args):
    
    idx, k, num_obs, ordered_neighbours, X_ordered, root_directory = args
    
    # Find the entries that make up this map component function
    entry_indices = list(np.arange(0,num_obs,1))
    entry_indices += list(np.asarray(ordered_neighbours[idx]) + num_obs) + [k]
    
    # We only need the last map component function
    skip_dimensions = len(entry_indices) - 1
    
    # Only extract the dimensions we need
    X_local = copy.copy(X_ordered[:,entry_indices])
    
    # Create and train the transport map
    transport_map = adaptive_spline_transport(
        skip_dimensions = skip_dimensions)
    
    transport_map.train_map(
        X = X_local, 
        lambda_initial = 2,
        optimize_lambdas = True)
    
    # Save everything needed to apply the map later without re-optimizing
    results_dict = {
        "entry_indices"       : entry_indices,
        "skip_dimensions"     : skip_dimensions,
        "k"                   : transport_map.k,
        "degree"              : transport_map.degree,
        "D"                   : transport_map.D,
        "lambdas"             : transport_map.lambdas,
        "betas"               : transport_map.betas,
        "scaler"              : transport_map.scaler,
        "Bx"                  : transport_map.Bx,
        "nbases"              : transport_map.nbases,
        "sparsity"            : transport_map.sparsity,
        "training_data_range" : transport_map.training_data_range
        }
    
    out_file = root_directory+"\\"+"map_components"+"\\"+"results_dict_k="+str(k).zfill(4)+".p"
    tmp_file = out_file + ".tmp"
    
    with open(tmp_file,"wb") as f:
        pickle.dump(results_dict, f, protocol = pickle.HIGHEST_PROTOCOL)
    
    os.replace(tmp_file, out_file)
    
    del transport_map
    jax.clear_caches()
    
    return k

if __name__ == "__main__":

    root_directory = os.path.dirname(os.path.realpath(__file__))
    
    np.random.seed(0)
    
    sigma_obs = 0.01
    
    Cholesky_dictionary = pickle.load(open("Cholesky_dictionary.p","rb"))
    
    S               = Cholesky_dictionary["S"]
    neighbors       = Cholesky_dictionary["neighbors"]
    X               = Cholesky_dictionary["X"]
    log_hk          = Cholesky_dictionary["log_hk"]
    obs_indices     = Cholesky_dictionary["obs_indices"]
    distvals        = Cholesky_dictionary["distvals"]
    grid_coords     = Cholesky_dictionary["grid_coords"]
    
    # How many observations do we have?
    num_obs = len(obs_indices)
    
    # Create a copy of X
    h = copy.copy(X)
    
    # Bring the neighbours into triangular structure
    reverse_order = np.argsort(S)
    ordered_neighbours = [[reverse_order[idx] for idx in ngbs] for ngbs in neighbors]
    
    # Re-order the vectorized states accordingly
    h_ordered = h[:,S]
    log_hk_ordered = log_hk[:,S]
    
    observed_index = 50
    
    # Take the 25th ensemble member as the truth
    truth = h_ordered[observed_index,:]
    
    # Extract the synthetic observations.
    y_star = truth[:num_obs] + scipy.stats.norm.rvs(scale = sigma_obs, size = num_obs)
    
    # Only keep the remainder
    log_hk_ordered = np.delete(log_hk_ordered, observed_index, axis=0)
    h_ordered = np.delete(h_ordered, observed_index, axis=0)
    
    # How many samples do we have?
    N = log_hk_ordered.shape[0]
    
    # Pad zero standard deviation variables to prevent numerical issues
    log_hk_ordered += scipy.stats.norm.rvs(scale = 1E-10, size = log_hk_ordered.shape)
    
    # Augment X_ordered with observation predictions
    log_hk_ordered = np.column_stack((
        h_ordered[:,:num_obs] + scipy.stats.norm.rvs(scale = sigma_obs, size = (N,num_obs)),
        log_hk_ordered))
    
    X_ordered = copy.copy(log_hk_ordered)
    
    
    # Pre-allocate space for X_star
    X_star = np.zeros_like(log_hk_ordered)*np.nan
    X_star[:,:num_obs] = y_star[None,:]
    
    # Pre-allocate space for X_star
    X_star_linear = np.zeros_like(log_hk_ordered)*np.nan
    X_star_linear[:,:num_obs] = y_star[None,:]
    
    # Run an EnKF for reference
    emp_cov_YX = np.cov(X_ordered.T)
    K = emp_cov_YX[num_obs:,:num_obs]@np.linalg.inv(emp_cov_YX[:num_obs,:num_obs])
    
    X_star_EnKS = copy.copy(X_ordered[:,num_obs:]).T - K@(copy.copy(X_ordered[:,:num_obs]).T - np.repeat(y_star[:,None],axis=1, repeats = N))
    X_star_EnKS = X_star_EnKS.T
    X_star_EnKS = np.column_stack((
        np.repeat(y_star[None,:],axis=0, repeats = N),
        X_star_EnKS))
    
    pickle.dump(X_star_EnKS,open("X_star_EnKS.p","wb"))
    
    # =====================================================================
    # Make output folder
    # =====================================================================
    
    if "map_components" not in os.listdir(root_directory):
        os.mkdir(root_directory+"\\"+"map_components")
    
    # =====================================================================
    # Parallel optimization (training only)
    # =====================================================================
    
    n_workers = max(1, os.cpu_count() - 1)
    
    task_list = []
    for idx,k in enumerate(np.arange(num_obs,X_ordered.shape[-1],1)):
        
        out_file = root_directory+"\\"+"map_components"+"\\"+"results_dict_k="+str(k).zfill(4)+".p"
        
        if not os.path.exists(out_file):
            task_list.append((idx, k, num_obs, ordered_neighbours, X_ordered, root_directory))
    
    if len(task_list) > 0:
        
        print("Optimizing {} map component functions in parallel".format(len(task_list)))
        
        with ProcessPoolExecutor(max_workers = n_workers) as pool:
            
            futures = [pool.submit(_optimize_one_component, task) for task in task_list]
            
            for fut in as_completed(futures):
                k_done = fut.result()
                print("Finished optimizing map component function S_"+str(k_done))
    
    # =====================================================================
    # Serial triangular update (use saved component state)
    # =====================================================================
    
    # Optional resume
    if os.path.exists("X_star.p"):
        X_star = pickle.load(open("X_star.p","rb"))
    if os.path.exists("X_star_linear.p"):
        X_star_linear = pickle.load(open("X_star_linear.p","rb"))
    
    for idx,k in enumerate(np.arange(num_obs,X_ordered.shape[-1],1)):
        
        # If already done, skip (resume support)
        if np.isfinite(X_star[0,k]):
            continue
        
        results_file = root_directory+"\\"+"map_components"+"\\"+"results_dict_k="+str(k).zfill(4)+".p"
        results_dict = pickle.load(open(results_file,"rb"))
        
        entry_indices   = results_dict["entry_indices"]
        skip_dimensions = results_dict["skip_dimensions"]
        
        print("Applying map component function S_"+str(k))
        
        # =====================================================================
        # Do a linear triangular map update
        # =====================================================================
        
        map_input = copy.copy(X_ordered[:,entry_indices])
        num_cond  = len(entry_indices) - 1
        
        emp_cov_YX = np.cov(map_input.T)
        K = emp_cov_YX[num_cond:,:num_cond]@np.linalg.inv(emp_cov_YX[:num_cond,:num_cond])
    
        X_star_linear[:,k] = (copy.copy(map_input[:,num_cond:]).T - K@(copy.copy(map_input[:,:num_cond]).T - copy.copy(X_star_linear[:,entry_indices[:-1]]).T)).T[:,0]
        
        # =====================================================================
        # Do a P-Spline triangular map update
        # =====================================================================
        
        transport_map = adaptive_spline_transport(
            skip_dimensions = skip_dimensions)
        
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
        
        # Apply the forward map (Z depends only on X_local + trained map)
        X_local = copy.copy(X_ordered[:,entry_indices])
        Z = transport_map.apply_forward_map(X_local)
        
        # Conditionally invert the map (serial triangular)
        X_star[:,k] = transport_map.apply_conditional_inverse_map(
            X_star  = copy.copy(X_star[:,entry_indices[:-1]]), 
            Z       = Z)[:,skip_dimensions:][:,0]
        
        # Print outputs
        print("Lambdas: {}".format(transport_map.lambdas))
        print("Average magnitude of update (EnTS): {:.2e}".format(np.mean(np.abs(X_star[:,k] - X_ordered[:,k]))))
        print("Average magnitude of update (EnTS linear): {:.2e}".format(np.mean(np.abs(X_star_linear[:,k] - X_ordered[:,k]))))
        print("Average magnitude of update (EnKS): {:.2e}".format(np.mean(np.abs(X_star_EnKS[:,k] - X_ordered[:,k]))))
        
        pickle.dump(X_star,open("X_star.p","wb"))
        pickle.dump(X_star_linear,open("X_star_linear.p","wb"))
        
        del transport_map
        jax.clear_caches()