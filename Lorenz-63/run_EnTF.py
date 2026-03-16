import numpy as np
import matplotlib.pyplot as plt
import jax
import scipy.stats
import copy
import os
import time
import pickle
import argparse
import traceback
from PSpline_triangular_transport import adaptive_spline_transport

def main(seed):
    

    # Clear previous figures and pickles
    root_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Lorenz-63 dynamics
    def lorenz_dynamics(t, Z, beta=8/3, rho=28, sigma=10):
        
        if len(Z.shape) == 1: # Only one particle
        
            dZ1ds   = - sigma*Z[0] + sigma*Z[1]
            dZ2ds   = - Z[0]*Z[2] + rho*Z[0] - Z[1]
            dZ3ds   = Z[0]*Z[1] - beta*Z[2]
            
            dyn     = np.asarray([dZ1ds, dZ2ds, dZ3ds])
            
        else:
            
            dZ1ds   = - sigma*Z[...,0] + sigma*Z[...,1]
            dZ2ds   = - Z[...,0]*Z[...,2] + rho*Z[...,0] - Z[...,1]
            dZ3ds   = Z[...,0]*Z[...,1] - beta*Z[...,2]
    
            dyn     = np.column_stack((dZ1ds, dZ2ds, dZ3ds))
    
        return dyn
    
    # Fourth-order Runge-Kutta scheme
    def rk4(Z,fun,t=0,dt=1,nt=1):#(x0, y0, x, h):
        
        """
        Parameters
            t       : initial time
            Z       : initial states
            fun     : function to be integrated
            dt      : time step length
            nt      : number of time steps
        
        """
        
        # Prepare array for use
        if len(Z.shape) == 1: # We have only one particle, convert it to correct format
            Z       = Z[np.newaxis,:]
            
        # Go through all time steps
        for i in range(nt):
            
            # Calculate the RK4 values
            k1  = fun(t + i*dt,           Z);
            k2  = fun(t + i*dt + 0.5*dt,  Z + dt/2*k1);
            k3  = fun(t + i*dt + 0.5*dt,  Z + dt/2*k2);
            k4  = fun(t + i*dt + dt,      Z + dt*k3);
        
            # Update next value
            Z   += dt/6*(k1 + 2*k2 + 2*k3 + k4)
        
        return Z
    
    # This function calculates the continuous rank probability score
    def crps_ens(forecasts,observations):
    
        """
        Calculates the continuos rank probability score based on a N-by-O ensemble
        of forecasts, and an O vector of observations.
        """
        
        import numpy as np
        import copy
        
        # Create local copies of the variables
        observations    = copy.copy(observations)
        forecasts       = copy.copy(forecasts)
        
        # Convert the observations into a vector
        if np.isscalar(observations):
            observations    = np.asarray([observations])
            
        # Makes sure forecasts are of correct size
        if len(forecasts.shape) < 2:
            forecasts       = forecasts[:,np.newaxis]
        elif len(forecasts.shape) > 2:
            raise ValueError("'forecasts' has to be a N-by-O matrix; current shape: "+str(forecasts.shape))
        
        # Get the ensemble size and number of observations
        N       = forecasts.shape[0]
        O       = forecasts.shape[1]
        
        # Sort the forecasts
        forecasts = np.sort(forecasts, axis=0)
        
        # Pre-allocate a list for the results
        result  = np.zeros(O)
        
        # Now go through all observations
        for o in range(O):
    
            # Initialize basic variables
            obs_cdf         = 0     # Starting value of observation cdf
            forecast_cdf    = 0     # Starting value of forecast cdf
            prev_forecast   = 0     # Previous forecast value
            integral        = 0     # Integral so far
        
            # Go through all samples
            for n, forecast in enumerate(forecasts[:,o]):
                
                # The first time we pass the observation, do this
                if obs_cdf == 0 and observations[o] < forecast:
                    
                    integral += (observations[o] - prev_forecast) * forecast_cdf ** 2
                    integral += (forecast - observations[o]) * (forecast_cdf - 1) ** 2
                    obs_cdf = 1     # The cdf jumps to 1 once we pass it
                
                else:
                    
                    # Add the mismatch between forecast and obs cdf squared
                    integral += ((forecast - prev_forecast)
                                  * (forecast_cdf - obs_cdf) ** 2)
        
                # Incremend the forecast cdf
                forecast_cdf += 1/N
                
                # Current foreacst becomes previous forecast
                prev_forecast = forecast
        
            if obs_cdf == 0:
                # forecast can be undefined here if the loop body is never executed
                # (because forecasts have size 0), but don't worry about that because
                # we want to raise an error in that case, anyways
                integral += observations[o] - forecast
        
            # Save into results
            result[o] = integral
            
        return result
            
    # For the spinup, we use a stochastic EnKF
    def stochastic_EnKF(X,y,R,H):
        
        """
        This function implements a stochastic EnKF update. It requires the follow-
        ing variables:
            
            X       - an N-by-D array of samples, where N is the ensemble size, and
                      D is the dimensionality of state space
            y       - a vector of length O containing the observations made
            R       - the O-by-O observation error covariance matrix
            H       - an O-by-D observation operator
        """
        
        # Get the number of particles
        N       = X.shape[0]
        
        # Get the state covariance matrix
        C       = np.cov(X.T)   # We need the transpose of X
        
        # Calculate the Kalman gain
        K       = np.linalg.multi_dot((
            C,
            H.T,
            np.linalg.inv(
                np.linalg.multi_dot((
                    H,
                    C,
                    H.T)) + R)))
        
        # Draw observation error realizations
        v       = scipy.stats.multivariate_normal.rvs(
            mean    = np.zeros(R.shape[0]),
            cov     = R,
            size    = N)
        
        # Perturb the observations
        obs     = y[np.newaxis,:] + v
        
        # Apply the stochastic Kalman update
        X += np.dot(
            K,
            obs.T - np.dot(H,X.T) ).T
        
        return X
        
    # -------------------------------------------------------------------------
    # Set up exercise
    # -------------------------------------------------------------------------
    
    # Define problem dimensions
    O                   = 3 # Observation space dimensions
    D                   = 3 # State space dimensions
    
    # Ensemble size
    Ns                  = [50,100,175,250,375,500,750,1000]
    
    # Set up time
    T                   = 1000  # Full time series length
    T_spinup            = 1000  # EnKF spinup period
    
    # Time step length
    dt                  = 0.1   # Time step length
    dti                 = 0.05  # Time step increment
    
    # Observation error
    obs_sd              = 2
    R                   = np.identity(O)*obs_sd**2
    
    # Random seeds for repeat simulations
    random_seeds        = [seed]
        
    #%%
    
    # =========================================================================
    # Start simulating
    # =========================================================================
    
    for random_seed in random_seeds:
        
        # =========================================================================
        # Create or load observations and synthetic reference
        # =========================================================================
    
        # Reset the random seed
        np.random.seed(random_seed)
        
        # If we haven't precalculated observations and truth, do so.
        if "synthetic_truth_L63_RS="+str(random_seed)+".p" not in list(os.listdir(root_directory)) or \
           "observations_L63_RS="+str(random_seed)+".p" not in list(os.listdir(root_directory)):
        
            # Create the synthetic reference
            synthetic_truth         = np.zeros((T_spinup+T,1,D))
            synthetic_truth[0,0,:]  = scipy.stats.norm.rvs(size=3)
            
            for t in np.arange(0,T_spinup+T-1,1):
                 
                # Make a Lorenz forecast
                synthetic_truth[t+1,:,:] = rk4(
                    Z           = copy.copy(synthetic_truth[t,:,:]),
                    fun         = lorenz_dynamics,
                    t           = 0,
                    dt          = dti,
                    nt          = int(dt/dti))
                
            # Remove the unnecessary particle index
            synthetic_truth     = synthetic_truth[:,0,:]
                
            # Create observations
            observations        = copy.copy(synthetic_truth) + scipy.stats.norm.rvs(scale = obs_sd, size = synthetic_truth.shape)
            
        
            # Save the synthetic truth and observations
            pickle.dump(synthetic_truth,    open("synthetic_truth_L63_RS="+str(random_seed)+".p","wb"))
            pickle.dump(observations,       open("observations_L63_RS="+str(random_seed)+".p","wb"))
        
        else:
            
            # If we already have synthetic observations and truth in storage, retrieve them
            synthetic_truth     = pickle.load(open("synthetic_truth_L63_RS="+str(random_seed)+".p","rb"))
            observations        = pickle.load(open("observations_L63_RS="+str(random_seed)+".p","rb"))
            
        # Go through all ensemble sizes
        for ni,N in enumerate(Ns):
            
            mod_sd = 5*N**(-1)
            
            try:
                
                lambdas             = []
                lambdas_initial     = 2 # Later gets overwritten by averaging
                average_counter     = 0
                
                # Create an empty dictionary for the degrees of freedom
                dofs = {}
                dofs = np.zeros((T,3,4))
                
                # Reset the random seed
                np.random.seed(random_seed)
                
                # -------------------------------------------------
                # Prepare the stochastic map filtering
                # -------------------------------------------------
                
                # Load filtering results from storage, if available
                if "TM_filter_N="+str(N).zfill(4)+"_RS="+str(random_seed)+".p" in list(os.listdir(root_directory)):
                    
                    # If we compare linear and nonlinear smoothers,
                    # we have already filtered in the second pass
                    # In that case, just do nothing and skip to
                    # filtering
                    
                    # Have we already filtered before?
                    second_pass     = True
                    
                    pass
                    
                # If no filtering results are available, simulate
                else:
                    
                    # Have we already filtered before?
                    second_pass     = False
                    
                    print("Now filtering: N="+str(N)+" | RS="+str(random_seed),end="")
                    
                    file = open("log.txt","a")
                    file.write("Now filtering: N="+str(N)+" | RS="+str(random_seed))
                    file.close()
                    
                    # =========================================
                    # Simulate the spinup period
                    # =========================================
                    
                    # If spinup results are available, load them                                    
                    if "Z0_N="+str(N).zfill(4)+"_RS="+str(random_seed)+".p" in list(os.listdir(root_directory)):
                        
                        # Load the spinup results
                        Z0   = pickle.load(
                            open(
                                "Z0_N="+str(N).zfill(4)+"_RS="+str(random_seed)+".p",
                                "rb"))
                        
                    # If not, simulate them
                    else:
                
                        # Initiate particles from a standard Gaussian
                        Z0          = np.zeros((T_spinup+1,N,D))
                        Z0[0,...]   = scipy.stats.norm.rvs(size=(N,D))
                        
                        # Create the observation operator
                        H           = np.identity(O)
                            
                        # Go through the spinup period
                        for t in np.arange(0,T_spinup+1,1):
                            
                            # Stochastic EnKF update
                            Z0[t,...] = stochastic_EnKF(
                                X       = copy.copy(Z0[t,...]),
                                y       = copy.copy(observations[t,:]),
                                R       = R,
                                H       = H)
    
                            # After the analysis step, make a forecast to the next timestep
                            if t < T_spinup:
                                
                                # Make a Lorenz forecast
                                Z0[t+1,:,:] = rk4(
                                    Z           = copy.copy(Z0[t,:,:]),
                                    fun         = lorenz_dynamics,
                                    t           = 0,
                                    dt          = dti,
                                    nt          = int(dt/dti))
                                
                        # Store the spinup samples
                        pickle.dump(Z0,
                            open(
                                "Z0_N="+str(N).zfill(4)+"_RS="+str(random_seed)+".p",
                                "wb"))
                    
                    #%%
                    
                    # =========================================
                    # Prepare the stochastic map filtering
                    # =========================================
                    
                    # Reset the random seed
                    np.random.seed(random_seed)
                    
                    # Initialize the filtering samples from the
                    # spinup dataset
                    Z_a         = np.zeros((T,N,D))
                    Z_a[0,:,:]  = copy.copy(Z0[-1,...])
                    
                    # Then delete the spinup data set; we can 
                    # re-load it later
                    del Z0
                    
                    # Initialize the array for the forecast
                    Z_f         = copy.copy(Z_a)
                    
                    # Define the subfolder and filename
                    subfolder = "memory_RS="+str(random_seed).zfill(2)
                    
                    # Ensure the subfolder exists
                    os.makedirs(subfolder, exist_ok=True)
                    pickle.dump(Z_a,open(os.path.join(subfolder, "memory_Z_0.p"),"wb"))
                    
                    with open(os.path.join(subfolder, "log_N="+str(N).zfill(4)+".txt"), "a") as log_file:
                        log_file.write("Now filtering: N="+str(N)+" | RS="+str(random_seed)+"\n")
                    
                    
                    # Initialize the list for the RMSE and CRPS
                    RMSE_list           = []
                    CRPS_list           = []
                    
                    # Start time measurement for the filtering
                    time_begin  = time.time()
                
                    # Start the filtering
                    for t in np.arange(0,T,1):
                        
                        # # Create a new entry in the dof dictionary
                        # dofs[t] = {}
                        
                        # Copy the forecast into the analysis matrix
                        Z_a[t,:,:]  = copy.copy(Z_f[t,:,:])
                    
                        # Assimilate the observations one at a time
                        for idx,perm in enumerate([[0,1,2],[1,0,2],[2,1,0]]):
                            
                            # Simulate observations
                            Y_sim = copy.copy(Z_a[t,:,:][:,idx]) + \
                                scipy.stats.norm.rvs(
                                    loc     = 0,
                                    scale   = obs_sd,
                                    size    = Z_a[t,:,:][:,idx].shape)
                                
                            # Create the uninflated map input
                            map_input = copy.copy(np.column_stack((
                                Y_sim[:,np.newaxis],   # First dimension: simulated observation
                                Z_a[t,:,:][:,perm])))           # Next D dimensions: predicted states
                            
                            sparsity = np.tri(4)
                            sparsity[2:,0] = 0
                            
                            if 'transport_map' not in locals() and 'transport_map' not in globals():
                                transport_map = adaptive_spline_transport(skip_dimensions = 1)
                            
                            if t < 10:
                                optimize_lambdas = True
                                lmbd_init = 0
                            else:
                                optimize_lambdas = False
                                if t == 10:
                                    lmbd_init = {key: np.median(lambdas[key],axis=0) for key in list(lambdas.keys())}
                            
                            # Extract the lambdas
                            transport_map.train_map(
                                X = map_input, 
                                sparsity = sparsity[1:,:],
                                lambda_initial = lmbd_init, #0, #lambdas_initial,
                                optimize_lambdas = optimize_lambdas)
                            
                            # Save the effective degrees of freedom
                            dofs[t,idx,:] = copy.deepcopy(transport_map.dofs)
                            
                            # Extract the lambdas
                            if average_counter == 0:
                                lambdas = copy.deepcopy(transport_map.lambdas)
                                for key in list(lambdas.keys()):
                                    lambdas[key] = lambdas[key][None,:]
                            else:
                                for key in list(lambdas_initial.keys()):
                                    lambdas[key] = np.vstack((lambdas[key],copy.copy(transport_map.lambdas[key])))
                                    
                            
                            # Set the next initial lambdas_value
                            if average_counter == 0:
                                lambdas_initial = copy.deepcopy(transport_map.lambdas)
                            else:
                                for key in list(lambdas_initial.keys()):
                                    lambdas_initial[key] = (average_counter*lambdas_initial[key] + transport_map.lambdas[key])/(average_counter + 1)
                            
                            # Increment the average counter
                            average_counter += 1
                            
                            Z = transport_map.apply_forward_map(map_input)
                            
                            # Create an array with replicated of the observations
                            X_star = np.repeat(
                                a       = observations[T_spinup+t,idx].reshape((1,1)),
                                repeats = N, 
                                axis    = 0)
                            
                            ret = transport_map.apply_conditional_inverse_map(X_star,Z)
                            ret = ret[:,1:]
                            
                            # Undo the permutation of the states
                            ret = ret[:,perm]
                            
                            
                            # Save the result in the analysis array
                            Z_a[t,...]  = copy.copy(ret)
                            
                        # Calculate ensemble mean RMSE
                        RMSE = (np.mean(Z_a[t,...],axis=0) - synthetic_truth[T_spinup+t,:])**2
                        RMSE = np.mean(RMSE)
                        RMSE = np.sqrt(RMSE)
                        RMSE_list.append(RMSE)
                        
                        print("time t="+str(t).zfill(4)+" | RMSE="+"{:.4f}".format(RMSE)+" | avg. RMSE="+"{:.4f}".format(np.mean(RMSE_list)), flush = True)
                        
                        with open(os.path.join(subfolder, "log_N="+str(N).zfill(4)+".txt"), "a") as log_file:
                            log_file.write("time t="+str(t).zfill(4)+" | RMSE="+"{:.4f}".format(RMSE)+" | avg. RMSE="+"{:.4f}".format(np.mean(RMSE_list))+"\n")
                        
                        if t % 10 == 0 or t == T-1:
                            
                            plt.figure(figsize=(8,6))
                            plt.plot(RMSE_list)
                            plt.xlabel("time steps")
                            plt.ylabel("RMSE")
                            plt.title("RMSE development for N="+str(N)+" | avg. RMSE="+str(np.mean(RMSE_list)))
                            plt.savefig(os.path.join(subfolder, "RMSE_development_N="+str(N).zfill(4)+".png"),bbox_inches="tight")
                            plt.close("all")
                            
                            plt.figure(figsize=(8,6))
                            for k,color in enumerate(["xkcd:orangish red","xkcd:goldenrod","xkcd:grass green","xkcd:cerulean"]):
                                for idx,marker in enumerate(["x","+","o"]):
                                    plt.plot(
                                        dofs[:t+1,idx,k],
                                        color = color,
                                        marker = marker,
                                        label = "DoF ($S_{"+str(k)+"}$ idx="+str(idx)+")")
                            plt.xlabel("time steps")
                            plt.ylabel("Effective degrees of freedom")
                            plt.legend()
                            plt.title("Degree of freedom development for N="+str(N))
                            plt.savefig(os.path.join(subfolder, "DoF_development_N="+str(N).zfill(4)+".png"),bbox_inches="tight")
                            plt.close("all")
                            
                        
                        # Calculate CRPS
                        CRPS = crps_ens(
                            Z_a[t,...],
                            synthetic_truth[T_spinup+t,:])
                        CRPS_list.append(CRPS)
    
                        # After the analysis step, make a forecast to the next timestep
                        if t < T-1:
                            
                            # Make a Lorenz forecast
                            Z_f[t+1,:,:] = rk4(
                                Z           = copy.copy(Z_a[t,:,:]),
                                fun         = lorenz_dynamics,
                                t           = 0,
                                dt          = dti,
                                nt          = int(dt/dti)) + \
                                scipy.stats.norm.rvs(
                                    loc     = 0,
                                    scale   = mod_sd,
                                    size    = Z_f[t+1,:,:].shape)
                                
                        del transport_map
                        jax.clear_caches()
                            
                # Stop the clock
                time_end    = time.time()
                
                # Store the results in the output dictionary
                output_dictionary                       = {}
                output_dictionary['Z_a']                = Z_a
                output_dictionary['Z_f']                = Z_f
                output_dictionary['lambdas']            = lambdas
                output_dictionary['dofs']               = dofs
                output_dictionary['RMSE_list']          = RMSE_list
                output_dictionary['duration']           = time_end-time_begin
                
                # Write the result dictionary to a file
                pickle.dump(output_dictionary,  open(
                    os.path.join(subfolder, "TM_filter_N="+str(N).zfill(4)+"_RS="+str(random_seed)+".p"),'wb'))
                
                del output_dictionary
       
                # Store the average RMSE of this run
                mat_RMSE     = np.mean(RMSE_list)
                
                if np.nanmin(mat_RMSE) == np.mean(RMSE_list):
                    
                    # Store the Z_a and Z_f
                    pickle.dump(Z_a,  open(
                        os.path.join(subfolder, "Z_a_N="+str(N).zfill(4)+"_opt.p"),'wb'))
                    pickle.dump(Z_f,  open(
                        os.path.join(subfolder, "Z_f_N="+str(N).zfill(4)+"_opt.p"),'wb'))
                    
            except Exception as e:
                
                print("An error occurred:", e)
                
                with open(os.path.join(subfolder, "log_N="+str(N).zfill(4)+".txt"), "a") as log_file:
                    log_file.write("Simulation terminated with error {}".format(traceback.format_exc())+"\n")
        
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Script with an optional random seed.")
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed value (default: 0)')
    args = parser.parse_args()
    
    # Pass the seed to the main function.
    main(args.seed)
