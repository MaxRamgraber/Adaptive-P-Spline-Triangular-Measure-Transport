import os
import subprocess
import matplotlib.pyplot as plt
import flopy
import numpy as np
import config
import copy
import pickle

root_directory = os.path.dirname(os.path.realpath(__file__))

if "models_posterior" not in os.listdir(root_directory):
    os.mkdir(root_directory+"\\"+"models_posterior")

def build_mf6_flow_model(
    silent=False,
):
    if config.buildModel:
        print("Building mf6gwf model...")
        gwfname = "gwf-" + name
        sim_ws = os.path.join(ws, "mf6gwf")
        sim = flopy.mf6.MFSimulation(
            sim_name=gwfname, sim_ws=sim_ws, exe_name="mf6"#config.mf6_exe
        )

        # Instantiating MODFLOW 6 time discretization
        tdis_rc = []
        tdis_rc.append((perlen, 1, 1.0))
        flopy.mf6.ModflowTdis(
            sim, nper=1, perioddata=tdis_rc, time_units=time_units
        )

        # Instantiating MODFLOW 6 groundwater flow model
        gwf = flopy.mf6.ModflowGwf(
            sim,
            modelname=gwfname,
            save_flows=True,
            model_nam_file="{}.nam".format(gwfname),
        )

        # Instantiating MODFLOW 6 solver for flow model
        imsgwf = flopy.mf6.ModflowIms(
            sim,
            print_option="SUMMARY",
            outer_dvclose=hclose,
            outer_maximum=nouter,
            under_relaxation="NONE",
            inner_maximum=ninner,
            inner_dvclose=hclose,
            rcloserecord=rclose,
            linear_acceleration="CG",
            scaling_method="NONE",
            reordering_method="NONE",
            relaxation_factor=relax,
            filename="{}.ims".format(gwfname),
        )
        sim.register_ims_package(imsgwf, [gwf.name])

        # Instantiating MODFLOW 6 discretization package
        flopy.mf6.ModflowGwfdis(
            gwf,
            length_units=length_units,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            delr=delr,
            delc=delc,
            top=top,
            botm=botm,
            idomain=idomain,
            filename="{}.dis".format(gwfname),
        )

        # Instantiating MODFLOW 6 initial conditions package for flow model
        strt[:, :, -1] = constantheadright
        flopy.mf6.ModflowGwfic(
            gwf, strt=strt, filename="{}.ic".format(gwfname)
        )

        # Instantiating MODFLOW 6 node-property flow package
        flopy.mf6.ModflowGwfnpf(
            gwf,
            save_flows=True,
            k33overk=False,
            icelltype=laytyp,
            k=hk,
            k33=vk,
            save_specific_discharge=True,
            save_saturation=True,
            filename="{}.npf".format(gwfname),
        )
        
        # Recharge via stress_period_data
        rch_rate = 1.0e-8  # m/s
        rch_arr = np.full((nrow, ncol), rch_rate, dtype=float)

        # concentration carried by recharge water (use T0 or your own array)
        rch_conc_arr = np.full((nrow, ncol), T0, dtype=float)

        spd0 = []
        for i in range(nrow):
            for j in range(ncol):
                rate = float(rch_arr[i, j])
                if rate != 0.0:
                    #            (cellid), recharge, AUX CONCENTRATION
                    spd0.append(((0, i, j), rate, float(rch_conc_arr[i, j])))

        rch_spd = {0: spd0}  # single stress period

        flopy.mf6.ModflowGwfrch(
            gwf,
            save_flows=True,
            stress_period_data=rch_spd,
            auxiliary=["CONCENTRATION"],
            pname="RCH-1",
            filename=f"{gwfname}.rch",
        )

        # Instantiate storage package
        flopy.mf6.ModflowGwfsto(
            gwf, 
            ss=0, 
            sy=0, 
            filename="{}.sto".format(gwfname),
            steady_state={0: True},
            transient={0: False},   
        )

        # Instantiating MODFLOW 6 constant head package
        # MF6 constant head boundaries:
        chdspd = []
        # Loop through the left & right sides for all layers.
        for k in range(nlay):
            for i in range(nrow):
                # right-most column:
                #              (l, r,      c),   head,                 conc
                chdspd.append([(k, i, ncol - 1), strt[k, i, ncol - 1], T0])

        chdspd = {0: chdspd}

        flopy.mf6.ModflowGwfchd(
            gwf,
            maxbound=len(chdspd),
            stress_period_data=chdspd,
            save_flows=False,
            auxiliary="CONCENTRATION",
            pname="CHD-1",
            filename="{}.chd".format(gwfname),
        )

        # Instantiating MODFLOW 6 output control package for flow model
        flopy.mf6.ModflowGwfoc(
            gwf,
            head_filerecord="{}.hds".format(gwfname),
            budget_filerecord="{}.bud".format(gwfname),
            saverecord=[
                ("HEAD", "ALL"),
                ("BUDGET", "ALL"),
            ],
            printrecord=[
                ("HEAD", "LAST"),
                ("BUDGET", "LAST"),
            ],
        )
        return sim
    return None
        
def build_mf6_transport_model(
    silent=False,
):
    if config.buildModel:
        # Instantiating MODFLOW 6 groundwater transport package
        print("Building mf6gwt model...")
        gwtname = "gwt-" + name
        sim_ws = os.path.join(ws, "mf6gwt")
        sim = flopy.mf6.MFSimulation(
            sim_name=gwtname, sim_ws=sim_ws, exe_name="mf6"#config.mf6_exe
        )

        # MF6 time discretization
        tdis_rc = [(perlen, nstp, 1.)]
        flopy.mf6.ModflowTdis(
            sim, nper=len(tdis_rc), perioddata=tdis_rc, time_units=time_units
        )

        gwtname = "gwt-" + name
        gwt = flopy.mf6.MFModel(
            sim,
            model_type="gwt6",
            modelname=gwtname,
            model_nam_file="{}.nam".format(gwtname),
        )
        gwt.name_file.save_flows = True

        # create iterative model solution and register the gwt model with it
        imsgwt = flopy.mf6.ModflowIms(
            sim,
            print_option="SUMMARY",
            outer_dvclose=hclose,
            outer_maximum=nouter,
            under_relaxation="NONE",
            inner_maximum=ninner,
            inner_dvclose=hclose,
            rcloserecord=rclose,
            linear_acceleration="BICGSTAB",
            scaling_method="NONE",
            reordering_method="NONE",
            relaxation_factor=relax,
            filename="{}.ims".format(gwtname),
        )
        sim.register_ims_package(imsgwt, [gwt.name])

        # Instantiating MODFLOW 6 transport discretization package
        flopy.mf6.ModflowGwtdis(
            gwt,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            delr=delr,
            delc=delc,
            top=top,
            botm=botm,
            idomain=idomain,
            filename="{}.dis".format(gwtname),
        )

        # Instantiating MODFLOW 6 transport initial concentrations
        flopy.mf6.ModflowGwtic(
            gwt, strt=sconc, filename="{}.ic".format(gwtname)
        )

        # Instantiating MODFLOW 6 transport advection package
        if mixelm >= 0:
            scheme = "UPSTREAM"
        elif mixelm == -1:
            scheme = "TVD"
        else:
            raise Exception()
        flopy.mf6.ModflowGwtadv(
            gwt, scheme=scheme, filename="{}.adv".format(gwtname)
        )

        # Instantiating MODFLOW 6 transport dispersion package
        if al != 0:
            flopy.mf6.ModflowGwtdsp(
                gwt,
                alh=al,
                ath1=ath1,
                atv=atv,
                diffc=dmcoef_arr,
                pname="DSP-1",
                filename="{}.dsp".format(gwtname),
            )

        # Instantiating MODFLOW 6 transport mass storage package
        flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=prsity,
            first_order_decay=False,
            decay=None,
            decay_sorbed=None,
            pname="MST-1",
            filename="{}.mst".format(gwtname),
        )

        # Instantiating MODFLOW 6 transport source-sink mixing package
        sourcerecarray = [("CHD-1", "AUX", "CONCENTRATION")]
        flopy.mf6.ModflowGwtssm(
            gwt,
            sources=sourcerecarray,
            print_flows=True,
            filename="{}.ssm".format(gwtname),
        )

        # 1) Define the CNC stress period data
        cnc_spd = {
            0: [
                (well_loc, Tinj),
                ((well_loc[0],well_loc[1]-1,well_loc[2]), Tinj),
                ((well_loc[0],well_loc[1]+1,well_loc[2]), Tinj),
                ((well_loc[0],well_loc[1],well_loc[2]-1), Tinj),
                ((well_loc[0],well_loc[1],well_loc[2]+1), Tinj)],  # steady value for the entire single stress period
        }
        
        # 2) Instantiate the CNC package
        flopy.mf6.ModflowGwtcnc(
            gwt,
            maxbound=len(cnc_spd[0]),
            stress_period_data=cnc_spd,
            save_flows=True,
            pname="CNC-1",
            filename=f"{gwtname}.cnc",
        )

        # Instantiating MODFLOW 6 Flow-Model Interface package
        flow_name = gwtname.replace("gwt", "gwf")
        pd = [
            ("GWFHEAD", "../mf6gwf/" + flow_name + ".hds", None),
            ("GWFBUDGET", "../mf6gwf/" + flow_name + ".bud", None),
        ]
        flopy.mf6.ModflowGwtfmi(gwt, packagedata=pd)

        # Instantiating MODFLOW 6 transport output control package
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord="{}.cbc".format(gwtname),
            concentration_filerecord="{}.ucn".format(gwtname),
            saverecord=[
                ("CONCENTRATION", "ALL"),
                ("BUDGET", "LAST"),
            ],
            printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
            filename="{}.oc".format(gwtname),
        )

        return sim
    return None

def write_mf6_models(sim_mf6gwf, sim_mf6gwt, silent=True):
    if config.writeModel:
        sim_mf6gwf.write_simulation(silent=silent)
        sim_mf6gwt.write_simulation(silent=silent)
        
@config.timeit
def run_model(sim_mf6gwf, sim_mf6gwt, silent=False):
    success = True
    if config.runModel:

        success, buff = sim_mf6gwf.run_simulation(silent=silent)
        success, buff = sim_mf6gwt.run_simulation(silent=silent)
        if not success:
            print(buff)
    return success

# Get the directory where the Python file is located
root_directory = os.path.dirname(os.path.abspath(__file__))

np.random.seed(0)

plt.close("all")

figure_size = (5.5, 2.75)

length_units = "meters"
time_units = "seconds"

parameter_units = {
    "peclet": "$unitless$",
    "gradient": "$m/m$",
    "seepagevelocity": "$m/s$",
    "constantheadright": "$m$",
}

nlay = 1  # Number of layers
nrow = 51  # Number of rows
ncol = 51  # Number of columns

observed_index = 50

N = 100
nrows = 51
ncols = 51

X_star = pickle.load(open("X_star.p","rb"))
X_star_EnKS = pickle.load(open("X_star_EnKS.p","rb"))

ws = ''
name = "model"

length_units = "meters"
time_units = "seconds"

parameter_units = {
    "peclet": "$unitless$",
    "gradient": "$m/m$",
    "seepagevelocity": "$m/s$",
    "constantheadright": "$m$",
}

nlay = 1  # Number of layers
nrow = 51  # Number of rows
ncol = 51  # Number of columns

delr = 1.0  # Column width ($m$)
delc = 1.0  # Row width ($m$)
delz = 10.0  # Layer thickness ($m$)

width = ncol*delc  # Simulation width ($m$)
length = nrow*delr  # Simulation length ($m$)

constantheadleft = 10
constantheadright = 5

top = delz  # Top of the model ($m$)
satthk = 13.0  # Saturated thickness ($m$)
hk = 8.0e-3  # Horizontal hydraulic conductivity($m/s$)
hk_max  = -5
hk_min  = -7
vk = 8.0e-3  # Vertical hydraulic conductivity($m/s$)
T0 = 0  # Initial temperature of aquifer ($K$)
Tinj = 1
prsity = 0.26  # Porosity
al = 0.50  # Longitudinal dispersivity ($m$)
trpt = 1  # Ratio of horizontal transverse dispersivity to longitudinal dispersivity
trpv = 1  # Ratio of vertical transverse dispersivity to longitudinal dispersivity
rhob = 1961.0  # Aquifer bulk density ($kg/m^3$)
sp1 = 2.103e-4  # Distribution coefficient ($m^3/kg$)
perlen = 864000000 # Simulation time ($seconds$)

botm = [top - delz * k for k in range(1, nlay + 1)]
laytyp = icelltype = 0

# Starting Heads:
strt = np.ones((nlay, nrow, ncol), dtype=float) * 5.0

# Active model domain
ibound = np.ones((nlay, nrow, ncol), dtype=int)
ibound[:, :, -1] = -1  # eastern boundary

peclet = 1.0
gradient = 1.2e-4

idomain = 1

# Transport related
icbund = np.ones((nlay, nrow, ncol))
icbund[:, :, 0] = -1

# Starting concentrations:
sconc = T0

# Dispersion
ath1 = al * trpt
atv = al * trpv
dmcoef_arr = 0

# Time variables
nstp = 1000
transport_stp_len = 86400  # seconds simulated per transport step
ttsmult = 1.0

# Advection
mixelm = -1
percel = 1.0

well_loc = (0, 25, 10)

mf6_bhe = [[well_loc, 1E-2]]

# Reactive transport related terms
isothm = 1  # sorption type; 1=linear isotherm (equilibrium controlled)
sp2 = 2.0  # w/ isothm = 1 this is read but not used
rhob = 1.7  # g/cm^3
sp1 = 0.176  # cm^3/g  (Kd: "Distribution coefficient")

nouter, ninner = 100, 300
hclose, rclose, relax = 5e-5, 1e-8, 1.0

Cholesky_dictionary = pickle.load(open("Cholesky_dictionary.p","rb"))

obs_indices = Cholesky_dictionary["obs_indices"]
num_obs = len(obs_indices)

# Get the map ordering
S = Cholesky_dictionary["S"]
reverse_order = np.argsort(S)

#%%

X_star_EnKS_grid = X_star_EnKS[:,num_obs:][:,reverse_order].reshape((N,nrows,ncols))

for ensemble_member in np.arange(0,100,1):

    counter = 0
    
    while counter < 1:
        
        counter += 1
        
        try:
            
            if "posterior_simulation_EnKS_"+str(ensemble_member).zfill(4)+".p" not in list(os.listdir(root_directory+"\\"+"models_posterior")):
            
                x = range(ncol)
                y = range(nrow)
                
                log_hk = X_star_EnKS_grid[ensemble_member,:]
                
                # Convert it to hydraulic conductivity
                hk      = 10**log_hk
                
                # Build the models
                sim_mf6gwf = build_mf6_flow_model()
                sim_mf6gwt = build_mf6_transport_model()
                
                # Write the models
                write_mf6_models(sim_mf6gwf, sim_mf6gwt, silent=True)
                
                # Define the command and the working directory
                command = ["mf6.exe"]  # Replace with your command and arguments
                working_directory = os.path.join(root_directory,"mf6gwf")  # Replace with your target directory
                result = subprocess.run(command, cwd=working_directory, capture_output=True, text=True)
                print("Output:\n", result.stdout)
                print("Errors:\n", result.stderr)
                
                command = ["mf6.exe"]  # Replace with your command and arguments
                working_directory = os.path.join(root_directory,"mf6gwt")  # Replace with your target directory
                result = subprocess.run(command, cwd=working_directory, capture_output=True, text=True)
                print("Output:\n", result.stdout)
                print("Errors:\n", result.stderr)
                
                # Get GWT simulation
                gwt = sim_mf6gwt.get_model("gwt-" + name)
                gwf = sim_mf6gwf.get_model("gwf-" + name)
                
                # Read the hydraulic head
                h = gwf.output.head().get_data()
                
                # Read the concentrations
                conc = np.asarray([gwt.output.concentration().get_data(kstpkper=(kstp,0)) for kstp in range(nstp)])
                
                # Save the results
                results = {
                    "log_hk"    : copy.copy(log_hk),
                    "h"         : copy.copy(h)[0,...],
                    "conc"      : copy.copy(conc)[-1,0,...]}
            
                pickle.dump(results,open(root_directory+"\\"+"models_posterior"+"\\"+"posterior_simulation_EnKS_"+str(ensemble_member).zfill(4)+".p","wb"))
                
                
                # Plot the results
                fig, axes = plt.subplots(3, 1, figsize=(6, 9), constrained_layout=True)
                
                # first subplot: hydraulic head
                ax = axes[0]
                ax.set_title("hydraulic head")
                modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
                pa = modelmap.plot_array(h, vmin=np.min(h), vmax=np.max(h))
                quadmesh = modelmap.plot_bc("CHD")
                # linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
                contours = modelmap.contour_array(
                    h,
                    colors="black",
                )
                ax.clabel(contours, fmt="%2.1f")
                cb = plt.colorbar(pa, shrink=0.5, ax=ax)
                ax.axis("equal")
                
                # second subplot: concentration
                ax = axes[1]
                ax.set_title("concentration")
                modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
                pa = modelmap.plot_array(conc[-1,...], vmin=np.min(conc), vmax=np.max(conc))
                quadmesh = modelmap.plot_bc("CHD")
                ax.clabel(contours, fmt="%2.1f")
                cb = plt.colorbar(pa, shrink=0.5, ax=ax)
                ax.axis("equal")
                
                # second subplot: hydraulic conductivity
                ax = axes[2]
                ax.set_title("hydraulic conductivity")
                modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
                pa = modelmap.plot_array(log_hk, vmin=np.min(log_hk), vmax=np.max(log_hk))
                quadmesh = modelmap.plot_bc("CHD")
                ax.clabel(contours, fmt="%2.1f")
                cb = plt.colorbar(pa, shrink=0.5, ax=ax)
                ax.axis("equal")
                
                plt.savefig(root_directory+"\\"+"models_posterior"+"\\"+"img_EnKS_"+str(ensemble_member).zfill(4)+".png",bbox_inches="tight")
                plt.close("all")
            
            break
            
        except:
            
            pass

#%%

X_star_grid = X_star[:,num_obs:][:,reverse_order].reshape((N,nrows,ncols))

for ensemble_member in np.arange(0,100,1):


    counter = 0
    
    while counter < 1:

        counter += 1
        
        try:
            
            if "posterior_simulation_"+str(ensemble_member).zfill(4)+".p" not in list(os.listdir(root_directory+"\\"+"models_posterior")):
            
                x = range(ncol)
                y = range(nrow)
                
                
                log_hk = X_star_grid[ensemble_member,:]
                
                # Convert it to hydraulic conductivity
                hk      = 10**log_hk
                
                # Build the models
                sim_mf6gwf = build_mf6_flow_model()
                sim_mf6gwt = build_mf6_transport_model()
                
                # Write the models
                write_mf6_models(sim_mf6gwf, sim_mf6gwt, silent=True)
                
                # Define the command and the working directory
                command = [r"C:\WRDAPP\mf6.6.3_win64\bin\mf6.exe"]  # Replace with your command and arguments
                working_directory = os.path.join(root_directory,"mf6gwf")  # Replace with your target directory
                
                print("command: "+str(command))
                print(working_directory)
                result = subprocess.run(command, cwd=working_directory, capture_output=True, text=True)
                print("Output:\n", result.stdout)
                print("Errors:\n", result.stderr)
                
                command = [r"C:\WRDAPP\mf6.6.3_win64\bin\mf6.exe"]  # Replace with your command and arguments
                working_directory = os.path.join(root_directory,"mf6gwt")  # Replace with your target directory
                result = subprocess.run(command, cwd=working_directory, capture_output=True, text=True)
                print("Output:\n", result.stdout)
                print("Errors:\n", result.stderr)
                
                # Get GWT simulation
                gwt = sim_mf6gwt.get_model("gwt-" + name)
                gwf = sim_mf6gwf.get_model("gwf-" + name)
                
                # Read the hydraulic head
                h = gwf.output.head().get_data()
                
                # Read the concentrations
                conc = np.asarray([gwt.output.concentration().get_data(kstpkper=(kstp,0)) for kstp in range(nstp)])
                
                # Save the results
                results = {
                    "log_hk"    : copy.copy(log_hk),
                    "h"         : copy.copy(h)[0,...],
                    "conc"      : copy.copy(conc)[-1,0,...]}
            
                pickle.dump(results,open(root_directory+"\\"+"models_posterior"+"\\"+"posterior_simulation_"+str(ensemble_member).zfill(4)+".p","wb"))
                
                
                # Plot the results
                fig, axes = plt.subplots(3, 1, figsize=(6, 9), constrained_layout=True)
                
                # first subplot: hydraulic head
                ax = axes[0]
                ax.set_title("hydraulic head")
                modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
                pa = modelmap.plot_array(h, vmin=np.min(h), vmax=np.max(h))
                quadmesh = modelmap.plot_bc("CHD")
                # linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
                contours = modelmap.contour_array(
                    h,
                    colors="black",
                )
                ax.clabel(contours, fmt="%2.1f")
                cb = plt.colorbar(pa, shrink=0.5, ax=ax)
                ax.axis("equal")
                
                # second subplot: concentration
                ax = axes[1]
                ax.set_title("concentration")
                modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
                pa = modelmap.plot_array(conc[-1,...], vmin=np.min(conc), vmax=np.max(conc))
                quadmesh = modelmap.plot_bc("CHD")
                ax.clabel(contours, fmt="%2.1f")
                cb = plt.colorbar(pa, shrink=0.5, ax=ax)
                ax.axis("equal")
                
                # second subplot: hydraulic conductivity
                ax = axes[2]
                ax.set_title("hydraulic conductivity")
                modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
                pa = modelmap.plot_array(log_hk, vmin=np.min(log_hk), vmax=np.max(log_hk))
                quadmesh = modelmap.plot_bc("CHD")
                ax.clabel(contours, fmt="%2.1f")
                cb = plt.colorbar(pa, shrink=0.5, ax=ax)
                ax.axis("equal")
                
                plt.savefig(root_directory+"\\"+"models_posterior"+"\\"+"img_"+str(ensemble_member).zfill(4)+".png",bbox_inches="tight")
                plt.close("all")
            
            break
            
        except:
            
            print("Something went wrong.")
            
            pass