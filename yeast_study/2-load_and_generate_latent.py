import helper as hp
import numpy as np
from configparser import ConfigParser
from evostrat.init_mlp import MLP
from docker.work.latent_analysis.helpers.npy_to_hdf5_yeast import store_as_hdf5

# parse arguments from configfile
configs = ConfigParser()
configs.read("latent_analysis/latent_config.ini")

lnminkm = float(configs['CONSTRAINTS']['min_km'])
lnmaxkm = float(configs['CONSTRAINTS']['max_km'])

pf_flag = int(configs['PARAMETER_FIXING']['pf_flag'])
met_model = configs['PATHS']['met_model']

# Load parameter names
names_km = hp.load_pkl(configs['PATHS']['param_names'])

n_sets = int(configs['LOAD_AND_GENERATE']['n_sets'])
path_to_weights = configs['LOAD_AND_GENERATE']['path_to_weights']
output_name = configs['LOAD_AND_GENERATE']['name_output']
output_path = configs['LOAD_AND_GENERATE']['path_output']

ss_idx = int(configs['EVOSTRAT']['ss_idx'])
save_latent_vectors_bool = bool(int(configs['LOAD_AND_GENERATE']['save_latent_vectors']))
save_unscaled_bool = bool(int(configs['LOAD_AND_GENERATE']['save_unscaled']))


# -------------------------------------------------------------------------------------------------
                    # VERIFY ALL INPUT / OUTPUT PATHS
# -------------------------------------------------------------------------------------------------

import os
import sys
print("\n=== PATH CHECK ===")

# 1. Check current working directory
print("Working directory:", os.getcwd())

# 2. Check weight file
if not os.path.exists(path_to_weights):
    sys.exit(f"ERROR: weights file not found → {path_to_weights}")
else:
    print(f"Found weights file: {path_to_weights}")

# 3. Check models folder
model_path = f"models/{met_model}"
if not os.path.exists(model_path):
    sys.exit(f"ERROR: model folder not found → {model_path}")
else:
    print(f"Found model folder: {model_path}")

# 4. Ensure output directory exists
os.makedirs(output_path, exist_ok=True)
print(f"Output path ready: {output_path}")

print("===================\n")

# ----------------------------------------------------------------------
                            # MAIN SCRIPT
# ----------------------------------------------------------------------


# Call neural network agent
cond_class = 1
mlp = MLP(cond_class, lnminkm, lnmaxkm, n_sets, names_km, param_fixing=pf_flag)

# Load saved weights
opt_weights = hp.load_pkl(path_to_weights)
mlp.generator.set_weights(opt_weights)

# Generate parameters (optionally store latent vector)
if save_latent_vectors_bool:
    if save_unscaled_bool:
        gen_params, gen_params_unscaled, latent_vector = mlp.sample_parameters(return_latent=True, return_unscaled=True)
        np.save(f"{output_path}{output_name}_latent.npy", latent_vector)
        np.save(f"{output_path}{output_name}_unscaled.npy", gen_params_unscaled)
    else:
        gen_params, latent_vector = mlp.sample_parameters(return_latent=True)
        np.save(f"{output_path}{output_name}_latent.npy", latent_vector)
else:
    gen_params = mlp.sample_parameters()

# Calculate Vmax and save results
store_as_hdf5(gen_params, met_model, names_km, ss_idx, output_path, output_name)
np.save(f"{output_path}{output_name}.npy", gen_params)