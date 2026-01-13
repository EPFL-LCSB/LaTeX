import os
import numpy as np
from configparser import ConfigParser
import helper as hp 
import sys
import pandas as pd

import helpers.helpers_latent as helperLatent

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Load configs ===

# latent specific configs
cfg= ConfigParser()
cfg.read("latent_analysis/latent_config.ini")
latent_cfg = cfg['LATENT_ANALYSIS']

# ================================================ Read values from latent_cfgig =================================================================  

TASK = latent_cfg.getint("task")
perturbations_list = [float(x) for x in latent_cfg.get("perturbations_list").split(",")]

GENERATOR_ID = latent_cfg.getint("generator_id")
SAMPLE_INDEX = latent_cfg.getint("sample_index")
INDEX_FEATURE_1 = latent_cfg.getint("indices_feature_1")
INDEX_FEATURE_2 = latent_cfg.getint("indices_feature_2")

# Paths now computed here, NOT inside helpers
latent_vectors_path = latent_cfg["latent_vectors_path"] % {
    "generator_id": GENERATOR_ID,
    "n_sets_sampled": latent_cfg.getint("n_sets_sampled")
}
max_eigen_path = latent_cfg["max_eigen_path"] % {
    "generator_id": GENERATOR_ID,
    "n_sets_sampled": latent_cfg.getint("n_sets_sampled")
}
unscaled_batch_path = latent_cfg["unscaled_sampled_parametersets_path"] % {
    "generator_id": GENERATOR_ID,
    "n_sets_sampled": latent_cfg.getint("n_sets_sampled")
}
output_folder_individual = latent_cfg.get("output_folder_individual") % {"generator_id": GENERATOR_ID}
output_file_individual = latent_cfg.get("output_file_individual")

output_folder_heatmap = latent_cfg.get("output_folder_heatmap") % {"generator_id": GENERATOR_ID}
output_file_heatmap = latent_cfg.get("output_file_heatmap") % {
    "generator_id": GENERATOR_ID,
    "indices_feature_1": INDEX_FEATURE_1,
    "indices_feature_2": INDEX_FEATURE_2
}

# Constraints
lnminkm = float(cfg["CONSTRAINTS"]["min_km"])
lnmaxkm = float(cfg["CONSTRAINTS"]["max_km"])

# Parameter names
names_km = hp.load_pkl(cfg["PATHS"]["param_names"])  

# Additional necessary latent_cfgig values
pf_flag = int(cfg["PARAMETER_FIXING"]["pf_flag"])    
n_sets = latent_cfg.getint("n_sets_sampled")               


#  =================================================== INDIVIDUAL FEATURE PERTURBATION =================================================== 

if TASK == 0:

    # === Load trained model & kinetics checker ===
    mlp, chk_jcbn = helperLatent.load_kinetic_model(
        generator_weights_path = latent_cfg["generator_weights_path"] % {"generator_id": GENERATOR_ID},
        thermo_file = cfg["PATHS"]["thermo_file"],
        kinetics_file = cfg["PATHS"]["kinetics_file"],
        ss_file = cfg["PATHS"]["ss_file"],
        ss_idx = int(cfg["EVOSTRAT"]["ss_idx"]),
        names_km = names_km,
        lnminkm = lnminkm,
        lnmaxkm = lnmaxkm,
        n_sets = n_sets,         
        pf_flag = pf_flag        
    )

    print(f"Loaded model for generator {GENERATOR_ID}.\n")

    os.makedirs(output_folder_individual, exist_ok=True)

    helperLatent.all_features_perturbed_individually(

        mlp = mlp,
        chk_jcbn = chk_jcbn,
        perturbations = perturbations_list,
        sample_index = SAMPLE_INDEX,
        output_name = os.path.join(output_folder_individual, f"{output_file_individual}_sampleIdx{SAMPLE_INDEX}_velocity_2.npy"),

        latent_vectors_path = latent_vectors_path,
        unscaled_batch_path = unscaled_batch_path,
        lnminkm = lnminkm,
        lnmaxkm = lnmaxkm,
        names_km = names_km
    )

# ===================================================  JOINT 2D PERTURBATION HEATMAP =================================================== 


elif TASK == 1:

    # === Load trained model & kinetics checker ===
    mlp, chk_jcbn = helperLatent.load_kinetic_model(
        generator_weights_path = latent_cfg["generator_weights_path"] % {"generator_id": GENERATOR_ID},
        thermo_file = cfg["PATHS"]["thermo_file"],
        kinetics_file = cfg["PATHS"]["kinetics_file"],
        ss_file = cfg["PATHS"]["ss_file"],
        ss_idx = int(cfg["EVOSTRAT"]["ss_idx"]),
        names_km = names_km,
        lnminkm = lnminkm,
        lnmaxkm = lnmaxkm,
        n_sets = n_sets,         
        pf_flag = pf_flag         
    )

    print(f"Loaded model for generator {GENERATOR_ID}.\n")

    os.makedirs(output_folder_heatmap, exist_ok=True)

    helperLatent.joint_perturbation_2_most_important_features(
        mlp = mlp,
        chk_jcbn = chk_jcbn,
        sample_index = SAMPLE_INDEX,
        feature_1_idx = INDEX_FEATURE_1,
        feature_2_idx = INDEX_FEATURE_2,
        save_path_folder = output_folder_heatmap,
        filename = output_file_heatmap,

        latent_vectors_path = latent_vectors_path,
        unscaled_batch_path = unscaled_batch_path,
        lnminkm = lnminkm,
        lnmaxkm = lnmaxkm,
        names_km = names_km,

        perturb_range = np.linspace(-2, 2, num=41)
    )

# ===================================================  FULL PIPELINE FOR 3 POINTS =================================================== 

elif TASK == 2:

    generator_ids = [95, 94, 92]
   
    seed_latent_vectors = [
        [53, 296, 167],
        [341, 495, 167],
        [93, 270, 137]
    ]
    
    helperLatent.run_three_point_analysis(
        cfg = cfg,
        generator_ids = generator_ids,
        seed_latent_vectors = seed_latent_vectors,
        perturbations_list = perturbations_list,
        names_km = names_km,
        lnminkm = lnminkm,
        lnmaxkm = lnmaxkm,
        pf_flag = pf_flag,
        output_base_folder = os.path.join(output_folder_individual, "3_point_run_percentiles_1")
    )


