import os
import sys
import numpy as np
import pandas as pd
from configparser import ConfigParser

import helper as hp
import helpers.helpers_latent_pipeline as helperLatent

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Load configs ===
cfg = ConfigParser()
cfg.read("latent_analysis/latent_config.ini")

latent_cfg = cfg["LATENT_ANALYSIS"]

# =====================================================
# Read values from config
# =====================================================

TASK = latent_cfg.getint("task")



# ---- GLOBAL experiment parameters (from DEFAULT) ----
GENERATOR_ID = cfg["DEFAULT"].getint("generator_id")
experiment_root = cfg["DEFAULT"].get("experiment_root")

LATENT_VEC_INDICES = [
    int(x.strip()) for x in latent_cfg.get("seed_latent_vectors").split(",")
]

N_SETS_SAMPLED = latent_cfg.getint("n_sets_sampled")

# =====================================================
# Resolve paths
# =====================================================

latent_vectors_path = latent_cfg["latent_vectors_path"] % {
    "generator_id": GENERATOR_ID,
    "n_sets_sampled": N_SETS_SAMPLED,
}

max_eigen_path = latent_cfg["max_eigen_path"] % {
    "generator_id": GENERATOR_ID,
    "n_sets_sampled": N_SETS_SAMPLED,
}

unscaled_batch_path = latent_cfg["unscaled_sampled_parametersets_path"] % {
    "generator_id": GENERATOR_ID,
    "n_sets_sampled": N_SETS_SAMPLED,
}

output_folder_individual = experiment_root % {
    "generator_id": GENERATOR_ID
}


output_folder_3_point = latent_cfg["output_folder_3_point_analysis"]

# =====================================================
# Constraints
# =====================================================

lnminkm = cfg.getfloat("CONSTRAINTS", "min_km")
lnmaxkm = cfg.getfloat("CONSTRAINTS", "max_km")

# =====================================================
# Parameter names
# =====================================================

names_km = hp.load_pkl(cfg["PATHS"]["param_names"])

# =====================================================
# Flags
# =====================================================

pf_flag = cfg.getint("PARAMETER_FIXING", "pf_flag")

# =====================================================
# FULL PIPELINE: 3-point analysis
# =====================================================

helperLatent.run_three_point_analysis_for_seeds(
    cfg=cfg,
    generator_id=GENERATOR_ID,
    seed_indices=LATENT_VEC_INDICES,
    names_km=names_km,
    lnminkm=lnminkm,
    lnmaxkm=lnmaxkm,
    pf_flag=pf_flag,
    output_base_folder=os.path.join(
        output_folder_individual,
        output_folder_3_point,
    ),
    prior_eigs_path=max_eigen_path,
    feature_selection_mode="span",
    perturbations=[+1.0, -1.0],
    perturb_range=np.linspace(-2, 2, num=41),
)
