import os
import sys
import numpy as np
import pandas as pd
from configparser import ConfigParser
from keras import backend as K

# --- parent path setup (as in your original file) ---
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

# --- project imports ---
import helper as hp
from evostrat.init_mlp import MLP
from docker.work.latent_analysis.helpers.jacobian_solver_yeast import check_jacobian
from docker.work.latent_analysis.helpers.npy_to_hdf5_yeast import store_as_hdf5


# ===============================================================
# 1. Model loading
# ===============================================================

def load_kinetic_model(
    generator_weights_path,
    thermo_file,
    kinetics_file,
    ss_file,
    ss_idx,
    names_km,
    lnminkm,
    lnmaxkm,
    n_sets=1,
    pf_flag=0
):
    """
    Load generator + kinetics model.
    """
    print('Loading generator and weights...')
    cond_class = 1
    mlp = MLP(cond_class, lnminkm, lnmaxkm, n_sets, names_km, param_fixing=pf_flag)

    opt_weights = hp.load_pkl(generator_weights_path)
    mlp.generator.set_weights(opt_weights)

    print('Loading kinetics and thermodynamics models...')
    chk_jcbn = check_jacobian()
    chk_jcbn._load_ktmodels(thermo_file, kinetics_file)
    chk_jcbn._load_ssprofile(ss_file, ss_idx)

    return mlp, chk_jcbn


# ===============================================================
# 2. Sampling / Scaling Utilities
# ===============================================================

def sample_parameters_once(mlp, noise, batchscales=None, do_Scaling=True):
    """
    Sample kinetic parameters for a given latent vector.
    If do_Scaling=False, returns raw generator output (unscaled params).
    """
    noise = np.array(noise).reshape(1, -1)
    sampled_label = np.array([[mlp.cond_class]])

    gen_par = mlp.generator.predict([noise, sampled_label], batch_size=1)

    if not do_Scaling:
        return gen_par, noise

    if batchscales is None:
        x_new, new_min, new_max = hp.unscale_range(
            gen_par,
            np.min(gen_par),
            np.max(gen_par),
            mlp.min_x,
            mlp.max_x
        )
    else:
        scale_min, scale_max = batchscales
        x_new, new_min, new_max = hp.unscale_range(
            gen_par,
            scale_min,
            scale_max,
            mlp.min_x,
            mlp.max_x
        )

    return x_new, noise


def concatenate_parametersets(set1, set2):
    """
    Simple vertical stack of two (N, P) parameter matrices.
    """
    return np.vstack((set1, set2))


def scaling_with_original_batch(original_sample_batch_unscaled,
                                perturbed_params_unscaled,
                                lnminkm, lnmaxkm,
                                og_sample_index):
    """
    Core scaling step for posterior distribution:

    Inputs:
        original_sample_batch_unscaled: M, shape (N0, P)  (e.g. 500)
        perturbed_params_unscaled:      N, shape (N1, P)
        lnminkm, lnmaxkm: scaling bounds
        og_sample_index: index of the chosen seed in M

    Steps:
        merged = [M; N]
        batch_min, batch_max = min/max over merged
        scale merged -> (M', N') into [lnminkm, lnmaxkm]

    Returns:
        scaled_N            : N', shape (N1, P)
        baseline_sample     : scaled posterior baseline sample (row og_sample_index in M')
        scaled_M_only       : M', shape (N0, P)
        batch_min, batch_max: min/max of merged used for scaling
    """
    merged = concatenate_parametersets(
        original_sample_batch_unscaled,
        perturbed_params_unscaled
    )

    batch_min = np.min(merged)
    batch_max = np.max(merged)

    # Unscale/scale into [lnminkm, lnmaxkm] using batch_min/batch_max
    scaled_merged, new_min, new_max = hp.unscale_range(
        merged,
        batch_min,
        batch_max,
        lnminkm,
        lnmaxkm
    )

    n_pert = len(perturbed_params_unscaled)
    n_orig = len(original_sample_batch_unscaled)

    scaled_M_only       = scaled_merged[:n_orig]
    scaled_perturbations = scaled_merged[-n_pert:]
    baseline_sample      = scaled_merged[og_sample_index]

    return scaled_perturbations, baseline_sample, scaled_M_only, batch_min, batch_max


# ===============================================================
# 3. Eigenvalue computation
# ===============================================================

def calc_eig(gen_param, chk_jcbn, names_km):
    chk_jcbn._prepare_parameters(gen_param, names_km)
    return chk_jcbn.calc_eigenvalues_recal_vmax()


def evaluate_eigenvalues_for_scaled_sets(
    scaled_samples,
    chk_jcbn,
    names_km
):
    """
    Evaluate λ_max for each row in scaled_samples.
    Returns shape (N,) array of eigenvalues.
    """
    eigenvalues = []

    for row in scaled_samples:
        val = calc_eig(row.reshape(1, -1), chk_jcbn, names_km)
        eigenvalues.append(val[0])

    return np.array(eigenvalues)


# ===============================================================
# 4. Individual Feature Perturbation (N)
# ===============================================================

def individual_feature_perturbance(mlp,
                                   initial_latent_vector,
                                   names_km,
                                   perturbance=0.5):
    """
    For a given latent vector z (length D), create D perturbed latent vectors,
    each with one coordinate multiplied by (1+perturbance), and sample parameters.

    Returns:
        perturbed_noise   : (D, D_latent)
        perturbed_params  : (D, P) unscaled params
    """
    dummy = np.array(initial_latent_vector, dtype=float).copy()
    perturbed_params_unscaled, perturbed_noise = [], []

    for idx in range(len(dummy)):
        tmp = dummy.copy()
        tmp[idx] *= (1 + perturbance)

        param_set_unscaled, latent_used = sample_parameters_once(
            mlp, tmp, batchscales=None, do_Scaling=False
        )

        perturbed_params_unscaled.append(np.ravel(param_set_unscaled))
        perturbed_noise.append(np.ravel(latent_used))

    return np.array(perturbed_noise), np.array(perturbed_params_unscaled)


def build_N_for_seed(
    mlp,
    chk_jcbn,
    sample_index,
    latent_vectors_path,
    unscaled_batch_path,
    names_km,
    perturbations=[+1.0, -1.0]
):
    """
    For a given seed index, build all individual perturbations:

    Returns:
        original_batch_unscaled : M, shape (N0, P)
        initial_latent_vector   : z_seed, shape (D,)
        N_unscaled_dict         : dict with keys 'plus', 'minus'
                                  each value: (D, P) unscaled param sets
    """
    original_batch = np.load(unscaled_batch_path)          # M
    latent_vectors = np.load(latent_vectors_path)
    initial_latent_vector = latent_vectors[sample_index]

    N_unscaled_dict = {}

    for pert in perturbations:
        pert_noise, pert_params_unscaled = individual_feature_perturbance(
            mlp,
            initial_latent_vector,
            names_km,
            perturbance=pert
        )

        if pert > 0:
            N_unscaled_dict["plus"] = pert_params_unscaled
        else:
            N_unscaled_dict["minus"] = pert_params_unscaled

    return original_batch, initial_latent_vector, N_unscaled_dict


# ===============================================================
# 5. Joint 2D Feature Perturbation (J)
# ===============================================================

def build_J_unscaled_for_seed(
    mlp,
    sample_index,
    feature_1_idx,
    feature_2_idx,
    latent_vectors_path,
    perturb_range=np.linspace(-2, 2, num=41)
):
    """
    Build unscaled parameter sets for joint 2D perturbation of two features.
    Returns:
        J_unscaled : (len(perturb_range)**2, P)
    """
    latent_vectors = np.load(latent_vectors_path)
    initial_latent_vector = latent_vectors[sample_index]

    param_sets_unscaled = []
    for p1 in perturb_range:
        for p2 in perturb_range:
            vec = initial_latent_vector.copy()
            vec[feature_1_idx] *= (1 + p1)
            vec[feature_2_idx] *= (1 + p2)

            param_set_unscaled, _ = sample_parameters_once(
                mlp, vec, batchscales=None, do_Scaling=False
            )
            param_sets_unscaled.append(param_set_unscaled.squeeze())

    return np.array(param_sets_unscaled)


# ===============================================================
# 6. Core per-seed analysis
# ===============================================================

def analyze_single_seed(
    mlp,
    chk_jcbn,
    seed_index,
    latent_vectors_path,
    unscaled_batch_path,
    names_km,
    lnminkm,
    lnmaxkm,
    prior_eigs_array,
    feature_selection_mode,
    perturbations=[+1.0, -1.0],
    perturb_range=np.linspace(-2, 2, num=41)
    
):
    """
    Do the full analysis for ONE seed, exactly as in your sketch.

    Returns a dictionary with everything:
        - seed_index
        - prior_baseline_lambda
        - posterior_baseline_lambda
        - N_unscaled_plus / N_unscaled_minus
        - N_scaled_plus / N_scaled_minus
        - batch_min, batch_max
        - pc_plus, pc_minus
        - top2_features
        - J_unscaled, J_scaled
        - lambda_grid, tau_grid
        - scaled_M_posterior (optional)
    """

    # --- Step 1: build N (individual perturbations) ---
    original_batch_unscaled, initial_latent_vector, N_unscaled_dict = build_N_for_seed(
        mlp,
        chk_jcbn,
        seed_index,
        latent_vectors_path,
        unscaled_batch_path,
        names_km,
        perturbations=perturbations
    )

    N_plus_unscaled = N_unscaled_dict["plus"]   # (D, P)
    N_minus_unscaled = N_unscaled_dict["minus"] # (D, P)

    # Stack N = [N_plus; N_minus]
    N_unscaled_all = np.vstack([N_plus_unscaled, N_minus_unscaled])

    # --- Step 2: M + N scaling (posterior) ---
    scaled_N_all, baseline_posterior_params, scaled_M_posterior, batch_min, batch_max = (
        scaling_with_original_batch(
            original_batch_unscaled,
            N_unscaled_all,
            lnminkm, lnmaxkm,
            seed_index
        )
    )

    # recover slices for N'+ and N'-
    D = N_plus_unscaled.shape[0]
    scaled_N_plus  = scaled_N_all[:D]
    scaled_N_minus = scaled_N_all[D:2 * D]

    # --- Step 3: posterior baseline λ and Δλ ---
    posterior_baseline_lambda = calc_eig(
        baseline_posterior_params.reshape(1, -1),
        chk_jcbn,
        names_km
    )[0]

    eig_N_all = evaluate_eigenvalues_for_scaled_sets(
        scaled_N_all,
        chk_jcbn,
        names_km
    )

    eig_N_plus  = eig_N_all[:D]
    eig_N_minus = eig_N_all[D:2 * D]

    # prior baseline from full prior distribution:
    prior_baseline_lambda = float(prior_eigs_array[seed_index])

    # percent changes relative to POSTERIOR baseline
    pc_plus  = (eig_N_plus  - posterior_baseline_lambda) / abs(posterior_baseline_lambda) * 100.0
    pc_minus = (eig_N_minus - posterior_baseline_lambda) / abs(posterior_baseline_lambda) * 100.0

    # --- Step 4: select top-2 important latent features ---
    if feature_selection_mode == "span":
        # Option A: absolute span |pc_plus - pc_minus|
        scores = np.abs(pc_plus - pc_minus)

    elif feature_selection_mode == "positive":
        # Option B: only negative-direction sensitivity |pc_minus|
        scores = np.abs(pc_minus)

    else:
        raise ValueError("feature_selection_mode must be 'span' or 'positive'.")

    # select top-2 features based on chosen scoring
    top2_features = np.argsort(scores)[-2:][::-1]

    f1, f2 = int(top2_features[0]), int(top2_features[1])


    # --- Step 5: build J (joint perturbation) unscaled ---
    J_unscaled = build_J_unscaled_for_seed(
        mlp,
        seed_index,
        feature_1_idx=f1,
        feature_2_idx=f2,
        latent_vectors_path=latent_vectors_path,
        perturb_range=perturb_range
    )

    # --- Step 6: scale J with SAME batch_min/batch_max ---
    J_scaled, _, _ = hp.unscale_range(
        J_unscaled,
        batch_min,
        batch_max,
        lnminkm,
        lnmaxkm
    )

    # --- Step 7: eigenvalues and taus on grid ---
    eig_J = evaluate_eigenvalues_for_scaled_sets(
        J_scaled,
        chk_jcbn,
        names_km
    )

    n = len(perturb_range)
    lambda_grid = eig_J.reshape(n, n)

    tau_grid = -60.0 / lambda_grid
    tau_grid[tau_grid < 0] = np.nan
    tau_grid[np.abs(tau_grid) > 2e10] = np.nan

    # --- Step 8: package EVERYTHING into dict ---
    result = {
        "seed_index": int(seed_index),
        "prior_baseline_lambda": prior_baseline_lambda,
        "posterior_baseline_lambda": float(posterior_baseline_lambda),

        "N_unscaled_plus":  N_plus_unscaled,
        "N_unscaled_minus": N_minus_unscaled,
        "N_scaled_plus":    scaled_N_plus,
        "N_scaled_minus":   scaled_N_minus,

        "batch_min": float(batch_min),
        "batch_max": float(batch_max),

        "pc_plus":  pc_plus,
        "pc_minus": pc_minus,

        "top2_features": top2_features,
        "J_unscaled": J_unscaled,
        "J_scaled":   J_scaled,

        "lambda_grid": lambda_grid,
        "tau_grid":    tau_grid,

        "scaled_M_posterior": scaled_M_posterior,
    }

    return result


# ===============================================================
# 7. Wrapper: run for given seeds (you already know indices)
# ===============================================================

def run_three_point_analysis_for_seeds(
    cfg,
    generator_id,
    seed_indices,
    names_km,
    lnminkm,
    lnmaxkm,
    pf_flag,
    output_base_folder,
    prior_eigs_path,
    feature_selection_mode,
    perturbations=[+1.0, -1.0],
    perturb_range=np.linspace(-2, 2, num=41),
    eigen_csv_column=0
):
    """
    Main entrypoint when you ALREADY KNOW which seed indices to use.

    For each seed index:
        - run analyze_single_seed
        - save a .npy with the full result dict

    prior_eigs_path: CSV file containing prior λ_max for all 500 samples
                     assumed to have eigenvalues in 2nd column (like your code).
    """

    os.makedirs(output_base_folder, exist_ok=True)

    # load prior eigenvalues
    df = pd.read_csv(prior_eigs_path)

    # depending on your setup the csv file containing the maxeigenvalues from sampling are in a list with indices or not. 
    try:
        prior_eigs = df.iloc[:, 1].values
    except IndexError:
        prior_eigs = df.iloc[:, 0].values

    # build generator-specific paths
    n_sets_sampled = cfg["LATENT_ANALYSIS"].getint("n_sets_sampled")

    latent_template = cfg.get("LATENT_ANALYSIS", "latent_vectors_path")
    unscaled_template = cfg.get("LATENT_ANALYSIS", "unscaled_sampled_parametersets_path")


    latent_template = latent_template.replace("%(generator_id)s", "{generator_id}") \
                                     .replace("%(n_sets_sampled)s", "{n_sets_sampled}")
    unscaled_template = unscaled_template.replace("%(generator_id)s", "{generator_id}") \
                                         .replace("%(n_sets_sampled)s", "{n_sets_sampled}")

    latent_path_gen = latent_template.format(
        generator_id=generator_id,
        n_sets_sampled=n_sets_sampled,
    )

    unscaled_path_gen = unscaled_template.format(
        generator_id=generator_id,
        n_sets_sampled=n_sets_sampled,
    )

    print("Latent vectors path: ", latent_path_gen)
    print("Unscaled params path:", unscaled_path_gen)

    # load model
    base_path = cfg["LATENT_ANALYSIS"]["generator_weights_path"]
    generator_weights_path = os.path.join(
        os.path.dirname(base_path),
        f"weights_{generator_id}.pkl"
    )
    print("Loading weights from:", generator_weights_path)

    K.clear_session()
    mlp, chk_jcbn = load_kinetic_model(
        generator_weights_path=generator_weights_path,
        thermo_file=cfg["PATHS"]["thermo_file"],
        kinetics_file=cfg["PATHS"]["kinetics_file"],
        ss_file=cfg["PATHS"]["ss_file"],
        ss_idx=int(cfg["EVOSTRAT"]["ss_idx"]),
        names_km=names_km,
        lnminkm=lnminkm,
        lnmaxkm=lnmaxkm,
        n_sets=1,
        pf_flag=pf_flag
    )

    base_out_gen = os.path.join(output_base_folder, f"gen{generator_id}")
    os.makedirs(base_out_gen, exist_ok=True)

    for seed in seed_indices:
        print(f"\n=== Running three-point analysis for generator {generator_id}, seed {seed} ===")

        result = analyze_single_seed(
            mlp=mlp,
            chk_jcbn=chk_jcbn,
            seed_index=seed,
            latent_vectors_path=latent_path_gen,
            unscaled_batch_path=unscaled_path_gen,
            names_km=names_km,
            lnminkm=lnminkm,
            lnmaxkm=lnmaxkm,
            prior_eigs_array=prior_eigs,
            feature_selection_mode = feature_selection_mode,
            perturbations=perturbations,
            perturb_range=perturb_range
        )

        out_path = os.path.join(
            base_out_gen,
            f"gen{generator_id}_seed{seed}_three_point_analysis.npy"
        )
        np.save(out_path, result)
        print("Saved analysis to:", out_path)
