
# sketch of what it should actually do:

'''
generator produced 500 samples of which I stored the max eigenvalues and the unscaled parameters as well as scaled parameters. Lets call this unscaled batch M.
From the scaled M* we calculated eigenvalues and looked at its distribution (which we call prior distribution of eigenvalues). From there we will pick 3 points to explore sitting at 10,50 and 90th percentile.
Lets look at only one point first:
- we do individual perturbation study and generate two times 99 unscaled parameter sets (because plus and minus) which we we call N.
- together with M we can do a new scaling: M+N -> M'+N' and save the batchmax and batchmin.
- we then. update the eigenvalue corresponding to the original point / seed which we will call baseline lambda from posterior distribution of eigenvalues on M'. It suffices to just select that updated parameter set and calculatinbg eigenalue)
- we compute the percentage changes of eigenvalues corresponding to each feature.
- we can select two important features either based on absolute span (so most negative+positive) or just the highest abs(change when + perturbed)
- we generate a joint perturbation study on these two features and save the indices of the features along with the unscaled parameter that each of the 41*41 perturbations produced. (we call this unscaled batch J)
- we then scale J based on the batchmin and batchmax we got from M+N.
- then we save literally everything we computed in a dictionary. A point gets: its own index, the original prior baseline lambda, the updated scaled posterior baseline lambda, the unscaled parameter sets N (from plus and minus), the scaled version of them N', the batchmin batchmax, tthe percentage changes for positive perturbation and the percentage changs of negative (compared to posterior baeline), the unscaled parameter sets J from the joint perturbtion study, the scaled parameter sets J' based on batchmin batchmax of posterior, the 2D array of lambdas and the 2D array of taus in hours.
'''

# problem: it does something else: 



import sys, os
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import sys
from configparser import ConfigParser
import numpy as np
import pandas as pd
import keras as ks

# renaissance scripts
import helper as hp
from evostrat.init_mlp import MLP
import multiprocessing as mp

# custom scripts for yeast
from docker.work.latent_analysis.helpers.jacobian_solver_yeast import check_jacobian
from docker.work.latent_analysis.helpers.npy_to_hdf5_yeast import store_as_hdf5



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
# Sampling / Scaling Utilities
# ===============================================================

def sample_parameters_once(mlp, noise, batchscales=None, do_Scaling=True):
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
    # because I always forget the name of this function
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
# Eigenvalue computation
# ===============================================================

def calc_eig(gen_param, chk_jcbn, names_km):
    chk_jcbn._prepare_parameters(gen_param, names_km)

    return chk_jcbn.calc_eigenvalues_recal_vmax()

def evaluate_eigenvalues_for_scaled_sets(
    scaled_samples,
    chk_jcbn,
    names_km
):
    eigenvalues = []

    for row in scaled_samples:
        val = calc_eig(row.reshape(1, -1), chk_jcbn, names_km)
        eigenvalues.append(val[0])

    return np.array(eigenvalues)


# ===============================================================
# Individual Feature Perturbation
# ===============================================================

def individual_feature_perturbance(mlp, chk_jcbn,
                                   initial_latent_vector,
                                   names_km,
                                   perturbance=0.5):
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


def all_features_perturbed_individually(
    mlp,
    chk_jcbn,
    perturbations,
    sample_index,
    output_name,
    latent_vectors_path,
    unscaled_batch_path,
    lnminkm,
    lnmaxkm,
    names_km,
    compute_eigenvalues_and_scale=True
):
    results = {}

    # Load once
    original_batch = np.load(unscaled_batch_path)
    initial_latent_vector = np.load(latent_vectors_path)[sample_index]

    # --------------------------------------------------------
    # 1. COLLECT all perturbations BEFORE scaling
    # --------------------------------------------------------
    all_perturb_unscaled = []
    pert_meta = {}   # store index slices for each perturbation

    for pert in perturbations:

        pert_noise, pert_params_unscaled = individual_feature_perturbance(
            mlp, chk_jcbn, initial_latent_vector, names_km,
            perturbance=pert
        )

        start_idx = len(all_perturb_unscaled)
        end_idx = start_idx + len(pert_params_unscaled)

        pert_meta[pert] = (start_idx, end_idx)

        all_perturb_unscaled.extend(pert_params_unscaled)

    all_perturb_unscaled = np.array(all_perturb_unscaled)

    if compute_eigenvalues_and_scale:
        # --------------------------------------------------------
        # 2. SCALE EVERYTHING TOGETHER (shared baseline)
        # --------------------------------------------------------

        scaled_all, baseline_params, scaled_M_only, batch_min, batch_max = scaling_with_original_batch(
            original_batch,
            all_perturb_unscaled,
            lnminkm, lnmaxkm,
            sample_index
        )

        # --------------------------------------------------------
        # 3. COMPUTE eigenvalues slice by slice
        # --------------------------------------------------------
        eigenvalues_all = []
        for gen_param in scaled_all:
            eigenvalues_all.append(calc_eig(gen_param.reshape(1, -1), chk_jcbn, names_km))

        baseline_eig = calc_eig(baseline_params.reshape(1, -1), chk_jcbn, names_km)

    # --------------------------------------------------------
    # 4. Re-assemble results dictionary
    # --------------------------------------------------------

        for pert in perturbations:
            s, e = pert_meta[pert]
            results[f"pert_{pert:+.2f}"] = {
                "perturbance": pert,
                "unscaled_samples": all_perturb_unscaled[s:e],
                "scaled_samples": scaled_all[s:e],
                "eigenvalues": eigenvalues_all[s:e],
                "baseline_lambda": baseline_eig,
                "baseline_params": baseline_params,
            }
    else:
        for pert in perturbations:
            s, e = pert_meta[pert]
            results[f"pert_{pert:+.2f}"] = {
                "perturbance": pert,
                "unscaled_samples": all_perturb_unscaled,
            }

    # Save
    np.save(output_name, results)
    print(f"Saved results to: {output_name}")
    return results


# ===============================================================
# Joint 2D Feature Perturbation
# ===============================================================

def joint_perturbation_2_most_important_features(
    mlp,
    chk_jcbn,
    sample_index,
    feature_1_idx,
    feature_2_idx,
    save_path_folder,
    filename,

    # new explicit args
    latent_vectors_path,
    unscaled_batch_path,
    lnminkm,
    lnmaxkm,
    names_km,

    perturb_range,
    compute_eigenvalues_and_scale=True
):
    initial_latent_vector = np.load(latent_vectors_path)[sample_index]
    original_batch = np.load(unscaled_batch_path)

    n = len(perturb_range)
    perturbed_eigs = np.full((n, n), np.nan)

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

    if compute_eigenvalues_and_scale:
    
        scaled_samples, baseline_params, _,_,_ = scaling_with_original_batch(
            original_batch,
            np.array(param_sets_unscaled),
            lnminkm, lnmaxkm,
            sample_index
        )

        evaluate_and_save_tau_matrix(
            scaled_samples,
            mlp,
            chk_jcbn,
            sample_index,
            feature_1_idx,
            feature_2_idx,
            save_path_folder,
            filename,
            names_km,
            perturb_range
        )


        
    np.save(os.path.join(save_path_folder, f"{filename}_params_unscaled.npy"),
                np.array(param_sets_unscaled))

    return np.array(param_sets_unscaled)


def evaluate_and_save_tau_matrix(scaled_samples,
                                 mlp,
                                 chk_jcbn,
                                 sample_index,
                                 feature_1_idx,
                                 feature_2_idx,
                                 save_path_folder,
                                 filename,
                                 names_km,

                                perturb_range=np.linspace(-2, 2, num=41)):
    n = len(perturb_range)
    perturbed_eigs = np.full((n, n), np.nan)

    for idx, gen_param in enumerate(scaled_samples):
            row, col = divmod(idx, n)
            perturbed_eigs[row, col] = calc_eig(
                gen_param.reshape(1, -1),
                chk_jcbn,
                names_km
            )[0]
    
    features = [feature_1_idx, feature_2_idx]
    tau_matrix = -60.0 / perturbed_eigs
    tau_matrix[tau_matrix < 0] = np.nan

    os.makedirs(save_path_folder, exist_ok=True)
    np.save(os.path.join(save_path_folder, f"{filename}_lambda.npy"),
                perturbed_eigs)
    np.save(os.path.join(save_path_folder, f"{filename}_tau.npy"),
                tau_matrix)
    
    np.save(os.path.join(save_path_folder, f"{filename}_features.npy"),
                features)

    print(f"Saved λmax and τmax heatmaps to {save_path_folder}")
    
    return perturbed_eigs
# ===============================================================
# FULL PIPELINE
# ===============================================================
from keras import backend as K
def run_three_point_analysis(
    cfg,
    generator_ids,
    seed_latent_vectors,
    perturbations_list,
    names_km,
    lnminkm,
    lnmaxkm,
    pf_flag,
    output_base_folder
):
    """
    Perform the full three-point analysis (TASK 2):
    - Loop over multiple generator IDs
    - For each ID: load generator + model
    - For each provided seed latent vector: run individual perturbation
    - Select top 2 most important features (+1 vs -1)
    - Run joint 2D perturbation heatmap for these indices

    Arguments:
        cfg: full ConfigParser object
        generator_ids: list of generator IDs to evaluate
        seed_latent_vectors: list of lists of seed latent IDs for each generator
        perturbations_list: list of perturbations to run (e.g. [+1, -1])
        names_km: parameter name list from main script
        lnminkm, lnmaxkm: scaling bounds
        pf_flag: parameter fixing flag
        output_base_folder: folder where "3_point_run/" outputs go
    """

    # Create main out folder
    os.makedirs(output_base_folder, exist_ok=True)

    n_sets_sampled = cfg["LATENT_ANALYSIS"].getint("n_sets_sampled")
    
    for i, gen_id in enumerate(generator_ids):
        print(f"\n===== WORKING ON GENERATOR {gen_id} =====")
        
        base_path = cfg["LATENT_ANALYSIS"]["generator_weights_path"]
        generator_weights_path = os.path.join(
            os.path.dirname(base_path),
            f"weights_{gen_id}.pkl"
        )
        print()
        print('getting weights from: ', generator_weights_path)
         



        # UPDATE ALL PATHS FOR NEW GENERATOR --------------------         
        latent_template = cfg.get("LATENT_ANALYSIS", "latent_vectors_path", raw=True)
        unscaled_template = cfg.get("LATENT_ANALYSIS", "unscaled_sampled_parametersets_path", raw=True)

        latent_template = latent_template.replace("%(generator_id)s", "{generator_id}") \
                                         .replace("%(n_sets_sampled)s", "{n_sets_sampled}")

        unscaled_template = unscaled_template.replace("%(generator_id)s", "{generator_id}") \
                                             .replace("%(n_sets_sampled)s", "{n_sets_sampled}")

        latent_path_gen = latent_template.format(
            generator_id=gen_id,
            n_sets_sampled=n_sets_sampled,
        )

        unscaled_path_gen = unscaled_template.format(
            generator_id=gen_id,
            n_sets_sampled=n_sets_sampled,
        )
        print("Latent vectors path: ", latent_path_gen)
        print("Unscaled params path:", unscaled_path_gen)

        # LOADING MODEL FOR THIS GENERATOR ------------------------------------------------

        K.clear_session() # clear old model

        mlp, chk_jcbn = load_kinetic_model(
            generator_weights_path = generator_weights_path,
            thermo_file = cfg["PATHS"]["thermo_file"],
            kinetics_file = cfg["PATHS"]["kinetics_file"],
            ss_file = cfg["PATHS"]["ss_file"],
            ss_idx = int(cfg["EVOSTRAT"]["ss_idx"]),
            names_km = names_km,
            lnminkm = lnminkm,
            lnmaxkm = lnmaxkm,
            n_sets = 1,
            pf_flag = pf_flag
        )
        print('Loaded')

        

        # make output folder
        base_out_gen = os.path.join(output_base_folder, f"gen{gen_id}")
        os.makedirs(base_out_gen, exist_ok=True)

        # Loop over all user-provided seed latent vector IDs    
        for seed_latent_id in seed_latent_vectors[i]:

            print(f"\n--- RUNNING seed {seed_latent_id} on generator {gen_id} ---")

            out_np = os.path.join(
                base_out_gen,
                f"gen{gen_id}_seed{seed_latent_id}.npy"
            )

            # Individual feature perturbation -------------------------------
            results = all_features_perturbed_individually(
                mlp = mlp,
                chk_jcbn = chk_jcbn,
                perturbations = perturbations_list,
                sample_index = seed_latent_id,
                output_name = out_np,

                latent_vectors_path = latent_path_gen,
                unscaled_batch_path = unscaled_path_gen,
                lnminkm = lnminkm,
                lnmaxkm = lnmaxkm,
                names_km = names_km,
                compute_eigenvalues_and_scale=False

            )

            # ---------------- run joint perturbation ----------------------

            joint_out = os.path.join(base_out_gen, "heatmaps")
            os.makedirs(joint_out, exist_ok=True)

            out_heatmap = f"joint_gen{gen_id}_seed{seed_latent_id}"

            unscaled_params_joint = joint_perturbation_2_most_important_features(
                mlp = mlp,
                chk_jcbn = chk_jcbn,
                sample_index = seed_latent_id,
                feature_1_idx = top2[0],
                feature_2_idx = top2[1],
                save_path_folder = joint_out,
                filename = out_heatmap,

                latent_vectors_path = latent_path_gen,
                unscaled_batch_path = unscaled_path_gen,
                lnminkm = lnminkm,
                lnmaxkm = lnmaxkm,
                names_km = names_km,

                perturb_range = np.linspace(-2, 2, num=41),
                compute_eigenvalues_and_scale=False
            )

            ## SCALING
            # we concat unscaled_params_joint and results[''pert_+1.00']['unscaledscaled_samples]


            original_batch = np.load(unscaled_path_gen)

            perturb_range = np.linspace(-2, 2, num=41)
            n_joint_sets = len(perturb_range)**2
       

            key_plus  = [k for k in results if "+1.00" in k][0]
            key_minus = [k for k in results if "-1.00" in k][0]

            indiv_plus  = results[key_plus]['unscaled_samples']
            indiv_minus = results[key_minus]['unscaled_samples']

            combined_unscaled_parameter_sets = np.vstack([
                unscaled_params_joint,   # shape (n_joint_sets, P)
                indiv_plus,              # shape (n_features,  P)
                indiv_minus,             # shape (n_features,  P)
            ])

            scaled_perturbations_all, scaled_baseline_sample, scaled_all_original_only = scaling_with_original_batch(
                original_batch,
                combined_unscaled_parameter_sets,
                lnminkm, lnmaxkm,
                seed_latent_id
            )

            # ---------------------------------------
            # 4. Evaluate eigenvalues for scaled sets
            # ---------------------------------------
            eig_all = evaluate_eigenvalues_for_scaled_sets(
                scaled_perturbations_all,
                chk_jcbn,
                names_km
            )

            eig_joint = eig_all[:n_joint_sets].reshape(len(perturb_range), len(perturb_range))
            eig_indiv_plus  = eig_all[n_joint_sets : n_joint_sets + indiv_plus.shape[0]]
            eig_indiv_minus = eig_all[n_joint_sets + indiv_plus.shape[0] :]

            tau_mat = -60.0 / eig_joint
            tau_mat[tau_mat < 0] = np.nan
            tau_mat[np.abs(tau_mat) > 2e4] = np.nan

            np.save(os.path.join(save_path, f"{filename}_lambda.npy"), eig_joint)
            np.save(os.path.join(save_path, f"{filename}_tau.npy"), tau_mat)
            np.save(os.path.join(save_path, f"{filename}_scaled_individual_plus.npy"),  eig_indiv_plus)
            np.save(os.path.join(save_path, f"{filename}_scaled_individual_minus.npy"), eig_indiv_minus)
            np.save(os.path.join(save_path, f"{filename}_baseline.npy"), baseline_params)



            # ---------------- top 2 feature detection ---------------------
            key_plus = [k for k in results if "+1.00" in k][0]
            key_minus = [k for k in results if "-1.00" in k][0]

            res_plus = results[key_plus]
            res_minus = results[key_minus]

            baseline_lambda = res_plus["baseline_lambda"][0]

            pc_plus = ((np.array(res_plus["eigenvalues"]).squeeze() - baseline_lambda)
                       / abs(baseline_lambda)) * 100
            pc_minus = ((np.array(res_minus["eigenvalues"]).squeeze() - baseline_lambda)
                        / abs(baseline_lambda)) * 100

            delta = np.abs(pc_minus - pc_plus)
            top2 = np.argsort(delta)[-2:][::-1]

            print(f"Top-2 most sensitive latent dims: {top2}")

            


### ----------------------------------------------------
#               PLOTTING AND REPORTING
###----------------------------------------------------

def parse_generators(root):
    """
    Expected structure:
    
    genXX/
        genXX_seed6.npy                <-- single
        genXX_seed108.npy
        genXX_seed309.npy
        heatmaps/                      <-- joint (tau/lambda)
            joint_genXX_seed6_tau.npy
            joint_genXX_seed6_lambda.npy
            ...
    """
    import os
    from collections import defaultdict
    
    def tree():
        return defaultdict(tree)

    data = tree()

    for gen in os.listdir(root):
        gen_path = os.path.join(root, gen)
        if not os.path.isdir(gen_path):
            continue
        
        gen_dict = data[gen]
        
        # -----------------------
        # 1. Parse SINGLE perturbation files
        # -----------------------
        for f in os.listdir(gen_path):
            if f.startswith(gen + "_seed") and f.endswith(".npy"):
                # extract seed: gen94_seed108.npy -> "108"
                seed = f.split("_seed")[1].split(".")[0]
                gen_dict["single"][seed] = os.path.join(gen_path, f)

        # -----------------------
        # 2. Parse HEATMAPS (joint tau/lambda)
        # -----------------------
        heatmap_path = os.path.join(gen_path, "heatmaps")
        if os.path.isdir(heatmap_path):
            for f in os.listdir(heatmap_path):
                if not f.endswith(".npy"):
                    continue
                if "joint" not in f:
                    continue
                
                # example: joint_gen94_seed108_tau.npy
                parts = f.split("_")
                seed_part = [p for p in parts if p.startswith("seed")]
                if not seed_part:
                    continue
                seed = seed_part[0].replace("seed", "")

                if "tau" in f:
                    gen_dict["joint"][seed]["tau"] = os.path.join(heatmap_path, f)
                elif "lambda" in f:
                    gen_dict["joint"][seed]["lambda"] = os.path.join(heatmap_path, f)

    return data



def results_loader(filepath):
    obj = np.load(filepath, allow_pickle=True).item()
    return obj

def heatmap_loader(filepath):
    return np.load(filepath)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Helper: percentage change
# --------------------------

def plot_single_bar(ax, plus, minus, baseline, y_limit=100):
    pc_plus  = (plus  - baseline) / np.abs(baseline) * 100
    pc_minus = (minus - baseline) / np.abs(baseline) * 100

    features = np.arange(len(pc_plus))
    w = 0.35

    ax.bar(features - w/2, pc_minus, width=w,
           color='royalblue', label='−100%')
    ax.bar(features + w/2, pc_plus,  width=w,
           color='indianred', label='+100%')

    ax.axhline(0, color="black", lw=1, alpha=0.6)
    ax.set_ylabel("% Δ λₘₐₓ")
    ax.set_xlabel("Feature index")
    ax.set_ylim(-y_limit, y_limit)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


# --------------------------
# Helper: tau heatmap
# --------------------------

from matplotlib.colors import TwoSlopeNorm

def plot_tau_heatmap(ax, tau_grid, perturb_range, cutoff=500, baseline_tau=None, centered_colorscale=False):
    """
    Plot τ heatmap where values > cutoff share one color,
    and the colorbar explicitly shows the top color = '≥ cutoff'.
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Clip high values so they map to the max color
    tau_clipped = np.clip(tau_grid, None, cutoff)

    # Mask invalid values
    mask = np.isnan(tau_clipped)

    # Determine color scale range 
    if centered_colorscale and baseline_tau is not None:
        vmin = 1
        vmax = cutoff
        norm = TwoSlopeNorm(vmin=vmin, vcenter=baseline_tau, vmax=vmax)
    else:
        norm = None



    # --- Create heatmap ---
    hm = sns.heatmap(
        tau_clipped,
        cmap="Spectral_r",
        norm=norm,      
        mask=mask,
        ax=ax,
        cbar=True,
        square=True
    )


    # --- Fix colorbar ticks ---
    cbar = hm.collections[0].colorbar


    # Custom tick list: pick a few and add last one at cutoff
    ticks = np.linspace(np.nanmin(tau_clipped), cutoff, 5)
    ticklabels = [f"{t:.0f}" for t in ticks]

    # Replace last label with ≥ cutoff
    ticklabels[-1] = f"≥ {cutoff}"

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)

    cbar.set_label(r"$\tau_{\max}$ (h)")

   
    # --- Axes setup ---
    ax.set_xticks(np.arange(0, len(perturb_range), 5))
    ax.set_xticklabels((perturb_range * 100).astype(int)[::5], rotation=45)

    ax.set_yticks(np.arange(0, len(perturb_range), 5))
    ax.set_yticklabels((perturb_range * 100).astype(int)[::5], rotation=45)

    ax.set_xlabel("% change feature 1")
    ax.set_ylabel("% change feature 2")

    ax.set_title(f"τ heatmap (top color = ≥ {cutoff} h)")


'''
def plot_tau_heatmap(ax, matrix, perturb_range):
    sns.heatmap(
        matrix,
        cmap="Spectral_r",
        mask=np.isnan(matrix),
        cbar_kws={"label": r"$\tau_{max}$"},
        square=True,
        ax=ax
    )

    ax.set_xticks(np.arange(0, len(perturb_range), 5))
    ax.set_xticklabels(np.linspace(-200, 200, len(perturb_range))[::5], rotation=45)

    ax.set_yticks(np.arange(0, len(perturb_range), 5))
    ax.set_yticklabels(np.linspace(-200, 200, len(perturb_range))[::5], rotation=45)

    ax.set_xlabel("% change feature 1")
    ax.set_ylabel("% change feature 2")
'''

# --------------------------
# Main: generator dashboard
# --------------------------
def plot_generator_single_pert_and_tau(
    generator_name,
    data_entry,
    results_loader,
    heatmap_loader,
    perturb_range,
    y_limit=100,
    seed_order=None,      # list of 3 seeds in desired order (AND latent idxs)
):

    percentiles = [10, 50, 90]   # fixed order mapping

    available = set(data_entry["single"].keys()) & set(data_entry["joint"].keys())
    if seed_order is not None:
        seeds = [str(s) for s in seed_order if str(s) in available]
    else:
        seeds = sorted(available, key=lambda s: int(s))

    fig, axes = plt.subplots(len(seeds), 2, figsize=(14, 4 * len(seeds)))
    if len(seeds) == 1:
        axes = np.array([axes])

    fig.suptitle(
        f"Generator {generator_name}: Single Perturbation Effects & τₘₐₓ Heatmaps",
        fontsize=18,
        y=0.95
    )

    for i, seed in enumerate(seeds):
        ax_bar = axes[i, 0]
        ax_tau = axes[i, 1]

        percentile  = percentiles[i]
        latent_idx  = seed        # <–– seed == latent_vector_index

        # -------- SINGLE FEATURE PERTURBATION --------
        path_single = data_entry["single"][seed]
        res = results_loader(path_single)

        plus     = np.array(res['pert_+1.00']['eigenvalues']).squeeze()
        minus    = np.array(res['pert_-1.00']['eigenvalues']).squeeze()
        baseline = res['pert_+1.00']['baseline_lambda']

        plot_single_bar(ax_bar, plus, minus, baseline, y_limit)

        ax_bar.set_title(
            f"Seed {seed} | pct={percentile}| "
            f"Δλₘₐₓ under ±100% perturbation | λ_baseline={np.round(baseline, 3)}"
        )

        # -------- TAU HEATMAP --------
        path_tau = data_entry["joint"][seed]["tau"]
        tau_mat  = heatmap_loader(path_tau)

        mask = np.abs(tau_mat) > 2e3
        tau_mat[mask] = np.nan

        plot_tau_heatmap(ax_tau, tau_mat, perturb_range)

        ax_tau.set_title(
            f"τₘₐₓ heatmap | Seed {seed} | pct={percentile} | latent_idx={latent_idx}"
        )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
    return fig

## ------------------------------------------------
#  POINT CHOSING IN SPACE 
## ------------------------------------------------

import pandas as pd

def farthest_three(points):
    """
    points: numpy array of shape (n, k)
    returns indices of 3 farthest-apart points
    """
    n = points.shape[0]

    # Step 1: pick any point (index 0)
    p = 0

    # Step 2: find point farthest from p
    dists = np.linalg.norm(points - points[p], axis=1)
    q = np.argmax(dists)

    # Step 3: find point farthest from both p and q
    d_p = np.linalg.norm(points - points[p], axis=1)
    d_q = np.linalg.norm(points - points[q], axis=1)

    # maximize the *minimum* distance to p and q
    min_d = np.minimum(d_p, d_q)
    r = np.argmax(min_d)

    return p, q, r


def get_3_most_distant_latent_vectors(latent_vectors_path, max_eigenvalues_path, threshold = -0.4):
    latent_vectors = np.load(latent_vectors_path)

    df = pd.read_csv(
        max_eigenvalues_path
    )

    max_eig = df.iloc[:, 1].values   # shape (500,)
    filtered_indices = np.where(max_eig < threshold)[0]
    filtered_latents = latent_vectors[filtered_indices]

    # get indices relative to filtered_latents
    i, j, k = farthest_three(filtered_latents)

    # map to original latent_vectors row indices
    orig_i = filtered_indices[i]
    orig_j = filtered_indices[j]
    orig_k = filtered_indices[k]

    # get the actual vectors
    return latent_vectors[[orig_i, orig_j, orig_k]] # (3, 99)


def get_3_most_distant_latent_vectors_indices(latent_vectors_path, max_eigenvalues_path, threshold = -0.4):
    latent_vectors = np.load(latent_vectors_path)

    df = pd.read_csv(
        max_eigenvalues_path
    )

    max_eig = df.iloc[:, 1].values   # shape (500,)
    filtered_indices = np.where(max_eig < threshold)[0]
    filtered_latents = latent_vectors[filtered_indices]

    # get indices relative to filtered_latents
    i, j, k = farthest_three(filtered_latents)

    # map to original latent_vectors row indices
    orig_i = filtered_indices[i]
    orig_j = filtered_indices[j]
    orig_k = filtered_indices[k]

    # get the actual vectors
    return [orig_i, orig_j, orig_k]# (3, 99)

def get_3_percentile_latent_vectors_indices(latent_vectors_path, max_eigenvalues_path, threshold=-0.4, csv_column=0):
    latent_vectors = np.load(latent_vectors_path)

    df = pd.read_csv(max_eigenvalues_path)
    max_eig = df.iloc[:, csv_column].values  # shape (N,)
    # Filter to points where eigenvalue < threshold
    filtered_indices = np.where(max_eig < threshold)[0]
    filtered_eigs = max_eig[filtered_indices]

    # Percentile targets
    percentiles = [10, 50, 90]
    targets = np.percentile(filtered_eigs, percentiles)

    # For each target percentile, find nearest eigenvalue
    selected_orig_indices = []
    for t in targets:
        idx_local = np.argmin(np.abs(filtered_eigs - t))
        selected_orig_indices.append(filtered_indices[idx_local])

    return selected_orig_indices  # returns list of 3 ints

import numpy as np
import pandas as pd

def get_3_percentile_latent_vectors_info(
    max_eigenvalues_path,
    csv_has_header=True,
    threshold=0
):

    # Load CSV with or without header
    df = pd.read_csv(max_eigenvalues_path, header=0 if csv_has_header else None)

    # Select eigenvalues column
    try:
        max_eig = df.iloc[:, 1].values  # second column
    except:
        max_eig = df.iloc[:, 0].values  # first column fallback

    # Filter eigenvalues < threshold
    filtered_indices = np.where(max_eig < threshold)[0]
    filtered_eigs = max_eig[filtered_indices]

    # Percentile targets
    percentiles = [10, 50, 90]
    targets = np.percentile(filtered_eigs, percentiles)

    # Build results DataFrame
    rows = []
    for p, t in zip(percentiles, targets):
        idx_local = np.argmin(np.abs(filtered_eigs - t))
        orig_idx = filtered_indices[idx_local]
        eig_val = max_eig[orig_idx]

        rows.append({
            "percentile": p,
            "target_value": float(t),
            "selected_index": int(orig_idx),
            "selected_eigenvalue": float(eig_val)
        })

    # Convert to DataFrame
    df_results = pd.DataFrame(rows)

    return df_results



import matplotlib.pyplot as plt
import numpy as np

    

    
def sanity_check_individual_vs_joint(path_dict, gen_id, point_id):

    gen = str(gen_id)
    point = str(point_id)


    gen_dict = path_dict["gen"+gen]
    path = gen_dict['single'][point+'_three_point_analysis']

    res = np.load(path, allow_pickle=True).item()

    # === Extract values ===
    pc_plus  = res["pc_plus"]
    pc_minus = res["pc_minus"]
    baseline = res["posterior_baseline_lambda"]

    lambda_grid = res["lambda_grid"]
    f1, f2 = res["top2_features"]

    perturb_range = np.linspace(-2, 2, 41)
    i0     = np.argmin(np.abs(perturb_range - 0.0))
    i_plus = np.argmin(np.abs(perturb_range - 1.0))
    i_minus= np.argmin(np.abs(perturb_range + 1.0))

    joint_baseline = lambda_grid[i0, i0]


    # === helper ===
    def pct_change(lam, lam0):
        return (lam - lam0) / abs(lam0) * 100


    print("====== SANITY CHECK: Joint vs Individual ======")
    print(f"Generator {gen} with seed point {point}")
    print(f"Top2 features: f1={f1}, f2={f2}")
    print(f"Baseline λ (grid center)of joint: {joint_baseline:.6f}")
    print(f"Baseline λ (for individual pert): {baseline:.6f}")
    print()


    # === 1. f1 +100% ===
    joint_f1_plus = pct_change(lambda_grid[i_plus, i0], joint_baseline)
    indiv_f1_plus = pc_plus[f1]

    print('lambda_grid[i_plus, i0]: ', lambda_grid[i_plus, i0])
    print('pc_plus[f1]: ', pc_plus[f1])
    print("f1 +100%:")
    print("  Joint:      %7.3f %%" % joint_f1_plus)
    print("  Individual: %7.3f %%" % indiv_f1_plus)
    print()


    # === 2. f1 -100% ===
    joint_f1_minus = pct_change(lambda_grid[i_minus, i0], joint_baseline)
    indiv_f1_minus = pc_minus[f1]

    print("f1 -100%:")
    print("  Joint:      %7.3f %%" % joint_f1_minus)
    print("  Individual: %7.3f %%" % indiv_f1_minus)
    print()


    # === 3. f2 +100% ===
    joint_f2_plus = pct_change(lambda_grid[i0, i_plus], joint_baseline)
    indiv_f2_plus = pc_plus[f2]

    print("f2 +100%:")
    print("  Joint:      %7.3f %%" % joint_f2_plus)
    print("  Individual: %7.3f %%" % indiv_f2_plus)
    print()


    # === 4. f2 -100% ===
    joint_f2_minus = pct_change(lambda_grid[i0, i_minus], joint_baseline)
    indiv_f2_minus = pc_minus[f2]

    print("f2 -100%:")
    print("  Joint:      %7.3f %%" % joint_f2_minus)
    print("  Individual: %7.3f %%" % indiv_f2_minus)
    print("================================================")


import numpy as np
import matplotlib.pyplot as plt


def plot_one_point(path_dict, gen_id, point_id, percentile, tau_cutoff = 24, centered_colorscale=False):

    gen_id = str(gen_id)
    point_id = str(point_id)
    gen_dict = path_dict["gen" + gen_id]
    path = gen_dict["single"][point_id + "_three_point_analysis"]
    res = np.load(path, allow_pickle=True).item()

    # Extract values
    seed_id     = res["seed_index"]
    plus     = res["pc_plus"]
    minus    = res["pc_minus"]
    baseline = res["posterior_baseline_lambda"]
    baseline_tau = -1.0 / baseline
    tau_grid = res["tau_grid"]/60
    f1, f2   = res["top2_features"]        # <-- IMPORTANT

    perturb_range = np.linspace(-2, 2, 41)

    # --- PLOT ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 12),gridspec_kw={"height_ratios": [1, 3]})

    # -------------------------
    #  BAR PLOT
    # -------------------------
    ax = axes[0]
    features = np.arange(len(plus))
    w = 0.35

    ax.bar(features - w/2, minus, width=w, color='royalblue',  label='-100%')
    ax.bar(features + w/2, plus,  width=w, color='indianred',  label='+100%')

    ax.axhline(0, color="black", lw=1, alpha=0.6)
    ax.set_ylabel("% Δ λₘₐₓ")
    ax.set_xlabel("Feature index")
    ax.set_title(f"Seed {seed_id} - Single-feature perturbations\nλ_baseline={baseline:.3f} - {percentile}th percentile ")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # -------------------------
    #  TAU HEATMAP
    # -------------------------
    ax = axes[1]
    plot_tau_heatmap(ax, tau_grid, perturb_range, cutoff = tau_cutoff, baseline_tau=baseline_tau, centered_colorscale=centered_colorscale)

    # Name the latent features on the τ map
    ax.set_title(f"τₘₐₓ heatmap — latent features Z1={f1}, Z2={f2}")

    plt.tight_layout()
    plt.show()
    fig.tight_layout()
    return fig


def plot_one_point_heatmap(
    path_dict,
    gen_id,
    point_id,
    percentile,
    tau_cutoff=24,
    centered_colorscale=False
):
    gen_id = str(gen_id)
    point_id = str(point_id)

    gen_dict = path_dict["gen" + gen_id]
    path = gen_dict["single"][point_id + "_three_point_analysis"]
    res = np.load(path, allow_pickle=True).item()

    seed_id = res["seed_index"]
    baseline = res["posterior_baseline_lambda"]
    baseline_tau = -1.0 / baseline
    tau_grid = res["tau_grid"] / 60
    f1, f2 = res["top2_features"]

    perturb_range = np.linspace(-2, 2, 41)

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_tau_heatmap(
        ax,
        tau_grid,
        perturb_range,
        cutoff=tau_cutoff,
        baseline_tau=baseline_tau,
        centered_colorscale=centered_colorscale,
    )

    ax.set_title(
        f"τₘₐₓ heatmap — Seed {seed_id}, Z1={f1}, Z2={f2}\n"
        f"{percentile}th percentile, λ_baseline={baseline:.3f}"
    )

    fig.tight_layout()
    return fig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_negative_eigenvalue_distribution(
    max_eigenvalues_path,
    bins=40,
    figsize=(8, 5),
    colors=None,
    show_percentiles=(10, 50, 90),
    offset = 0
):
    """
    Plot histogram of negative eigenvalues with percentile markers.

    Parameters
    ----------
    max_eigenvalues_path : str
        Path to CSV with eigenvalues
    bins : int
        Number of histogram bins
    figsize : tuple
        Figure size
    colors : dict or None
        Color mapping for percentiles, e.g. {10: '#FF8285', 50: '#F6A500', 90: '#00B4D2'}
    show_percentiles : tuple
        Percentiles to mark

    Returns
    -------
    fig : matplotlib.figure.Figure
    percentiles : dict
        Mapping {percentile: value}
    """

    if colors is None:
        colors = {
            10: '#FF8285',
            50: '#F6A500',
            90: '#00B4D2',
            "hist": '#484D5E',
        }
        

    # --- Load data ---
    df = pd.read_csv(max_eigenvalues_path, header=None)

    try:
        eig = df.iloc[:, 1].values
    except IndexError:
        eig = df.iloc[:, 0].values

    eig_neg = eig[eig < 0]

    # --- Compute percentiles ---
    perc_values = {
        p: np.percentile(eig_neg, p) for p in show_percentiles
    }

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        eig_neg,
        bins=bins,
        color=colors["hist"],
        edgecolor="black",
        alpha=0.75,
        label="λ < 0",
    )

    ypos = None

    for p, val in perc_values.items():
        ax.axvline(val, color=colors[p], linestyle="--", linewidth=2)
        if ypos is None:
            ypos = ax.get_ylim()[1] * 0.85
        ax.text(
            val-offset,
            ypos,
            f"{p}th",
            color=colors[p],
            fontsize=10,
            ha="right",
            va="center",
        )

    ax.set_xlabel("Eigenvalue λ (only λ < 0)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Negative Eigenvalues with 10/50/90 Percentiles")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig, perc_values



def get_incidence_at_training(df, threshold = -0.4, rows_per_iter=21):

    n_iters = len(df) // rows_per_iter
    incidence = [] # incidence = % of models with max_eigenvalue < threshold

    for i in range(n_iters):
        final_row = df.iloc[i * rows_per_iter + (rows_per_iter - 1)].values
        below = np.sum(final_row < threshold)
        percent = below / len(final_row) * 100
        incidence.append(percent)

    return incidence


def plot_training_incidence(incidence_list, points_to_highlight, threshold=-0.4):

    darkblue = '#007191'
    lightblue = '#62C9D4'
    orange = '#F37A00'
    red = '#D42011'

    n_iters = len(incidence_list)

    plt.figure(figsize=(10, 3))
    plt.plot(range(1, n_iters + 1), incidence_list, marker='o', linestyle='-', color = lightblue)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
    plt.title(f"Fraction of Eigenvalues < {threshold} per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Incidence %")
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # iterations to highlight 

    highlight_iters = [i + 1 for i in points_to_highlight]
    highlight_x = highlight_iters
    highlight_y = [incidence_list[i - 1] for i in highlight_iters]

    # highlight points
    plt.scatter(
        highlight_x,
        highlight_y,

        color=red,
        s=80,
        zorder=5,
        label="Highlighted iterations"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    # report values
    for point in points_to_highlight:
        print('Incidence Generator ', point, ': ', incidence_list[point])


def plot_training_incidence(incidence_list, points_to_highlight, threshold=-0.4):

    darkblue = '#007191'
    lightblue = '#62C9D4'
    orange = '#F37A00'
    red = '#D42011'

    n_iters = len(incidence_list)

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(
        range(1, n_iters + 1),
        incidence_list,
        marker='o',
        linestyle='-',
        color=lightblue
    )

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
    ax.set_title(f"Fraction of Eigenvalues < {threshold} per Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Incidence %")
    ax.grid(True, linestyle='--', alpha=0.4)

    # iterations to highlight (points_to_highlight assumed 0-based)
    highlight_iters = [i + 1 for i in points_to_highlight]
    highlight_y = [incidence_list[i] for i in points_to_highlight]

    ax.scatter(
        highlight_iters,
        highlight_y,
        color=red,
        s=80,
        zorder=5,
        label="Highlighted iterations"
    )

    ax.legend()
    fig.tight_layout()

    # report values
    for point in points_to_highlight:
        print(f"Incidence Generator {point}: {incidence_list[point]:.2f}%")

    return fig