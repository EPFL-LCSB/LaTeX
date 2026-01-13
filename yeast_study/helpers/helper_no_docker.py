import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    return plt


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

