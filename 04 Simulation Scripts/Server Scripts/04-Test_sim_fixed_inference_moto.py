#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline for testing inference on average ground truth RDM

@author: alex
"""
import os
import random
from time import gmtime, strftime
from copy import deepcopy
import mask_utils
import numpy as np
import pandas as pd
import rsatoolbox

# unused now to get correspondence to permutation right
order = [
    "n01443537",
    "n01943899",
    "n01976957",
    "n02071294",  # water animals
    "n01621127",
    "n01846331",
    "n01858441",
    "n01677366",
    "n02190790",
    "n02274259",  # air and land animals (non-mammals)
    "n02128385",
    "n02139199",
    "n02416519",
    "n02437136",
    "n02437971",  # land-Mammals
    "n02951358",
    "n03272010",
    "n03482252",
    "n03495258",  # humans in the picture
    "n04254777",
    "n03237416",
    "n03124170",
    "n03379051",
    "n04572121",  # clothing
    "n02824058",
    "n02882301",
    "n03345837",
    "n04387400",
    "n03716966",
    "n03584254",
    "n04533802",
    "n03626115",
    "n03941684",
    "n03954393",
    "n04507155",  # small, handy objects
    "n02797295",
    "n02690373",
    "n02916179",
    "n02950256",
    "n03122295",
    "n04252077",  # machines
    "n03064758",
    "n04210120",
    "n04554684",
    "n03452741",
    "n03761084",  # large indoor objects
    "n03710193",
    "n03455488",
    "n03767745",
    "n04297750",
]  # landmarks


def collect_RDMs(n_subs=1, source="PyRSA", beta_type=None):
    roi_rdms = []
    for sub in range(1, n_subs + 1):
        fname = os.path.join(
            source,
            "rdms",
            "sub-" + str(sub).zfill(2),
            "RDM_" + beta_type,
        )
        rdm = rsatoolbox.rdm.rdms.load_rdm(fname, file_type="hdf5")

        # Collect single RDMs
        if isinstance(roi_rdms, list):
            roi_rdms = rdm
        else:
            roi_rdms.append(rdm)
    return roi_rdms


def collect_model_rdms(rdms_full, rois, prec_type="none", method=None):
    model_rdms = []
    for roi in rois:
        rdms = rdms_full.subset("roi", roi).subset("prec_type", prec_type)
        model_rdm = rsatoolbox.util.inference_util.pool_rdm(rdms, method=method)
        model_rdm.rdm_descriptors.update({"roi": np.array([roi])})
        if isinstance(model_rdms, list):
            model_rdms = model_rdm
        else:
            model_rdms.append(model_rdm)
    return model_rdms


def pattern_subset_rdms_sparse(model_rdms_full, data_rdms, n_stim, permutation=None):
    model_rdms_list = []
    data_rdms_list = []
    factor_list = []
    n_cond = model_rdms_full.n_cond

    if permutation is not None:
        # subsampling orders the permutation
        # to match the data rdms we reverse this
        o = np.argsort(permutation)
        o_inv = np.empty_like(o)
        o_inv[o] = np.arange(len(o_inv))
        model_rdms_full = model_rdms_full.subsample_pattern("index", permutation - 1)
        model_rdms_full.dissimilarities = np.nan_to_num(model_rdms_full.dissimilarities)
        model_rdms_full.reorder(o_inv)
        model_rdms_full.pattern_descriptors["index"] = np.arange(n_cond)
    for stim in n_stim:
        parts = partition_sets(n_cond, stim)
        model_rdms_list.append(model_rdms_full.subset_pattern("index", parts[0]))
        data_rdms_list.append(data_rdms.subset_pattern("index", parts[0]))
        factor_list.append(np.array([stim]))
    factors = np.stack(factor_list, axis=1).T
    return model_rdms_list, data_rdms_list, factors


def partition_sets(n_cond, stim, sampling="random"):
    parts = []
    lst = list(range(n_cond))
    if sampling == "random":
        random.shuffle(lst)

    for i in range(0, n_cond, stim):
        parts.append(lst[i : (i + stim)])

    return parts


def check_next_model_idx(winner_idx, model_names, gt_model, k):
    tie_winner = None
    if k == len(winner_idx):
        tie_winner = winner_idx[k]
    else:
        if model_names[winner_idx[k]][0] != gt_model:
            tie_winner = winner_idx[k]
        else:
            tie_winner = check_next_model_idx(winner_idx, model_names, gt_model, k + 1)
    return tie_winner


def results_summary(results, roi_h):
    evaluations = results.evaluations
    if len(evaluations.shape) > 2:
        evaluations = np.squeeze(np.transpose(evaluations, (2, 1, 0)), axis=None)
    noise_ceiling = results.noise_ceiling

    # Names
    model_names = [m.name for m in results.models]
    gt_model = roi_h
    gt_model_idx = np.where(np.array(model_names) == roi_h)[0][0]

    # Determine winner model
    point_estimators = np.nanmean(evaluations, axis=0)
    standard_deviations = np.nanstd(evaluations, axis=0)
    best = np.nanmax(point_estimators)
    winner_idx = np.where(point_estimators == best)[0]

    if len(winner_idx) > 1:  # handle ties
        winner_idx_tmp = check_next_model_idx(winner_idx, model_names, gt_model, 0)
        winner_idx = np.array(winner_idx_tmp)
    elif len(winner_idx) == 0:
        winner_idx = np.array([0])

    winner_model = model_names[winner_idx[0]]
    recovery = winner_model == gt_model

    # Significance testing
    p_pairwise, p_zero, p_noise = results.test_all()
    significance = fdr_control(p_pairwise, alpha=0.05)
    better = significance[gt_model_idx, :]

    # Noise ceiling tests
    if len(noise_ceiling.shape) > 1:
        noise_ceilings = np.nanmean(noise_ceiling, axis=1)
    else:
        noise_ceilings = noise_ceiling
    above_nc = best > noise_ceilings[0]
    # p = rsatoolbox.util.inference_util.t_test_nc(
    #    evaluations, variances, noise_ceilings[0], noise_ceil_var[:, 0], dof
    # )
    nc_significance = p_noise[gt_model_idx]

    # Putting everything together
    summary = {
        "GT": str(gt_model),
        "winner": str(winner_model),
        "point_est": point_estimators[winner_idx],
        "std": standard_deviations[winner_idx],
        "recovered": int(recovery),
        "n_sig_better": sum(better),
        "nc_low": noise_ceilings[0],
        "nc_high": noise_ceilings[1],
        "above_nc": int(above_nc),
        "dif_from_nc_sig": nc_significance,
    }
    # for multi in [pe, stds, pw_sigs]:
    #     summary.update(multi)
    return summary


def fdr_control(p_values, alpha=0.05):
    ps = rsatoolbox.util.rdm_utils.batch_to_vectors(np.array([p_values]))[0][0]
    ps = np.sort(ps)
    criterion = alpha * (np.arange(ps.shape[0]) + 1) / ps.shape[0]
    k_ok = ps < criterion
    if np.any(k_ok):
        k_max = np.max(np.where(ps < criterion)[0])
        crit = criterion[k_max]
    else:
        crit = 0
    significant = p_values < crit

    return significant


def main():
    ###############################################################################
    out_dir = os.path.join(os.environ["SOURCE"], "derivatives", "results")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # Set directories and specify ROIs
    ds_dir = os.environ["SOURCE"]
    n_subs = 5
    # directory in which subject-specific volumetric ROI masks are saved by FS
    freesurfer_mri = "mri_glasser"
    mask_dir = os.path.join(
        ds_dir, "derivatives", "freesurfer", "sub-" + str(1).zfill(2), freesurfer_mri
    )
    mask_dict = mask_utils.load_dict(
        os.path.join(mask_dir, "sub-" + str(1).zfill(2) + "_mask_dict_EPI_disjoint.npy")
    )
    roi_h_list = list(mask_dict.keys())
    mask_dict = None
    n_stim = np.flip([5, 10, 20, 30, 50])
    comp_methods = ["corr", "cosine_cov", "cosine"]

    # load permutation
    permutations = np.loadtxt(
        os.path.join(os.environ.get("INTERMEDIATE"), "perms.csv"),
        delimiter=",",
        dtype=int,
    )
    ###############################################################################
    results_list = []
    df = pd.DataFrame()
    df_idx = -1

    rdms = collect_RDMs(
        n_subs=n_subs,
        source=os.path.join(os.environ.get("INTERMEDIATE"), "PyRSA"),
        beta_type="data_perm_mixed",
    )
    rdms.pattern_descriptors["index"] = np.array(
        [int(i) for i in rdms.pattern_descriptors["index"]]
    )
    prec_types = np.unique(rdms.rdm_descriptors["prec_type"])
    run_subsets = np.flip(np.unique(rdms.rdm_descriptors["n_runs"]))
    snr_range = np.flip(np.unique(rdms.rdm_descriptors["snr_rel"]))
    perms_range = np.unique(rdms.rdm_descriptors["perm"])
    signal_rdms = collect_RDMs(
        n_subs=n_subs,
        source=os.path.join(os.environ.get("SOURCE"), "derivatives", "PyRSA_GT"),
        beta_type="signal",
    )
    # puts model RDMs back into SPM order (nXX)
    signal_rdms.sort_by(synset="alpha")
    rdms.sort_by(synset="alpha")
    signal_rdms.pattern_descriptors["index"] = np.arange(signal_rdms.n_cond)
    rdms.pattern_descriptors["index"] = np.arange(rdms.n_cond)
    assert signal_rdms.pattern_descriptors["stim"] == rdms.pattern_descriptors["stim"]
    # method = 'cosine'
    for method in comp_methods:
        # prec_type = 'res-total'
        for prec_type in prec_types:
            # for prec_type in ["res-total"]:
            print(f"Starting inference for {method}:{prec_type}")
            # Collect full model RDMs (pooled according to precision type and comparison method)
            model_rdms_full = collect_model_rdms(
                signal_rdms, roi_h_list, prec_type=prec_type, method=method
            )
            # roi_h = 'V1_left'
            for roi_h in roi_h_list:
                # snr = 1
                for snr in snr_range:
                    # perm = 1
                    for perm in perms_range:
                        # n_runs = 32
                        for n_runs in run_subsets:
                            summary = {}

                            # Collect data RDMs
                            data_rdms = (
                                rdms.subset("prec_type", prec_type)
                                .subset("snr_rel", snr)
                                .subset("perm", perm)
                                .subset("n_runs", n_runs)
                                .subset("roi", roi_h)
                            )

                            if permutations.ndim == 1:
                                i_perm = permutations
                            else:
                                i_perm = permutations[:, perm - 1]
                            # Pattern subset model RDMs
                            (
                                model_rdms_list,
                                data_rdms_list,
                                factors,
                            ) = pattern_subset_rdms_sparse(
                                model_rdms_full,
                                data_rdms,
                                n_stim,
                                permutation=i_perm,
                            )
                            n_subsets = len(model_rdms_list)

                            # Do fixed inference for each subset
                            # comb = 0
                            for comb in range(n_subsets):
                                fixed_models = []
                                fixed_results = []
                                model_rdms = model_rdms_list[comb]
                                data_rdms_sub = data_rdms_list[comb]

                                # Model selection
                                for i_model in roi_h_list:
                                    fixed_models.append(
                                        rsatoolbox.model.ModelFixed(
                                            i_model, model_rdms.subset("roi", i_model)
                                        )
                                    )

                                # ModelFixed class rearranges pattern descriptor indices, and
                                # we need to redo this for data_rdms_sub, otherwise
                                # eval_bootstrap_pattern will throw up an exception
                                data_rdms_sub.pattern_descriptors["index"] = np.arange(
                                    data_rdms_sub.n_cond
                                )

                                # Perform inference
                                fixed_results = (
                                    rsatoolbox.inference.eval_bootstrap_pattern(
                                        fixed_models, data_rdms_sub, method=method
                                    )
                                )

                                # rsatoolbox.vis.plot_model_comparison(fixed_results)
                                summary = results_summary(fixed_results, roi_h)
                                oe = dict(
                                    zip(
                                        [
                                            "sub-"
                                            + str(i + 1).zfill(2)
                                            + "_data_rdm_oe_rel"
                                            for i in range(data_rdms_sub.n_rdm)
                                        ],
                                        data_rdms_sub.rdm_descriptors["oe_rel"],
                                    )
                                )
                                roi_size = dict(
                                    zip(
                                        [
                                            "sub-"
                                            + str(i + 1).zfill(2)
                                            + "_GT_roi_size"
                                            for i in range(data_rdms_sub.n_rdm)
                                        ],
                                        data_rdms_sub.rdm_descriptors["roi_size"],
                                    )
                                )
                                for multi in [oe, roi_size]:
                                    summary.update(multi)
                                summary.update(
                                    {
                                        "comparison_method": method,
                                        "pattern_subset": int(factors[comb]),
                                        "prec_type": prec_type,
                                        "snr_rel": snr,
                                        "perm_num": perm,
                                        "n_runs": n_runs,
                                    }
                                )
                                df_idx += 1
                                summary_df = pd.DataFrame(summary, index=[df_idx])
                                df = pd.concat([df, summary_df])
                                results_list.append(fixed_results)

    csv_fname = os.path.join(
        os.environ["SOURCE"],
        "derivatives",
        "results",
        "results_" + strftime("%Y-%m-%d_%H-%M", gmtime()) + ".csv",
    )
    df.to_csv(csv_fname)
    npy_fname = os.path.join(
        os.environ["SOURCE"],
        "derivatives",
        "results",
        "results_" + strftime("%Y-%m-%d_%H-%M", gmtime()) + ".npy",
    )
    np.save(npy_fname, results_list)
    # df_2 = pd.read_csv(csv_fname, index_col=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
