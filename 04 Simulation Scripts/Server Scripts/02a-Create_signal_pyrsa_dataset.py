#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline and utility functions for generating simulated rsatoolbox datasets
for volumetric ROIs (from freesurfer)
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd
from nibabel import nifti1
import mask_utils
import rsatoolbox


def rewrite_label_dict(label_dict):
    vals = list(label_dict.values())
    keys_new = [str(i) for i in range(1, 51)]
    return dict(zip(keys_new, vals))


def stim_id_to_label(stim_ids, label_dict, n_stim=None, crop=False):
    """
    Transform beta_id to object labels

    Inputs:
    - stim_ids (list / df): sorted list of all GLM predictors
    - n_stim: The first n_stim GLM predictors which represent
              stimulus identities of interest
    - label_dict (dict): {synset ID : object label}
    - crop (bool): Use only first term in object label before the comma

    Output:
    - labels (list): sorted list of the stimulus object labels
                     corresponding to GLM predictors
    """
    labels = []
    if n_stim is None:
        n_stim = len(stim_ids)

    for i in range(n_stim):
        synset = stim_ids["RegressorNames"].astype(str)[:n_stim][i]
        if crop:
            labels.append(label_dict[synset].split(",")[0])
        else:
            labels.append(label_dict[synset])

    return labels


def get_stim_desc(descriptors_path, label_dict=None):

    stim_ids = pd.read_csv(descriptors_path, sep="[, _]", header=None, engine="python")
    stim_ids.columns = ["row", "RegressorNames", "run"]
    runs = stim_ids["run"].to_list()
    labels = stim_id_to_label(stim_ids, label_dict, n_stim=None, crop=True)
    synset = stim_ids["RegressorNames"]
    return labels, synset, runs


def cronbachs_alpha(stim_measurements):
    cmatrix = np.corrcoef(stim_measurements, rowvar=True)
    N = len(cmatrix)
    rs = list(cmatrix[np.triu_indices(N, k=1)])
    mean_r = np.mean(rs)
    alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
    return alpha


def remove_empty_voxels(pooled_observations, voxel_ids=None, nan_threshold=0):
    """
    After mask smoothing, resampling and thresholding,
    some mask voxels end up outside of the brain and need to be removed

    Args:
        pooled_observations (ndarray):
            n_obs resp. n_res x n_voxels (either measurements or residuals)
        voxel_ids (ndarray):
            n_voxel x 3 - dimensional array of the X,Y,Z coordinates
            of voxels that are contained in mask
        nan_threshold (float OR int):
            if between 0 and 1 - maximal percentage of allowed NaN entries
            if int > 1 - maximal number of allowed NaN entries

    Returns:
        pooled_observations_cleaned (ndarray):
            n_obs resp. n_res x n_brain_voxels
        voxel_ids_cleaned (ndarray):
            n_brain_voxels x 3 - dimensional array of the X,Y,Z coordinates
            of voxels that are contained in mask
        bad_cols (list):
            indices of critical voxels

    """
    assert nan_threshold >= 0, "nan_threshold must be a nonnegative number"

    obs_shape = np.shape(pooled_observations)

    if voxel_ids is not None:
        ROI_shape = np.shape(voxel_ids)
        assert (
            obs_shape[1] == ROI_shape[0]
        ), "Dimension mismatch between data \
            array and number of voxels"

    if nan_threshold < 1:
        max_nans = np.around(obs_shape[0] * nan_threshold).astype(int)
    else:
        max_nans = nan_threshold.astype(int)

    nan_rows = np.sum(np.isnan(pooled_observations).astype(int), axis=0)
    bad_cols = np.where(nan_rows > max_nans)

    if len(bad_cols[0]) > 0:
        print(
            "\nOut of",
            obs_shape[0],
            "observations",
            np.round(len(nan_rows) / obs_shape[0] * 100, 1),
            "% contained at least one NaN-value.",
        )
        print(
            "The following voxels contained at least",
            max_nans,
            "NaN-values:\n",
            bad_cols[0],
            "\n",
        )
        print(
            "These",
            len(bad_cols[0]),
            "voxels made up",
            np.round(len(bad_cols[0]) / obs_shape[1] * 100, 1),
            "% of all voxels in this ROI and will be removed"
            + " from further analyses.\n",
        )
        pooled_observations_cleaned = np.nan_to_num(
            np.delete(pooled_observations, bad_cols, axis=1)
        )
    else:
        pooled_observations_cleaned = pooled_observations

    if voxel_ids is not None:
        voxel_ids_cleaned = np.delete(voxel_ids, bad_cols, axis=0)
    else:
        voxel_ids_cleaned = None

    return pooled_observations_cleaned, voxel_ids_cleaned, bad_cols[0]


def remove_empty_voxels_pipeline(
    measurements=None, residuals=None, voxel_ids=None, nan_threshold=0
):
    """
    Pipeline for remove_empty_voxels for every processing_mode

    Args:
        measurements (ndarray):
            numpy array (pattern vector x ROI size)
        residuals (ndarray):
            numpy array (residual vector x ROI size)
        voxel_ids (ndarray):
            n_voxel x 3 - dimensional array of the X,Y,Z coordinates
            of voxels that are contained in mask
        nan_threshold (float OR int):
            if between 0 and 1 - maximal percentage of allowed NaN entries
            if int > 1 - maximal number of allowed NaN entries

    Returns:
        measurements_cleaned (ndarray):
            n_obs x n_brain_voxels
        residuals_cleaned (ndarray):
            n_res x n_brain_voxels
        voxel_ids_cleaned (ndarray):
            n_brain_voxels x 3 - dimensional array of the X,Y,Z coordinates
            of voxels that are contained in mask

    """

    assert isinstance(measurements, np.ndarray) or isinstance(
        residuals, np.ndarray
    ), "Provide at least one nonempty array."

    measurements_cleaned = []
    residuals_cleaned = []
    voxel_ids_cleaned = []

    if isinstance(measurements, np.ndarray):
        measurements_cleaned, voxel_ids_cleaned, bad_cols_m = remove_empty_voxels(
            measurements, voxel_ids=voxel_ids, nan_threshold=nan_threshold
        )

    if isinstance(residuals, np.ndarray):
        if len(voxel_ids_cleaned) == 0:
            residuals_cleaned, voxel_ids_cleaned, bad_cols_r = remove_empty_voxels(
                residuals, voxel_ids=voxel_ids, nan_threshold=nan_threshold
            )
        else:
            residuals_cleaned, _, bad_cols_r = remove_empty_voxels(
                residuals, voxel_ids=None, nan_threshold=nan_threshold
            )

    if (
        isinstance(measurements, np.ndarray)
        and isinstance(residuals, np.ndarray)
        and not np.array_equal(bad_cols_m, bad_cols_r)
    ):
        print(
            "There was a mismatch between bad voxels in measurements"
            + "and residuals, bad voxel indices will be pooled"
        )
        bad_cols = np.union1d(bad_cols_m, bad_cols_r).astype(int)
        measurements_cleaned = np.nan_to_num(np.delete(measurements, bad_cols, axis=1))
        residuals_cleaned = np.nan_to_num(np.delete(residuals, bad_cols, axis=1))
        voxel_ids_cleaned = np.delete(voxel_ids, bad_cols, axis=0)

    return measurements_cleaned, residuals_cleaned, voxel_ids_cleaned


###############################################################################


def main():
    # Data analysis parameters
    spm_type = "Dual_GLM"
    task = "perception"
    stimulus_set = "Test"
    ses_type = task + "_" + stimulus_set
    save_dataset = True
    delete_inputs = False

    # Set directories, specify ROIs and load dictionary for labels
    ds_dir = os.environ.get("SOURCE")
    # directory of mask descriptors
    spm_dir = os.path.join(os.environ.get("SOURCE"), "derivatives", spm_type)
    # directory in which subject-specific volumetric ROI masks are saved by FS
    freesurfer_mri = "mri_glasser"
    label_dict = np.load(
        os.path.join(ds_dir, "custom_synset_dictionary.npy"), allow_pickle="TRUE"
    ).item()
    # label_dict = rewrite_label_dict(label_dict_o)

    n_subs = len(glob.glob(spm_dir + os.sep + "sub*"))

    ##############################################################################

    for sub in range(1, n_subs + 1):

        # Set output directories
        ds_output_dir = os.path.join(
            ds_dir, "derivatives", "PyRSA_GT", "datasets", "sub-" + str(sub).zfill(2)
        )
        if not os.path.isdir(ds_output_dir):
            os.makedirs(ds_output_dir)
        res_output_dir = os.path.join(
            ds_dir, "derivatives", "PyRSA_GT", "noise", "sub-" + str(sub).zfill(2)
        )
        if not os.path.isdir(res_output_dir):
            os.makedirs(res_output_dir)

        # Load mask dictionaries and 4d image
        mask_dir = os.path.join(
            ds_dir,
            "derivatives",
            "freesurfer",
            "sub-" + str(sub).zfill(2),
            freesurfer_mri,
        )
        mask_dict = mask_utils.load_dict(
            os.path.join(
                mask_dir, "sub-" + str(sub).zfill(2) + "_mask_dict_EPI_disjoint.npy"
            )
        )
        voxel_ids_dict = mask_utils.load_dict(
            os.path.join(
                mask_dir,
                "sub-" + str(sub).zfill(2) + "_voxel_IDs_dict_EPI_disjoint.npy",
            )
        )
        voxel_ids_dict_tmp = voxel_ids_dict.copy()

        # Collect measurements as well as respective descriptors
        glm_dir = os.path.join(spm_dir, "sub-" + str(sub).zfill(2))
        signal_4d = None
        signal_path = os.path.join(glm_dir, ses_type + "_signal.nii.gz")
        descriptors_path = os.path.join(glm_dir, ses_type + "_signal.csv")
        signal_4d = nifti1.load(signal_path)

        noise_4d = None
        noise_path = os.path.join(glm_dir, ses_type + "_noise.nii.gz")
        noise_4d = nifti1.load(noise_path)

        stimulus_descriptors, synset, run_descriptors = get_stim_desc(
            descriptors_path, label_dict=label_dict
        )
        # Generate and save rsatoolbox dataset and pooled residuals for each ROI
        for roi_h in mask_dict.keys():
            measurements = mask_utils.apply_roi_mask_4d(
                signal_4d, mask_dict[roi_h], method="custom"
            )
            residuals = mask_utils.apply_roi_mask_4d(
                noise_4d, mask_dict[roi_h], method="custom"
            )

            # Remove voxels that are outside of the brain
            # (artifact of mask smoothing, resampling and thresholding)
            measurements_cleaned, residuals_cleaned, voxel_ids_cleaned = [], [], []
            (
                measurements_cleaned,
                residuals_cleaned,
                voxel_ids_cleaned,
            ) = remove_empty_voxels_pipeline(
                measurements=measurements,
                residuals=residuals,
                voxel_ids=voxel_ids_dict[roi_h],
                nan_threshold=0.01,
            )
            voxel_ids_dict_tmp[roi_h] = voxel_ids_cleaned

            if save_dataset:
                # Create rsatoolbox dataset
                dataset = rsatoolbox.data.Dataset(
                    measurements_cleaned,
                    descriptors={"ROI": roi_h},
                    obs_descriptors={
                        "stim": stimulus_descriptors,
                        "synset": [str(i) for i in synset],
                        "run": run_descriptors,
                    },
                    channel_descriptors={"positions": voxel_ids_dict_tmp[roi_h]},
                )

                # Save dataset and residuals array
                dataset_filename = os.path.join(
                    ds_output_dir, "RSA_dataset_" + roi_h + "_signal"
                )
                dataset.save(dataset_filename, file_type="hdf5", overwrite=True)
                print("Created pyrsa dataset:", dataset_filename)

                residuals_filename = os.path.join(
                    res_output_dir, "Residuals_" + roi_h + "_signal"
                )
                np.save(residuals_filename, residuals_cleaned)
                print("Created residuals dataset:", residuals_filename)

        if delete_inputs:
            shutil.rmtree(glm_dir)


if __name__ == "__main__":
    main()
