#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline for pooling the results of SPM's GLM for simulated results

@author: alex
"""
import os
import glob
import shutil
import numpy as np
import pandas as pd
from nibabel import nifti1


def get_perm_range(img_output_dir, ses_type):
    files = glob.glob(os.path.join(img_output_dir, ses_type + "*", "run*", "*perm*"))
    p_num = [int(file.split("_")[-3]) for file in files]
    return np.unique(p_num)


def load_stim_ids_dict(
    beta_descriptors_path, first_n_betas=None, beta_vector=None, tosynset=False
):
    """
    Read in a list of IDs for the beta coefficient outputs of SPM to get
    """
    events = pd.read_csv(beta_descriptors_path, sep="[\t]", engine="python")
    regressor_names = np.array(events["RegressorNames"].to_list())
    regressor_indices = np.array(events.index.values) + 1
    n_betas = len(regressor_names)

    if not isinstance(beta_vector, (np.ndarray, list)):
        beta_vector = np.ones(n_betas, bool)
        if isinstance(first_n_betas, int) and first_n_betas <= n_betas:
            beta_vector[first_n_betas:n_betas] = 0

    stim_ids = regressor_names[beta_vector]
    stim_indices = regressor_indices[beta_vector]

    if tosynset:
        stim_ids = convert_to_synset(stim_ids)

    stim_ids_dict = dict(zip(stim_ids, stim_indices))
    return stim_ids_dict


def convert_to_synset(stim_ids):
    synset_ids = []
    for stim in stim_ids:
        synset_ids.append("n0" + stim.split(".")[0])
    return synset_ids


def pooled_sim_betas_pipeline(
    sub, spm_dir, ses_type, snr=1, perm=1, first_n_betas=None
):
    run_counter = 0
    fourth_dimension_descriptors = []
    beta_array_superset = []
    n_ses = len(
        glob.glob(os.path.join(spm_dir, "sub-" + str(sub).zfill(2), ses_type + "*"))
    )

    for ses in range(1, n_ses + 1):
        ses_dir = os.path.join(
            spm_dir, "sub-" + str(sub).zfill(2), ses_type + str(ses).zfill(2)
        )
        n_runs = len(glob.glob(ses_dir + os.sep + "run*"))

        for run in range(1, n_runs + 1):
            run_counter += 1
            if perm is None:
                glm_dir = os.path.join(ses_dir, "run-" + str(run).zfill(2))
            else:
                glm_dir = os.path.join(
                    ses_dir,
                    "run-" + str(run).zfill(2),
                    "GLM_Data_perm_mixed_" + str(perm).zfill(4) + "_snr_" + str(snr),
                )
            beta_descriptors_path = os.path.join(glm_dir, "spm_beta_ids.txt")
            stim_ids_dict = load_stim_ids_dict(
                beta_descriptors_path, first_n_betas=first_n_betas, tosynset=True
            )

            for condition in stim_ids_dict.keys():
                num = stim_ids_dict[condition]
                beta_image_path = os.path.join(glm_dir, "beta_" + str(num).zfill(4))
                beta_image = nifti1.load(beta_image_path)
                beta_array_superset.append(beta_image.get_fdata())
                fourth_dimension_descriptors.append(condition + "_" + str(run_counter))

    # Get affine matrix
    generic_affine = beta_image.affine.copy()

    # Pool the arrays
    pooled_beta_array = np.stack(beta_array_superset, axis=3)

    return pooled_beta_array, generic_affine, fourth_dimension_descriptors


def pooled_sim_residuals_pipeline(sub, spm_dir, ses_type, snr=1, perm=1):
    run_counter = 0
    fourth_dimension_descriptors_r = []
    residual_array_superset = []
    n_ses = len(
        glob.glob(os.path.join(spm_dir, "sub-" + str(sub).zfill(2), ses_type + "*"))
    )

    for ses in range(1, n_ses + 1):
        ses_dir = os.path.join(
            spm_dir, "sub-" + str(sub).zfill(2), ses_type + str(ses).zfill(2)
        )
        n_runs = len(glob.glob(ses_dir + os.sep + "run*"))

        for run in range(1, n_runs + 1):
            run_counter += 1
            if perm is None:
                glm_dir = os.path.join(ses_dir, "run-" + str(run).zfill(2))
            else:
                glm_dir = os.path.join(
                    ses_dir,
                    "run-" + str(run).zfill(2),
                    "GLM_Data_perm_mixed_" + str(perm).zfill(4) + "_snr_" + str(snr),
                )

            n_res = len(glob.glob(os.path.join(glm_dir, "Res_*")))
            for res in range(1, n_res + 1):
                res_image_path = os.path.join(glm_dir, "Res_" + str(res).zfill(4))
                res_image = nifti1.load(res_image_path)
                residual_array_superset.append(np.array(res_image.get_fdata()))
                fourth_dimension_descriptors_r.append(
                    str(res).zfill(4) + "_" + str(run_counter)
                )

    # Get affine matrix
    generic_affine_r = res_image.affine.copy()

    # Pool the arrays
    pooled_res_array = np.stack(residual_array_superset, axis=3)
    return pooled_res_array, generic_affine_r, fourth_dimension_descriptors_r


##############################################################################


def main(sub=1, data=False):
    if not data:
        # Data analysis parameters: Simulation
        spm_type = "Data_perm"
        task = "perception"
        stimulus_set = "Test"
        ses_type = "ses-" + task + stimulus_set
        first_n_betas = 50
        snr_range = [0.1, 1, 10]
        delete_inputs = True

        # Set directories, specify ROIs and load dictionary for labels
        ds_dir = os.environ.get("SOURCE")
        # spm_dir = os.path.join(ds_dir, "derivatives", spm_type)
        spm_dir = os.path.join(os.environ.get("INTERMEDIATE"), spm_type)
        img_output_dir = os.path.join(spm_dir, "sub-" + str(sub).zfill(2))
        nifti_filename = os.path.join(
            img_output_dir,
            task + "_" + stimulus_set + "_data_perm_mixed_{perm:04d}_snr_{snr}",
        )
        perm_range = get_perm_range(img_output_dir, ses_type)
    else:
        # Data analysis parameters: True
        spm_type = "Dual_GLM"
        task = "perception"
        stimulus_set = "Test"
        ses_type = "ses-" + task + stimulus_set
        first_n_betas = 50
        snr_range = [None]
        delete_inputs = False

        # Set directories, specify ROIs and load dictionary for labels
        ds_dir = os.environ.get("SOURCE")
        spm_dir = os.path.join(ds_dir, "derivatives", spm_type)
        img_output_dir = os.path.join(spm_dir, "sub-" + str(sub).zfill(2))
        nifti_filename = os.path.join(img_output_dir, task + "_" + stimulus_set)
        perm_range = [None]

    processing_mode = "both"  # Options: 'datasets', 'residuals' or 'both'

    ##############################################################################

    for snr in snr_range:
        for perm in perm_range:
            if not data:
                nifti_fn = nifti_filename.format(perm=perm, snr=snr)
            else:
                nifti_fn = nifti_filename
            csv_filename = nifti_fn + "_signal.csv"
            csv_filename_r = nifti_fn + "_noise.csv"
            nifti_filename_r = nifti_fn + "_noise.nii.gz"
            nifti_fn = nifti_fn + "_signal.nii.gz"
            if processing_mode in ["datasets", "both"]:
                # Load and stack 3d arrays for each GLM predictor for each run
                (
                    pooled_betas_array,
                    generic_affine,
                    fourth_dimension_descriptors,
                ) = pooled_sim_betas_pipeline(
                    sub,
                    spm_dir,
                    ses_type,
                    snr=snr,
                    perm=perm,
                    first_n_betas=first_n_betas,
                )

                # Make subject-specific 4d nifti image of beta coeffients
                pooled_betas = nifti1.Nifti1Image(pooled_betas_array, generic_affine)
                nifti1.save(pooled_betas, nifti_fn)
                print("Pooled subject data to:", nifti_fn)

                # Save corresponding descriptors for the 4th dimension
                df = pd.DataFrame({"descriptor": fourth_dimension_descriptors})
                df.to_csv(csv_filename, header=False)

            if processing_mode in ["residuals", "both"]:
                # Load and stack 3d arrays of all residuals for each run
                (
                    pooled_residuals_array,
                    generic_affine_r,
                    fourth_dimension_descriptors_r,
                ) = pooled_sim_residuals_pipeline(
                    sub, spm_dir, ses_type, snr=snr, perm=perm
                )

                # Make subject-specific 4d nifti image of residuals
                pooled_residuals = nifti1.Nifti1Image(
                    pooled_residuals_array, generic_affine_r
                )
                nifti1.save(pooled_residuals, nifti_filename_r)
                print("Pooled subject noise to:", nifti_filename_r)

                # Save corresponding descriptors for the 4th dimension
                df_r = pd.DataFrame({"descriptor": fourth_dimension_descriptors_r})
                df_r.to_csv(csv_filename_r, header=False)

    if delete_inputs:
        to_be_deleted = glob.glob(
            os.path.join(spm_dir, "sub-" + str(sub).zfill(2), ses_type + "*")
        )
        for d in to_be_deleted:
            shutil.rmtree(d)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--sub", help="Subject to run [1..5]", type=int, default=1
    )
    parser.add_argument("--data", help="run original data", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
