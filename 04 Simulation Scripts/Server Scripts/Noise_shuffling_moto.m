%==========================================================================
%     Noise shuffling and scaling while preserving serial autocorrelations
%     and spatial structure (w/ AR(2) model)
%==========================================================================

%% 1. Preparations
%----- Custom paths
clc
clear all
format long g
Dirs.BIDSdir = '/moto/nklab/projects/ds001246_r/';
matlab_docs = '/moto/home/hs3110/fmri-simulations/matlab';
% cd(matlab_docs);
addpath(genpath(fullfile(matlab_docs, 'nifti_utils')));
addpath(fullfile(matlab_docs, 'spm12'));
addpath(fullfile(matlab_docs, 'Simulation_Utils')); 

Opts = struct();
Opts.n_permutations = 2;
Opts.ar_n = 2;
Opts.task = 'perception';
Opts.subtask = 'Test';
Opts.session_type = [Opts.task, Opts.subtask];
Opts.n_stim_betas = 50;
Opts.pool_inference = false;
Opts.rewrite = true; % overwrites previously saved outputs
Dirs = parse_bids_base_name(Dirs, 'Noise_perm_r'); % Parse BIDS directory
Dirs.GLM_results = fullfile(Dirs.BIDSdir, 'derivatives', 'Dual_GLM_r');

for i = 1 : Dirs.n_subs
    Dirs = parse_bids_sub(Dirs, Opts, i);
    r = 0;
    for s = 1 : Dirs.n_ses  
        Dirs = get_runs(Dirs, s);

        for n = 1 : Dirs.n_runs
            
            r = r+1;
            Dirs = add_res_files(Dirs, Opts, i, s, n);
            
            % Get and mask residuals
            [~, res_mat, mask_vec, Opts] = load_residual_matrix(Dirs, Opts);
            y = res_mat(mask_vec,:)';
            
            % Get AR(n) predictions (w/ least squares) and residuals
            [y_ar, eps, A] = ar_n_bare_metal(y, Opts.ar_n);
            
            % Get and apply Opts.n_permutations permutations to AR(n)
            % residuals (eps) and re-add them to AR(n) predictions
            p_mat = get_permutation_vectors(Opts);
            y_perm = permute_residuals_f(A, eps, p_mat, Opts);
            
            % Reshape permuted residual time series back into original
            % shape of the nii object and save it
            E_perm = repaste_to_5d(res_mat, y_perm, ...
                mask_vec, Opts);
            save_as_nii(E_perm, Dirs, Opts, i, r, 1, 'Res_perm');

            % Get predicted time series of GLM, reshape and save them
            [y_glm, ~, Opts] = get_GLM_predicted_timeseries(mask_vec, Dirs, Opts);
            S = repaste_to_4d(res_mat, y_glm, mask_vec, Opts);   
            save_as_nii(S, Dirs, Opts, i, r, 1, 'Signal');
            
        end
    end
end
