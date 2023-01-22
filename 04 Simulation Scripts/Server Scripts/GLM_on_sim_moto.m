function GLM_on_sim_moto(i)
%==========================================================================
%     Run-wise first-level GLM on simulated data
%==========================================================================
% for subject i 

%% 1. Preparations
rng('shuffle')
%----- Custom paths
format long g
Dirs.BIDSdir = getenv('SOURCE');
matlab_docs = getenv('MATLAB_CODE');
addpath(genpath(fullfile(matlab_docs, 'nifti_utils')));
addpath(fullfile(matlab_docs, 'spm12'));
addpath(fullfile(matlab_docs, 'Simulation_Utils')); 

Opts = struct();
Opts.snr = [10, 1, 0.1];
Opts.snr_type = 'total';
Opts.sim_type = 'mixed'; % 'mixed', 'signal', 'noise'
Opts.task = 'perception';
Opts.subtask = 'Test';
Opts.session_type = [Opts.task, Opts.subtask];
Opts = load_task_json(Opts, Dirs); % add task-related MRI specs to Opts
Opts.pool_inference = false;
Opts.rewrite = true; % overwrites previously saved outputs
Opts.delete_input = false;
Dirs = parse_bids_base_name(Dirs, 'Data_perm'); % Parse BIDS directory
Dirs.inputdir = fullfile(getenv('INTERMEDIATE'), 'Noise_perm');  % fullfile(Dirs.BIDSdir, 'derivatives', 'Noise_perm');
Dirs.GLM_results = fullfile(Dirs.BIDSdir, 'derivatives', 'Dual_GLM');
Dirs.outputdir = fullfile(getenv('INTERMEDIATE'), 'Data_perm');
spm('Defaults','fMRI'); %Initialise SPM fmri
spm_get_defaults('mask.thresh', -inf)
spm_jobman('initcfg');  %Initialise SPM batch mode

Dirs = parse_bids_sub(Dirs, Opts, i);
r = 0;

for s = 1 : Dirs.n_ses  
    Dirs = get_runs(Dirs, s);
    
    for n = 1 : Dirs.n_runs
        
        r = r+1;
        Dirs = add_sim_files(Dirs, Opts, i, s, n);
        for p = 1 : Dirs.n_permutations
            [S_mat_masked, S_mat, mask_vec, Opts] = ...
                get_masked_signal(Dirs, Opts, p);
            % S_power = get_timeseries_power(S_mat_masked, Opts);
            cd(Dirs.output_dir)
            [N_mat_masked, ~] = get_masked_residuals(Dirs, p, mask_vec);
            for snr_num = 1 : length(Opts.snr)
%                 N_power = get_timeseries_power(N_mat_masked, Opts);
%                 w = find_noise_scalar(S_power, N_power, Opts);
                snr = Opts.snr(snr_num);
                w = inv(snr);
                N_mat_masked_scaled = N_mat_masked .* w;
                D = generate_sim_4d(S_mat, S_mat_masked, ...
                    N_mat_masked_scaled, mask_vec, Opts);
                nii_file = save_as_nii(D, Dirs, Opts, i, r, p, ...
                    ['Data_perm_', Opts.sim_type]);
                level_1_GLM(nii_file, i, s, n, p, snr, Dirs, Opts)
                delete(nii_file);
            end
            
            if Opts.sim_type == "signal"
                break
            end
        end
        if Opts.delete_input
            rmdir(Dirs.input_dir, 's');
        end
        
    end
end
