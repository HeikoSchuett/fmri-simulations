function [Y, betas_mat, Opts] = get_permuted_GLM_pred(mask_vec, perm, Dirs, Opts)
cd(Dirs.input_dir)
load('SPM.mat')

% Designate stimulus-related design matrix columns
% n_reg = size(SPM.xX.X,2);
% basis_init = eye(Opts.n_stim_betas);
% zero_pads = zeros(n_reg - Opts.n_stim_betas, Opts.n_stim_betas);
% basis = vertcat(basis_init, zero_pads);

% Load and reshape betas
B  = spm_data_read(SPM.Vbeta);
Opts.size_B = size(B);
betas_mat = reshape(B, [], size(B, 4));

Y = zeros(size(SPM.xX.xKXs.X, 1), sum(mask_vec), Opts.n_permutations);

for i = 1:Opts.n_permutations
    betas_mat_masked = betas_mat(mask_vec, [perm(:, i); Opts.size_B(4)])';
    Y(:,:,i) = SPM.xX.xKXs.X*betas_mat_masked;
end
% calculate fitted (predicted) data (Y = X1*beta)
% Y = SPM.xX.xKXs.X*basis*pinv(basis)*betas_mat_masked;
cd(Dirs.output_dir)
end




