function [S_mat_masked, S_mat, mask_vec, Opts] = get_masked_signal(Dirs, Opts, p)
    if ~exist('p', 'var') || isempty(p)
        p = 1;
    end
    cd(Dirs.input_dir)
    mask = niftiread('mask.nii');
    mask_vec = logical(mask(:));
    S = niftiread(Dirs.signal_file{p});
    Opts.nii_header = load_untouch_header_only(Dirs.signal_file{p});
    Opts.size_S = size(S);
    S_mat = reshape(S, [], size(S, 4));
    S_mat_masked = S_mat(mask_vec,:);
end