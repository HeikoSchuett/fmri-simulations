function [E, nii_hdr] = get_4d_residuals(Dirs)
    nii_hdr = load_untouch_header_only(Dirs.res_files{1});
    E = load_untouch_nii_img(Dirs.res_files{1});
    E = repmat(E, 1, 1, 1, Dirs.n_res)
    for i = 2 : Dirs.n_res 
%                 res = niftiread(Dirs.res_files{i});
        res = load_untouch_nii_img(Dirs.res_files{i});
        E(:,:,:,i) = res;
    end
end
