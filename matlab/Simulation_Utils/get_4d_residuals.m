function [E, nii_hdr] = get_4d_residuals(Dirs)
    E = [];   
    nii_hdr = [];
        for i = 1 : Dirs.n_res 
%                 res = niftiread(Dirs.res_files{i});
                res = load_untouch_nii(Dirs.res_files{i});
                E = cat(4, E, res);
        end
    nii_hdr = load_untouch_header_only(Dirs.res_files{i});
end

