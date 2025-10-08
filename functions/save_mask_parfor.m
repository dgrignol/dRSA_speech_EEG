function save_mask_parfor(fname, maskFromCleanRawFun)
    % Helper for saving inside parfor
    save(fname, 'maskFromCleanRawFun', '-v7.3');
end