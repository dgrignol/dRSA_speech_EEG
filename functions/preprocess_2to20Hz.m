function [EEG, maskFromCleanRawFun] = preprocess_2to20Hz(EEG, maskFile)
% PREPROCESS_2TO20HZ Apply a 2-20 Hz band-pass to a prefilt EEG dataset.
%
%   [EEG, maskFromCleanRawFun] = preprocess_2to20Hz(EEG, maskFile)
%
% Inputs:
%   EEG      - EEGLAB EEG structure already cleaned with CleanRaw/ICA.
%   maskFile - Path to the CleanRaw sample mask saved during Heilbron preprocessing.
%
% Output:
%   EEG                  - EEGLAB structure after 2-20 Hz filtering and trimming.
%   maskFromCleanRawFun  - Logical mask (1 = rejected samples) from CleanRaw.
%
% The function assumes `EEG` corresponds to the "*_prefilt.set" dataset
% saved by `preprocess_Heilbron_EEG` before the spectral filtering stage.
% After filtering we remove the mirrored padding introduced by
% `pop_eegfiltnew` so the time axis matches the original audio.

    % Load CleanRaw mask if available
    if exist(maskFile, 'file')
        tmp = load(maskFile);
        if isfield(tmp, 'maskFromCleanRawFun')
            maskFromCleanRawFun = tmp.maskFromCleanRawFun;
        else
            warning('Mask file %s does not contain maskFromCleanRawFun. Using zeros.', maskFile);
            maskFromCleanRawFun = zeros(1, size(EEG.data, 2));
        end
    else
        warning('Mask file not found: %s. Using zeros.', maskFile);
        maskFromCleanRawFun = zeros(1, size(EEG.data, 2));
    end

    % Apply 2-20 Hz band-pass filter
    originalLen = size(EEG.data, 2);
    EEG = pop_eegfiltnew(EEG, 2, 20, [], 0, [], 0);

    % Remove mirrored padding introduced by zero-phase filtering
    filteredLen = size(EEG.data, 2);
    extraSamples = filteredLen - originalLen;
    if extraSamples > 0
        trimLeading = floor(extraSamples / 2);
        trimTrailing = extraSamples - trimLeading;
        firstPoint = trimLeading + 1;
        lastPoint = filteredLen - trimTrailing;
        EEG = pop_select(EEG, 'point', [firstPoint, lastPoint]);
    end

    % Re-reference (common average)
    EEG = pop_reref(EEG, []);

    % Final consistency check
    EEG = eeg_checkset(EEG);

    % Ensure mask length matches the EEG length after trimming
    eegLen = size(EEG.data, 2);
    if numel(maskFromCleanRawFun) > eegLen
        maskFromCleanRawFun = maskFromCleanRawFun(1:eegLen);
    elseif numel(maskFromCleanRawFun) < eegLen
        maskFromCleanRawFun = [maskFromCleanRawFun zeros(1, eegLen - numel(maskFromCleanRawFun))];
    end

    maskFromCleanRawFun = maskFromCleanRawFun(:)';
end
