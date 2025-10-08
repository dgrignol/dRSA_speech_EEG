function [EEG, maskFromCleanRawFun] = preprocess_Heilbron_EEG(EEG, do_interp)
% PREPROCESS_HEILBRON_EEG - Preprocess EEG as per Heilbron et al. (2022)
%
% Inputs:
%   EEG          - EEG struct (with fields 'data', 'srate', etc.)
%   do_interp    - (true/false) whether to interpolate bad channels
%   interp_chans - cell array of bad channel labels to interpolate (e.g. {'A1','D9'})
%
% Output:
%   EEG - EEGLAB EEG structure
    

    % keep a copy that still contains all channels 
    originalEEG = EEG;           % save channel locations & labels

    % run Clean Rawdata 
    
    [EEG_completeRej,~,EEG] = clean_artifacts(EEG); % first output with time windows rejection; third has no winodow rejection
    % check length is the same
    if size(EEG.data,2)  ~= size(originalEEG.data,2)
        error('Dim not matching EEG_completeRej and EEG. Check clean_artifacts parameters')
    end
    maskFromCleanRawFun = EEG_completeRej.etc.clean_sample_mask==0; % save mask
    
    % --> bad channels are now *gone* from EEG.chanlocs

    rejection = 'ICLabel';
    EEG = ICA(EEG, rejection);

%     % Remove blink components (manual or automated ICLabel here if desired)
%     % For example, open the GUI for manual rejection:
%     EEG = pop_selectcomps(EEG, 1:30); % Adjust number as needed
%     EEG = pop_subcomp(EEG); % Removes manually rejected components
%     
    if do_interp
        % put the deleted channels back by interpolation 
        EEG = pop_interp(EEG, originalEEG.chanlocs , 'spherical');  % default    
    end
    
%     % Interpolate bad channels if needed
%     if do_interp && ~isempty(interp_chans)
%         EEG = pop_interp(EEG, interp_chans, 'spherical');
%     else
%         % Prompt user for manual channel rejection (visual inspection)
%         pop_eegplot(EEG, 1, 1, 1); % interactive plotting for visual inspection
%         interp_chans = inputdlg('Enter bad channels to interpolate (comma-separated):', ...
%                                  'Manual channel selection', [1 50], {''});
%         interp_chans = strtrim(strsplit(interp_chans{1}, ','));
%         EEG = pop_interp(EEG, interp_chans, 'spherical');
%     end

    % Save dataset before the band-pass filter so we can tweak filters without rerunning ICA
    [~, baseName, ~] = fileparts(EEG.filename);
    prefilterFilename = sprintf('%s_prefilt.set', baseName);
    EEG_prefilt = EEG;
    EEG_prefilt.stage = 'ica_rej_prefilt';
    EEG_prefilt.filename = prefilterFilename;
    EEG_prefilt.filepath = EEG.filepath;
    pop_saveset(EEG_prefilt, 'filename', prefilterFilename, 'filepath', EEG.filepath, 'savemode', 'onefile');

    % Band-pass filter: 0.5–8 Hz, zero-phase FIR
    originalLen = size(EEG.data, 2);
    EEG = pop_eegfiltnew(EEG, 0.5, 8, [], 0, [], 0);
    
    % Remove any mirrored padding to maintain alignment with the audio
    filteredLen = size(EEG.data, 2);
    extraSamples = filteredLen - originalLen;
    if extraSamples > 0
        trimLeading = floor(extraSamples / 2);
        trimTrailing = extraSamples - trimLeading;
        firstPoint = trimLeading + 1;
        lastPoint = filteredLen - trimTrailing;
        EEG = pop_select(EEG, 'point', [firstPoint, lastPoint]);
    end

    % Reference (optional)
    EEG = pop_reref(EEG, []);

    % Return final dataset
    EEG = eeg_checkset(EEG);
end
