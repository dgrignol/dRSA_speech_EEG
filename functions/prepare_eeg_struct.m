function EEG = prepare_eeg_struct(subjectNum, runNum, basePath, includeMastoids)
% PREPARE_EEG_STRUCT -ASSUME EEG and AUDIO have same fs
% -Load EEG data
% -Trim to audio length
% -Format into EEGLAB-compatible structure.
%
% EEG = prepare_eeg_struct(subjectNum, runNum, basePath, includeMastoids)
%
% Inputs:
%   - subjectNum      : Integer, subject number (e.g., 1–19)
%   - runNum          : Integer, run number (e.g., 1–20)
%   - basePath        : String, base folder containing data files
%   - includeMastoids : Logical, whether to include mastoid channels (default: false)
%
% Output:
%   - EEG             : EEGLAB EEG structure

    if nargin < 4
        includeMastoids = false;
    end

    % Build filename
    subjStr = sprintf('Subject%d', subjectNum);
    runStr = sprintf('Run%d', runNum);
    filePath = fullfile(basePath, subjStr, sprintf('%s_%s.mat', subjStr, runStr));

    if ~isfile(filePath)
        error('File not found: %s', filePath);
    end

    % Load data
    load(filePath, 'eegData', 'fs', 'mastoids');  % Transpose if needed
    eegData = eegData';  % [chan x time]
    if includeMastoids
        mastoids = mastoids';
        data = [eegData; mastoids];  % [130 x time]
    else
        data = eegData;  % [128 x time]
    end
    
    % Trim to audio
    load('audio_lengths.mat')
    if lengths(runNum) <= size(data,2)
        % Trim EEG to match audio length
        data = data(:,1:lengths(runNum));
    else
        % Pad EEG with the last samples to match audio length
        diffLen = lengths(runNum) - size(data,2);
        padSegment = data(:, end-diffLen+1:end); % last diffLen samples
        data = [data padSegment];
        warning('Subj %d Run %d: EEG padded by %d samples (EEG=%d, Audio=%d)', ...
            subjectNum, runNum, diffLen, size(data,2), lengths(runNum));
        
    end


    % Create EEG structure
    EEG = pop_importdata('dataformat', 'array', 'data', data, ...
        'srate', fs, ...
        'nbchan', size(data, 1));


    % Load standard Biosemi 128 locations
    EEG = populate_chanlocs_from_csv(EEG, fullfile(basePath,'biosemi128.csv'));
    
    EEG.setname = sprintf('%s_%s', subjStr, runStr);
    EEG.filename = [EEG.setname '.set'];
    EEG.filepath = fullfile(basePath, subjStr);
    EEG.subject = subjStr;

    % Final check
    EEG = eeg_checkset(EEG);
end