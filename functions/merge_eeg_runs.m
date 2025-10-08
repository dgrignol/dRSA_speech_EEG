function EEG_merged = merge_eeg_runs(subjectID, baseDir, numOfStim, fileSuffix)
% Merges EEG runs for one subject into a single EEG dataset, based on file suffix
% example file name - 'Subject1_Run1_ICA_rej_preprocHeilb.set'
% Inputs:
%   subjectID  - number, e.g., 11
%   baseDir    - path to base directory, e.g., '/path/to/ds004408'
%   numOfStim  - number of stimuli/runs to merge
%   fileSuffix - string, file suffix to load, e.g., 'preproc' or 'trimmed'
%
% Output:
%   EEG_merged - EEGLAB structure with merged data

    eegPath = fullfile(baseDir,  sprintf('Subject%d', subjectID));
    EEG_all = cell(1, numOfStim);

    % Load datasets with specified suffix
    for stimIdx = 1:numOfStim
        targetFile = sprintf('Subject%d_Run%d_%s.set', subjectID, stimIdx, fileSuffix);
        targetPath = fullfile(eegPath, targetFile);

        if exist(targetPath, 'file')
            EEG_all{stimIdx} = pop_loadset('filename', targetFile, 'filepath', eegPath);
        else
            error('❌ EEG file missing: %s\n', targetPath);
        end
    end

    % Merge datasets
    EEG_merged = EEG_all{1};
    for stimIdx = 2:numOfStim
        EEG_merged = pop_mergeset(EEG_merged, EEG_all{stimIdx});
    end

    % Save merged EEG
    mergedFile = sprintf('Subject%d_%s_merged.set', subjectID, fileSuffix);
    pop_saveset(EEG_merged, 'filename', mergedFile, 'filepath', eegPath, 'savemode','onefile');

    fprintf('✅ Merged EEG saved: %s\n', fullfile(eegPath, mergedFile));
end