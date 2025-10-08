clearvars;
close all;
clc;

paths = load_paths_config();
addpath(paths.eeglab)
addpath(paths.functions)

basePathStimuli = paths.dataStimuli;
basePathEEG = paths.dataEEG;
basePathMasks = paths.masks;
targetFs = 128;
numOfStim = 20;
sartEnd_span = 3;
preproc_type_for_check = '0.01to20Hz';

if ~isfolder(basePathStimuli)
    error('A2_preprocess_stimuli:StimuliPathMissing', 'Stimuli folder not found: %s', basePathStimuli);
end

if ~isempty(paths.eeglab)
    eeglab nogui;
else
    warning('A2_preprocess_stimuli:EEGLABPathMissing', 'EEGLAB path not set in config_paths.json.');
end

mergedAudioPath = fullfile(basePathStimuli, 'Audio', 'audio_merged.wav');
if isfile(mergedAudioPath)
    fprintf('Merged audio already present: %s\n', mergedAudioPath);
else
    fprintf('Resampling audio files to %d Hz...\n', targetFs);
    resample_audio_files(basePathStimuli, numOfStim, targetFs);

    eegMergedFile = sprintf('Subject%d_ICA_rej_preproc%s_merged.set', 1, preproc_type_for_check);
    eegMergedPath = fullfile(basePathEEG, 'Subject1', eegMergedFile);

    if ~isfile(eegMergedPath)
        error('A2_preprocess_stimuli:MergedEEGMissing', ...
              'Merged EEG dataset for Subject 1 not found: %s', eegMergedPath);
    end

    EEG_merged = pop_loadset(eegMergedFile, fullfile(basePathEEG, 'Subject1'));

    stimuliDir = fullfile(basePathStimuli, 'Audio');
    testMode = false;
    [mergedAudio, fs_check, maskConcat] = merge_audio_files(stimuliDir, basePathMasks, testMode, sartEnd_span);

    if numel(mergedAudio) ~= numel(EEG_merged.data)
        error('A2_preprocess_stimuli:LengthMismatch', ...
              'Length of merged audio (%d samples) does not match EEG data (%d samples).', ...
              numel(mergedAudio), numel(EEG_merged.data));
    end

    fprintf('Audio preprocessing complete. Output saved to: %s\n', mergedAudioPath);
    fprintf('Resampled fs reported by merge_audio_files: %g Hz\n', fs_check);
    fprintf('Mask saved to: %s\n', fullfile(basePathMasks, 'mask_concat.mat'));
end
