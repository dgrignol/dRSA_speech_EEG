clearvars;
close all;
clc;

paths = load_paths_config();
addpath(paths.eeglab)
addpath(paths.functions)

basePathEEG = paths.dataEEG;
preproc_type = '0.01to20Hz'; % options: '0.01to20Hz', '2to20Hz', 'Heilb', 'Custom'
numOfStim = 20;
numOfSubj = 19;
overwrite_preproc = false;

if ~isfolder(basePathEEG)
    error('A1_preprocess_EEG:EEGPathMissing', 'EEG data folder not found: %s', basePathEEG);
end

if ~isempty(paths.eeglab)
    eeglab nogui;
else
    warning('A1_preprocess_EEG:EEGLABPathMissing', 'EEGLAB path not set in config_paths.json.');
end

for subjNum = 1:numOfSubj
    subjStr = sprintf('Subject%d', subjNum);
    subjFolder = fullfile(basePathEEG, subjStr);
    if ~isfolder(subjFolder)
        mkdir(subjFolder);
    end

    for runNum = 1:numOfStim
        runStr = sprintf('Run%d', runNum);
        filename = sprintf('%s_%s.set', subjStr, runStr);
        fileFullPath = fullfile(subjFolder, filename);

        if ~isfile(fileFullPath)
            EEG = prepare_eeg_struct(subjNum, runNum, basePathEEG, false);
            EEG.filename = filename;
            EEG.filepath = subjFolder;
            EEG.stage = 'raw';
            EEG = pop_saveset(EEG, 'filename', EEG.filename, 'filepath', EEG.filepath, 'savemode', 'onefile');
        end

        preprocFilename = fullfile(subjFolder, sprintf('%s_%s_ICA_rej_preproc%s.set', subjStr, runStr, preproc_type));

        if isfile(preprocFilename) && ~overwrite_preproc
            fprintf('Preprocessed file already exists: %s\nSkipping preprocessing.\n', preprocFilename);
            continue;
        end

        fprintf('Running preprocessing for %s (overwrite=%d)\n', preprocFilename, overwrite_preproc);

        EEG = pop_loadset('filename', filename, 'filepath', subjFolder);
        EEG.stage = 'preproc';
        EEG.preproc_type = preproc_type;

        switch preproc_type
            case 'Heilb'
                do_interp = true;
                [EEG, maskFromCleanRawFun] = preprocess_Heilbron_EEG(EEG, do_interp);
            case 'Custom'
                do_interp = true;
                [EEG, maskFromCleanRawFun] = preprocess_Custom_EEG(EEG, do_interp);
            case '2to20Hz'
                prefiltFilename = sprintf('%s_%s_ICA_rej_prefilt.set', subjStr, runStr);
                prefiltPath = fullfile(subjFolder, prefiltFilename);
                if ~isfile(prefiltPath)
                    preprocess_Heilbron_EEG(EEG, true);
                    warning('Prefilt EEG not found: %s. Ran Heilbron preprocessing.', prefiltPath);
                end
                if ~isfile(prefiltPath)
                    error('A1_preprocess_EEG:PrefiltMissing', 'Prefilt EEG still not found: %s', prefiltPath);
                end
                EEG = pop_loadset('filename', prefiltFilename, 'filepath', subjFolder);
                EEG.stage = 'preproc';
                EEG.preproc_type = preproc_type;
                EEG.filepath = subjFolder;
                EEG.filename = sprintf('%s_%s_ICA_rej.set', subjStr, runStr);
                EEG.setname = sprintf('%s_%s_ICA_rej', subjStr, runStr);
                maskFile = fullfile(subjFolder, sprintf('%s_%s_mask_From_CleanRaw.mat', subjStr, runStr));
                [EEG, maskFromCleanRawFun] = preprocess_2to20Hz(EEG, maskFile);
            case '0.01to20Hz'
                prefiltFilename = sprintf('%s_%s_ICA_rej_prefilt.set', subjStr, runStr);
                prefiltPath = fullfile(subjFolder, prefiltFilename);
                if ~isfile(prefiltPath)
                    preprocess_Heilbron_EEG(EEG, true);
                    warning('Prefilt EEG not found: %s. Ran Heilbron preprocessing.', prefiltPath);
                end
                if ~isfile(prefiltPath)
                    error('A1_preprocess_EEG:PrefiltMissing', 'Prefilt EEG still not found: %s', prefiltPath);
                end
                EEG = pop_loadset('filename', prefiltFilename, 'filepath', subjFolder);
                EEG.stage = 'preproc';
                EEG.preproc_type = preproc_type;
                EEG.filepath = subjFolder;
                EEG.filename = sprintf('%s_%s_ICA_rej.set', subjStr, runStr);
                EEG.setname = sprintf('%s_%s_ICA_rej', subjStr, runStr);
                maskFile = fullfile(subjFolder, sprintf('%s_%s_mask_From_CleanRaw.mat', subjStr, runStr));
                [EEG, maskFromCleanRawFun] = preprocess_0p01to20Hz(EEG, maskFile);
            otherwise
                error('A1_preprocess_EEG:UnknownPipeline', ...
                      'Unknown preprocessing type: %s', preproc_type);
        end

        maskFilename = sprintf('%s_%s_mask_From_CleanRaw.mat', subjStr, runStr);
        save_mask_parfor(fullfile(subjFolder, maskFilename), maskFromCleanRawFun);

        [~, nome, ~] = fileparts(EEG.filename);
        EEG.filename = sprintf('%s_preproc%s.set', nome, preproc_type);
        EEG = pop_saveset(EEG, 'filename', EEG.filename, 'filepath', EEG.filepath, 'savemode', 'onefile');
    end

    mergedFilename = sprintf('Subject%d_ICA_rej_preproc%s_merged.set', subjNum, preproc_type);
    mergedPath = fullfile(basePathEEG, subjStr, mergedFilename);

    if isfile(mergedPath)
        fprintf('Merged EEG file already exists: %s\nSkipping concatenation.\n', mergedPath);
    else
        fprintf('Merging EEG runs for Subject %02d...\n', subjNum);
        fileSuffix = sprintf('ICA_rej_preproc%s', preproc_type);
        merge_eeg_runs(subjNum, basePathEEG, numOfStim, fileSuffix);
    end
end

fprintf('EEG preprocessing completed for %d subject(s).\n', numOfSubj);
