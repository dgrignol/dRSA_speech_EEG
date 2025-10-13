% Reset MATLAB environment for a clean run.
clearvars;
close all;
clc;

% Load repository path configuration and add required helper folders.
paths = load_paths_config();
addpath(paths.eeglab)
addpath(paths.functions)

% Define core directories and dRSA configuration parameters.
basePathEEG = paths.dataEEG;
basePathMasks = paths.masks;
basePathOutput = paths.results;
basePathModels = paths.models;
preproc_type = '2to20Hz';
numOfSubj = 1;
re_run_dRSA = true;
lagWindowSec = [-3 3];
wav2vec2LayerIndices = [0 6 12 18 24]; % indices of wav2vec2 layers to include (e.g., [0 6 12 18 24])
wav2vec2DistanceMeasure = 'correlation'; % distance measure for wav2vec2 models

% Verify that essential directories exist (create outputs if needed).
if ~isfolder(basePathEEG)
    error('C1_dRSA_run:EEGPathMissing', 'EEG data folder not found: %s', basePathEEG);
end
if ~isfolder(basePathMasks)
    error('C1_dRSA_run:MasksPathMissing', 'Masks folder not found: %s', basePathMasks);
end
if ~isfolder(basePathOutput)
    mkdir(basePathOutput);
end
if ~isfolder(paths.subsamples)
    mkdir(paths.subsamples);
end

% Launch EEGLAB (headless) if the toolbox path is configured.
if ~isempty(paths.eeglab)
    eeglab nogui;
else
    warning('C1_dRSA_run:EEGLABPathMissing', 'EEGLAB path not set in config_paths.json.');
end

% hilbertModelPath = fullfile(basePathModels, 'Envelopes_Hilbert_128Hz', 'envelope_Hilbert.mat');
heilbModelPath = fullfile(basePathModels, 'Envelopes_Heilb_128Hz', 'envelope_Heilb.mat');

% Confirm required model artefacts are present before proceeding.
% if ~isfile(hilbertModelPath)
%     error('C1_dRSA_run:HilbertModelMissing', 'Hilbert envelope model not found: %s', hilbertModelPath);
% end
if ~isfile(heilbModelPath)
    error('C1_dRSA_run:HeilbronModelMissing', 'Heilbron envelope model not found: %s', heilbModelPath);
end
env_Heilbron = load(heilbModelPath);

[referenceInfo, wav2vec2Models] = prepare_wav2vec2_models( ...
    basePathModels, basePathEEG, preproc_type, numOfSubj, wav2vec2LayerIndices);
referenceLen = referenceInfo.length;
referenceFs = referenceInfo.fs;
preloadedEEG = referenceInfo.preloadedEEG;
firstSubjectIdx = referenceInfo.firstSubjectIdx;
wav2vec2Resampled = wav2vec2Models.data;
wav2vec2Labels = wav2vec2Models.labels;

% Load resampled raw audio stimulus and align with reference grid.
rawAudioPath = fullfile(paths.dataStimuli, 'Audio', 'audio_resampled_merged.wav');
if ~isfile(rawAudioPath)
    error('C1_dRSA_run:RawAudioMissing', 'Resampled audio model not found: %s', rawAudioPath);
end
[rawAudio, rawAudioFs] = audioread(rawAudioPath);
if size(rawAudio, 2) > 1
    rawAudio = mean(rawAudio, 2); % collapse stereo to mono if needed
end

if abs(rawAudioFs - referenceFs) > 1e-6
    error('C1_dRSA_run:RawAudioFsMismatch', ...
        'Raw audio sampling rate (%.6f Hz) differs from reference (%.6f Hz).', rawAudioFs, referenceFs);
end

rawAudio = rawAudio'; % ensure row orientation
if size(rawAudio, 2) ~= referenceLen
    error('C1_dRSA_run:RawAudioLengthMismatch', ...
        'Raw audio length (%d) differs from reference length (%d).', numel(rawAudio), referenceLen);
end

% Package model matrices for the dRSA pipeline.
models.data = [{rawAudio}, {env_Heilbron.env_model_data'}, wav2vec2Resampled{:}];
models.labels = [{'rawAudio'}, {'AudioEnvelopeHeilbron'}, wav2vec2Labels{:}];

% Iterate through subjects, executing the dRSA analysis as needed.
for subjNum = 1:numOfSubj
    fprintf('\nSubject %02d\n', subjNum);
    resultPattern = sprintf('*_dRSA_%s_subj%02d.mat', preproc_type, subjNum);
    existingResults = dir(fullfile(basePathOutput, resultPattern));

    % Skip subjects with existing results unless re-run is requested.
    if ~re_run_dRSA && ~isempty(existingResults)
        fprintf('Skipping subject %02d: existing results %s\n', subjNum, existingResults(1).name);
        continue;
    elseif ~isempty(existingResults) && re_run_dRSA
        fprintf('Re-running dRSA (overwrite enabled). Previous results: %s\n', existingResults(1).name);
    end

    timestampPrefix = char(datetime('now', 'Format', 'yyyy-MM-dd-HH-mm'));
    resultFileName = sprintf('%s_dRSA_%s_subj%02d.mat', timestampPrefix, preproc_type, subjNum);
    resultFile = fullfile(basePathOutput, resultFileName);

    eegMergedFile = sprintf('Subject%d_ICA_rej_preproc%s_merged.set', subjNum, preproc_type);
    eegMergedPath = fullfile(basePathEEG, sprintf('Subject%d', subjNum));
    % Confirm the subject's merged EEG dataset is available.
    if ~isfile(fullfile(eegMergedPath, eegMergedFile))
        warning('C1_dRSA_run:MergedEEGMissing', ...
                'Merged EEG dataset not found for Subject %02d (%s). Skipping.', ...
                subjNum, fullfile(eegMergedPath, eegMergedFile));
        continue;
    end

    % Reuse the reference EEG if available; otherwise load the subject data.
    if subjNum == firstSubjectIdx && ~isempty(preloadedEEG)
        EEG_merged = preloadedEEG;
    else
        EEG_merged = pop_loadset(eegMergedFile, eegMergedPath);
    end

    % Verify subject EEG matches the reference grid.
    neural_len = size(EEG_merged.data, 2);
    neural_fs = EEG_merged.srate;
    if neural_len ~= referenceLen
        error('C1_dRSA_run:NeuralLengthMismatch', ...
            'Subject %02d EEG length (%d) differs from reference length (%d).', ...
            subjNum, neural_len, referenceLen);
    end
    if abs(neural_fs - referenceFs) > 1e-6
        warning('C1_dRSA_run:SamplingRateMismatch', ...
            'Subject %02d EEG sampling rate (%.6f Hz) differs from reference (%.6f Hz).', ...
            subjNum, neural_fs, referenceFs);
    end

    % Ensure every model has the same temporal length as the EEG.
    for modelNum = 1:numel(models.data)
        model_len = size(models.data{modelNum}, 2);
        if size(EEG_merged.data, 2) ~= model_len
            error('C1_dRSA_run:LengthMismatch', ...
                  'EEG length (%d) does not match model %s length (%d).', ...
                  size(EEG_merged.data, 2), models.labels{modelNum}, model_len);
        end
    end

    % Load masks describing concatenation boundaries and bad EEG segments.
    mask_concat_file = fullfile(basePathMasks, 'mask_concat.mat');
    mask_bad_wins_file = fullfile(basePathMasks, 'bad_wins', sprintf('mask_bad_wins_Subject%02d.mat', subjNum));

    if isfile(mask_bad_wins_file)
        tmp = load(mask_bad_wins_file);
        maskBadWins = tmp.maskBadWins;
    else
        warning('C1_dRSA_run:MaskBadWinsMissing', ...
                'Bad windows mask not found for subject %02d. Using zeros.', subjNum);
        maskBadWins = zeros(1, size(EEG_merged.data, 2));
    end

    if ~isfile(mask_concat_file)
        error('C1_dRSA_run:MaskConcatMissing', 'Concatenation mask not found: %s', mask_concat_file);
    end
    maskConcat = load(mask_concat_file);

    mask = {maskConcat.maskConcat; maskBadWins};
    maskLabels = {'6 sec junct pnts'; sprintf('Bad EEG windows Subject %d', subjNum)};

    % Reshape EEG data to the tensor layout expected by dRSA.
    Y = reshape(EEG_merged.data, [1, 1, size(EEG_merged.data, 1), size(EEG_merged.data, 2)]);

    % Configure dRSA options and sampling parameters.
    opt.SubSampleDurSec = 5;
    opt.nSubSamples = 300;
    opt.nIter = 100;
    opt.dRSA.corrMethod = 'corr';
    opt.dRSA.Normalize = 'Rescale';
    opt.distanceMeasureModel = [{'euclidean'}, {'euclidean'}, repmat({wav2vec2DistanceMeasure}, 1, numel(wav2vec2Resampled))];
    opt.distanceMeasureNeural = 'correlation';
    opt.sampleDur = 1 / EEG_merged.srate;
    opt.SubSampleDur = round(opt.SubSampleDurSec / opt.sampleDur);
    opt.spaceStartSec = 0;
    opt.spacingSec = 0.1;
    opt.spaceStart = round(opt.spaceStartSec / opt.sampleDur);
    opt.spacing = round(opt.spacingSec / opt.sampleDur);
    opt.allModels = 1:numel(models.data);
    opt.autocorr = 1;
    opt.mask = mask;
    opt.maskLabels = maskLabels;
    opt.modelsLabels = models.labels;

    cfg.modelVec = 1;
    cfg.modelRegressout = [];
    cfg.condVec = 1;
    cfg.ROINumber = 3;
    cfg.saving = 0;
    cfg.SubSampleDir = paths.subsamples;

    [dRSA, nRSA, mRSA] = dRSA_coreFunction(Y, models, opt, cfg);
    general_info.neural_fs = neural_fs;
    general_info.preproc_type = preproc_type;
    save(resultFile, 'dRSA', 'nRSA', 'mRSA', 'opt','general_info');

    fprintf('Saved dRSA results: %s\n', resultFile);

    for model_num = 1:size(dRSA, 3)
        figure;
        title_plot = opt.modelsLabels{model_num};
        sgtitle(title_plot, 'Interpreter', 'none');

        subplot(3, 2, 1);
        imagesc(dRSA(:, :, model_num));
        axis square; colorbar;
        title('dRSA', 'Interpreter', 'none');

        subplot(3, 2, 3);
        imagesc(mRSA(:, :, model_num));
        axis square; colorbar;
        title('mRSA', 'Interpreter', 'none');

        subplot(3, 2, 5);
        imagesc(nRSA);
        axis square; colorbar;
        title('nRSA', 'Interpreter', 'none');

        [dRSA_diag_avg, dRSA_diag_std] = all_diagonal_averages(dRSA(:, :, model_num));
        [mRSA_diag_avg, mRSA_diag_std] = all_diagonal_averages(mRSA(:, :, model_num));
        [nRSA_diag_avg, nRSA_diag_std] = all_diagonal_averages(nRSA);

        Fs = general_info.neural_fs; %EEG_merged.srate;
        plot_dRSA_subj_avg(dRSA_diag_avg, dRSA_diag_std, ...
                           mRSA_diag_avg, mRSA_diag_std, ...
                           nRSA_diag_avg, nRSA_diag_std, ...
                           Fs, title_plot);
    end
end

dRSA_files = dir(fullfile(basePathOutput, sprintf('*_dRSA_%s_subj*.mat', preproc_type)));
if isempty(dRSA_files)
    warning('C1_dRSA_run:NoResults', 'No dRSA results found. Skipping group averaging.');
    return;
end

dRSA_group_all = [];
optGroup = [];

for fileIdx = 1:numel(dRSA_files)
    tmp = load(fullfile(dRSA_files(fileIdx).folder, dRSA_files(fileIdx).name), ...
               'dRSA', 'mRSA', 'nRSA', 'opt');
    token = regexp(dRSA_files(fileIdx).name, ['_dRSA_' preproc_type '_subj(\d{2})\.mat'], 'tokens', 'once');
    if ~isempty(token)
        subjNum = str2double(token{1});
    else
        subjNum = [];
    end
    if isempty(subjNum)
        subjNum = fileIdx;
    end

    if isempty(dRSA_group_all)
        nModels = size(tmp.dRSA, 3);
        matrixSize = size(tmp.dRSA, 1);
        diagLen = 2 * matrixSize - 1;
        dRSA_group_all = nan(matrixSize, matrixSize, nModels, numOfSubj);
        mRSA_group_all = nan(matrixSize, matrixSize, nModels, numOfSubj);
        nRSA_group_all = nan(matrixSize, matrixSize, numOfSubj);
        dRSA_diag_group_all = nan(numOfSubj, diagLen, nModels);
        mRSA_diag_group_all = nan(numOfSubj, diagLen, nModels);
        nRSA_diag_group_all = nan(numOfSubj, diagLen);
        optGroup = tmp.opt;
    end

    dRSA_group_all(:, :, :, subjNum) = tmp.dRSA;
    mRSA_group_all(:, :, :, subjNum) = tmp.mRSA;
    nRSA_group_all(:, :, subjNum) = tmp.nRSA;

    for model_num = 1:nModels
        dDiagAvg = all_diagonal_averages(tmp.dRSA(:, :, model_num));
        mDiagAvg = all_diagonal_averages(tmp.mRSA(:, :, model_num));
        dRSA_diag_group_all(subjNum, :, model_num) = dDiagAvg;
        mRSA_diag_group_all(subjNum, :, model_num) = mDiagAvg;
    end
    nDiagAvg = all_diagonal_averages(tmp.nRSA);
    nRSA_diag_group_all(subjNum, :) = nDiagAvg;
end

if isempty(dRSA_group_all)
    warning('C1_dRSA_run:NoGroupData', 'Group data not assembled. Skipping plots.');
    return;
end

if isfield(optGroup, 'sampleDur') && optGroup.sampleDur > 0
    Fs_group = round(1 / optGroup.sampleDur);
else
    Fs_group = 128;
end

validSubsMask = any(~isnan(nRSA_diag_group_all), 2);
numValidSubs = sum(validSubsMask);

for model_num = 1:nModels
    if numValidSubs == 0
        warning('C1_dRSA_run:NoValidSubs', ...
                'No valid subjects available for model %d. Skipping.', model_num);
        continue;
    end

    dRSA_mean_matrix = squeeze(mean(dRSA_group_all(:, :, model_num, validSubsMask), 4, 'omitnan'));
    mRSA_mean_matrix = squeeze(mean(mRSA_group_all(:, :, model_num, validSubsMask), 4, 'omitnan'));
    nRSA_mean_matrix = squeeze(mean(nRSA_group_all(:, :, validSubsMask), 3, 'omitnan'));
    dRSA_diag_avg = mean(dRSA_diag_group_all(validSubsMask, :, model_num), 1, 'omitnan');
    dRSA_diag_std = std(dRSA_diag_group_all(validSubsMask, :, model_num), 0, 1, 'omitnan');
    mRSA_diag_avg = mean(mRSA_diag_group_all(validSubsMask, :, model_num), 1, 'omitnan');
    mRSA_diag_std = std(mRSA_diag_group_all(validSubsMask, :, model_num), 0, 1, 'omitnan');
    nRSA_diag_avg = mean(nRSA_diag_group_all(validSubsMask, :), 1, 'omitnan');
    nRSA_diag_std = std(nRSA_diag_group_all(validSubsMask, :), 0, 1, 'omitnan');

    title_plot = sprintf('Preproc: %s - Model: %s - Group Average (n=%d)', ...
                         preproc_type, optGroup.modelsLabels{model_num}, numValidSubs);

    figure;
    sgtitle(title_plot, 'Interpreter', 'none');
    subplot(3, 2, 1);
    imagesc(dRSA_mean_matrix);
    axis square; colorbar;
    title('dRSA (group mean)', 'Interpreter', 'none');
    subplot(3, 2, 3);
    imagesc(mRSA_mean_matrix);
    axis square; colorbar;
    title('mRSA (group mean)', 'Interpreter', 'none');
    subplot(3, 2, 5);
    imagesc(nRSA_mean_matrix);
    axis square; colorbar;
    title('nRSA (group mean)', 'Interpreter', 'none');

    plot_dRSA_subj_avg(dRSA_diag_avg, dRSA_diag_std, ...
                       mRSA_diag_avg, mRSA_diag_std, ...
                       nRSA_diag_avg, nRSA_diag_std, ...
                       Fs_group, title_plot, lagWindowSec);
end
