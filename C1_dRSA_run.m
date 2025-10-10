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
preproc_type = '0.01to20Hz';
numOfSubj = 19;
re_run_dRSA = false;
lagWindowSec = [-3 3];
wav2vec2LayerIndices = [24]; % indices of wav2vec2 layers to include (e.g., [0 6 12 18 24])
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

hilbertModelPath = fullfile(basePathModels, 'Envelopes_Hilbert_128Hz', 'envelope_Hilbert.mat');
heilbModelPath = fullfile(basePathModels, 'Envelopes_Heilb_128Hz', 'envelope_Heilb.mat');
wav2vec2MetadataPath = fullfile(basePathModels, 'wav2vec2', 'wav2vec2_embeddings.json');

% Confirm required model artefacts are present before proceeding.
if ~isfile(hilbertModelPath)
    error('C1_dRSA_run:HilbertModelMissing', 'Hilbert envelope model not found: %s', hilbertModelPath);
end
if ~isfile(heilbModelPath)
    error('C1_dRSA_run:HeilbronModelMissing', 'Heilbron envelope model not found: %s', heilbModelPath);
end
if ~isfile(wav2vec2MetadataPath)
    error('C1_dRSA_run:wav2vec2MetadataMissing', 'wav2vec2 metadata not found: %s', wav2vec2MetadataPath);
end
wav2vec2Metadata = jsondecode(fileread(wav2vec2MetadataPath));
if ~isfield(wav2vec2Metadata, 'layers') || isempty(wav2vec2Metadata.layers)
    error('C1_dRSA_run:wav2vec2LayersMissing', ...
        'wav2vec2 metadata does not describe layer MAT files: %s', wav2vec2MetadataPath);
end
layerIndicesAvailable = arrayfun(@(layer) layer.index, wav2vec2Metadata.layers);
[isLayerPresent, layerPositions] = ismember(wav2vec2LayerIndices, layerIndicesAvailable);
if ~all(isLayerPresent)
    missingLayers = wav2vec2LayerIndices(~isLayerPresent);
    error('C1_dRSA_run:wav2vec2LayerNotFound', ...
        'Requested wav2vec2 layer indices not available: %s', num2str(missingLayers));
end
selectedLayerEntries = wav2vec2Metadata.layers(layerPositions);
env_Hilbert = load(hilbertModelPath);
env_Heilbron = load(heilbModelPath);

% Identify a reference subject to establish the shared EEG grid to resample the wav2vec2 model.
referenceLen = [];
referenceFs = [];
preloadedEEG = [];
firstSubjectIdx = NaN;
for subjProbe = 1:numOfSubj
    eegCandidateFile = sprintf('Subject%d_ICA_rej_preproc%s_merged.set', subjProbe, preproc_type);
    eegCandidatePath = fullfile(basePathEEG, sprintf('Subject%d', subjProbe));
    candidateFullPath = fullfile(eegCandidatePath, eegCandidateFile);
    if isfile(candidateFullPath)
        preloadedEEG = pop_loadset(eegCandidateFile, eegCandidatePath);
        referenceLen = size(preloadedEEG.data, 2);
        referenceFs = preloadedEEG.srate;
        firstSubjectIdx = subjProbe;
        fprintf('Using Subject %02d to define wav2vec2 resampling grid (%d samples @ %.2f Hz).\n', ...
            subjProbe, referenceLen, referenceFs);
        break;
    end
end

% Abort if no EEG dataset could be located.
if isempty(referenceLen)
    error('C1_dRSA_run:NoEEGFound', ...
        'Unable to locate any merged EEG dataset to define wav2vec2 resampling.');
end

% Upsample the selected wav2vec2 embeddings once using the reference EEG grid.
wav2vec2FieldsRequired = {'embeddings', 'time_axis'};
wav2vec2Resampled = cell(1, numel(selectedLayerEntries));
wav2vec2Labels = cell(1, numel(selectedLayerEntries));
for layerIdx = 1:numel(selectedLayerEntries)
    layerMatPath = fullfile(basePathModels, 'wav2vec2', selectedLayerEntries(layerIdx).mat_file);
    if ~isfile(layerMatPath)
        error('C1_dRSA_run:wav2vec2LayerMissing', ...
            'wav2vec2 layer file not found: %s', layerMatPath);
    end
    wav2vec2LayerData = load(layerMatPath);
    if ~all(isfield(wav2vec2LayerData, wav2vec2FieldsRequired))
        error('C1_dRSA_run:wav2vec2FieldsMissing', ...
            'wav2vec2 layer file %s missing fields: %s', layerMatPath, strjoin(wav2vec2FieldsRequired, ', '));
    end
    wav2vec2Resampled{layerIdx} = upsample_wav2vec2_embeddings(wav2vec2LayerData, referenceLen, referenceFs)';
    layerLabel = selectedLayerEntries(layerIdx).label;
    if isstring(layerLabel) || ischar(layerLabel)
        layerLabel = char(layerLabel);
    else
        layerLabel = sprintf('layer%d', selectedLayerEntries(layerIdx).index);
    end
    wav2vec2Labels{layerIdx} = sprintf('wav2vec2.%s', layerLabel);
end

% Package model matrices for the dRSA pipeline.
models.data = [{env_Hilbert.env_model_data'}, {env_Heilbron.env_model_data'}, wav2vec2Resampled{:}];
models.labels = [{'AudioEnvelopeHilbert'}, {'AudioEnvelopeHeilbron'}, wav2vec2Labels{:}];

% Iterate through subjects, executing the dRSA analysis as needed.
for subjNum = 1:numOfSubj
    fprintf('\nSubject %02d\n', subjNum);
    resultFile = fullfile(basePathOutput, sprintf('dRSA_%s_subj%02d.mat', preproc_type, subjNum));

    % Skip subjects with existing results unless re-run is requested.
    if ~re_run_dRSA && isfile(resultFile)
        fprintf('Skipping subject %02d: existing results %s\n', subjNum, resultFile);
        continue;
    elseif isfile(resultFile) && re_run_dRSA
        fprintf('Re-running dRSA (overwrite enabled).\n');
    end

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
    opt.distanceMeasureModel = [{'euclidean', 'euclidean'}, repmat({wav2vec2DistanceMeasure}, 1, numel(wav2vec2Resampled))];
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
    save(resultFile, 'dRSA', 'nRSA', 'mRSA', 'opt');

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

        Fs = EEG_merged.srate;
        plot_dRSA_subj_avg(dRSA_diag_avg, dRSA_diag_std, ...
                           mRSA_diag_avg, mRSA_diag_std, ...
                           nRSA_diag_avg, nRSA_diag_std, ...
                           Fs, title_plot);
    end
end

dRSA_files = dir(fullfile(basePathOutput, sprintf('dRSA_%s_subj*.mat', preproc_type)));
if isempty(dRSA_files)
    warning('C1_dRSA_run:NoResults', 'No dRSA results found. Skipping group averaging.');
    return;
end

dRSA_group_all = [];
optGroup = [];

for fileIdx = 1:numel(dRSA_files)
    tmp = load(fullfile(dRSA_files(fileIdx).folder, dRSA_files(fileIdx).name), ...
               'dRSA', 'mRSA', 'nRSA', 'opt');
    subjNum = sscanf(dRSA_files(fileIdx).name, ['dRSA_' preproc_type '_subj%02d.mat']);
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

function resampled = upsample_wav2vec2_embeddings(wav2vec2Data, targetLen, targetFs)
%UPSAMPLE_WAV2VEC2_EMBEDDINGS Interpolate wav2vec2 embeddings onto neural timeline.

    required = {'embeddings', 'time_axis'};
    if ~all(isfield(wav2vec2Data, required))
        error('upsample_wav2vec2_embeddings:MissingFields', ...
            'Expected fields missing from wav2vec2 data: %s', strjoin(required, ', '));
    end

    originalTimes = wav2vec2Data.time_axis(:);
    embeddings = wav2vec2Data.embeddings;

    if numel(originalTimes) ~= size(embeddings, 1)
        error('upsample_wav2vec2_embeddings:TimeMismatch', ...
            'Length of time axis (%d) does not match embedding rows (%d).', ...
            numel(originalTimes), size(embeddings, 1));
    end

    startTime = originalTimes(1);
    targetTimes = startTime + (0:targetLen - 1)' ./ targetFs;

    embedClass = class(embeddings);
    resampled = interp1(originalTimes, double(embeddings), targetTimes, 'linear', 'extrap');
    if ~strcmp(embedClass, 'double')
        resampled = cast(resampled, embedClass);
    end
end
