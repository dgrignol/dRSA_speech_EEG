close all; clear all;

eeglabDir = '/Users/damiano/Documents/MATLAB/eeglab2025.0.0'; % e.g., '/path/to/toolboxes/eeglab2025.0.0'
repoBase = '/Users/damiano/Documents/UniTn/Dynamo_ch/dRSA_speech_EEG';      % e.g., '/path/to/dRSA_speech_V02'

if isempty(repoBase)
    repoBase = fileparts(mfilename('fullpath'));
end

if ~exist(repoBase, 'dir')
    error('main_V02:RepoNotFound', 'Repository base directory not found: %s', repoBase);
end

addpath(repoBase);
addpath(fullfile(repoBase, 'functions'));
addpath(fullfile(repoBase, 'functions', 'dRSA'));
addpath(fullfile(repoBase, 'functions', 'modelCreation'));
addpath(fullfile(repoBase, 'functions', 'investigsate_Heilbron_models'));

if ~isempty(eeglabDir)
    if exist(eeglabDir, 'dir')
        addpath(eeglabDir);
    else
        warning('main_V02:EEGLABNotFound', 'EEGLAB directory not found: %s', eeglabDir);
    end
end

paths.repoBase = repoBase;
paths.functions = fullfile(repoBase, 'functions');
paths.dRSA = fullfile(paths.functions, 'dRSA');
paths.modelCreation = fullfile(paths.functions, 'modelCreation');
paths.investigateHeilbron = fullfile(paths.functions, 'investigsate_Heilbron_models');
paths.data = fullfile(repoBase, 'data');
paths.dataEEG = fullfile(paths.data, 'EEG');
paths.dataStimuli = fullfile(paths.data, 'Stimuli');
paths.masks = fullfile(repoBase, 'Masks');
paths.results = fullfile(repoBase, 'results');
paths.models = fullfile(repoBase, 'Models');
paths.subsamples = fullfile(repoBase, 'subsamples');
paths.figures = fullfile(repoBase, 'figures');
paths.eeglab = eeglabDir;

% choose param
preproc_type = '0.01to20Hz'; %'2to20Hz' -'Heilb' (Heilbron et al., 2022)

basePathEEG = paths.dataEEG;
basePathStimuli = paths.dataStimuli;
basePathMasks = paths.masks;
basePathOutput = paths.results;
basePathModels = paths.models;


sartEnd_span = 3; % number of seconds to mask before and after a concat event
eeglab nogui;

numOfStim = 20;
numOfSubj = 19;
for subjNum = 1:numOfSubj
    for runNum = 1:numOfStim
        subjStr = sprintf('Subject%d', subjNum);
        runStr = sprintf('Run%d', runNum);
        filename = sprintf('%s_%s.set', subjStr, runStr);
        subjFolder = fullfile(basePathEEG, subjStr);
        fileFullPath = fullfile(subjFolder, filename);
       

        % Check if EEG set file already exists
        if ~exist(fileFullPath, 'file') 
            EEG = prepare_eeg_struct(subjNum, runNum, basePathEEG, false);  % set to true if mastoids needed

            EEG.filename = filename;
            EEG.filepath = subjFolder;
            EEG.stage = 'raw'; % set the stage for saving
            EEG = pop_saveset(EEG, 'filename', EEG.filename, 'filepath', EEG.filepath, 'savemode', 'onefile');
        end

        %% Preprocess
        % Define expected output filename
        preprocFilename = fullfile(subjFolder, sprintf('%s_%s_ICA_rej_preproc%s.set', subjStr, runStr, preproc_type));
 
        % Overwrite flag (set to true to force reprocessing)
        overwrite_preproc = false;

        % Check if it already exists
        if exist(preprocFilename, 'file') && ~overwrite_preproc
            fprintf('Preprocessed file already exists: %s\nSkipping preprocessing.\n', preprocFilename);
        else
            fprintf('Running preprocessing: %s (overwrite_preproc=%d)\n', preprocFilename, overwrite_preproc);
            
            EEG = pop_loadset('filename', filename, 'filepath', subjFolder);
            
            EEG.stage = 'preproc'; % set the stage for saving
            EEG.preproc_type = preproc_type; % give the preproc pipeline a name

            % Preprocess EEG and generate mask
            if strcmp(preproc_type, 'Heilb')
                do_interp = true; % interpolate bad channels
                [EEG, maskFromCleanRawFun] = preprocess_Heilbron_EEG(EEG, do_interp);
            elseif strcmp(preproc_type, 'Custom')
                do_interp = true; % interpolate bad channels
                [EEG, maskFromCleanRawFun] = preprocess_Custom_EEG(EEG, do_interp);
            elseif strcmp(preproc_type, '2to20Hz')
                % Load prefilt dataset saved by the Heilbron pipeline and apply a 2-20 Hz filter
                prefiltFilename = sprintf('%s_%s_ICA_rej_prefilt.set', subjStr, runStr);
                prefiltPath = fullfile(subjFolder, prefiltFilename);
                if ~exist(prefiltPath, 'file')
                    preprocess_Heilbron_EEG(EEG, true);
                    warning('Prefilt EEG not found: %s. Running the Heilbron preprocessing first.', prefiltPath);
                end
                if ~exist(prefiltPath, 'file')
                    error('STILL NO Prefilt EEG. %s. Run the Heilbron preprocessing first.', prefiltPath);
                end
                EEG = pop_loadset('filename', prefiltFilename, 'filepath', subjFolder);
                EEG.stage = 'preproc';
                EEG.preproc_type = preproc_type;
                EEG.filepath = subjFolder;
                EEG.filename = sprintf('%s_%s_ICA_rej.set', subjStr, runStr);
                EEG.setname = sprintf('%s_%s_ICA_rej', subjStr, runStr);
                maskFile = fullfile(subjFolder, sprintf('%s_%s_mask_From_CleanRaw.mat', subjStr, runStr));
                [EEG, maskFromCleanRawFun] = preprocess_2to20Hz(EEG, maskFile);
            elseif strcmp(preproc_type, '0.01to20Hz')
                % Load prefilt dataset saved by the Heilbron pipeline and apply a 0.01-20 Hz filter
                prefiltFilename = sprintf('%s_%s_ICA_rej_prefilt.set', subjStr, runStr);
                prefiltPath = fullfile(subjFolder, prefiltFilename);
                if ~exist(prefiltPath, 'file')
                    preprocess_Heilbron_EEG(EEG, true);
                    warning('Prefilt EEG not found: %s. Running the Heilbron preprocessing first.', prefiltPath);
                end
                if ~exist(prefiltPath, 'file')
                    error('STILL NO Prefilt EEG. %s. Run the Heilbron preprocessing first.', prefiltPath);
                end
                EEG = pop_loadset('filename', prefiltFilename, 'filepath', subjFolder);
                EEG.stage = 'preproc';
                EEG.preproc_type = preproc_type;
                EEG.filepath = subjFolder;
                EEG.filename = sprintf('%s_%s_ICA_rej.set', subjStr, runStr);
                EEG.setname = sprintf('%s_%s_ICA_rej', subjStr, runStr);
                maskFile = fullfile(subjFolder, sprintf('%s_%s_mask_From_CleanRaw.mat', subjStr, runStr));
                [EEG, maskFromCleanRawFun] = preprocess_0p01to20Hz(EEG, maskFile);
            end
            % Save the mask
            maskFilename = sprintf('%s_%s_mask_From_CleanRaw.mat', subjStr, runStr);
            save_mask_parfor(fullfile(subjFolder, maskFilename), maskFromCleanRawFun); % not using just save because forbidden in parfor

            % Save the preprocessed EEG
            [~, nome, ~] = fileparts(EEG.filename);
            EEG.filename = [nome sprintf('_preproc%s.set', preproc_type)]; % 
            EEG = pop_saveset(EEG, 'filename', EEG.filename, 'filepath', EEG.filepath, 'savemode', 'onefile');
        end
        
    end
    
%% Concatenate all EEG runs (1 concatenated x subj) and all stimuli (same per all subj)

    % Expected merged filename (e.g., 'Subject01_ICA_rej_preprocHeilb.set')
    mergedFilename = sprintf('Subject%s_ICA_rej_preproc%s_merged.set', num2str(subjNum), preproc_type);
    mergedPath = fullfile(basePathEEG, sprintf('Subject%s', num2str(subjNum)), mergedFilename);

    if exist(mergedPath, 'file')
        fprintf('Merged EEG file already exists: %s\nSkipping concatenation.\n', mergedPath);
        EEG_merged = pop_loadset('filename', mergedFilename, 'filepath', fileparts(mergedPath));
    else
        fprintf('Merging EEG runs for Subject %02d...\n', subjNum);
        fileSuffix = sprintf('ICA_rej_preproc%s', preproc_type);
        EEG_merged = merge_eeg_runs(subjNum, basePathEEG, numOfStim, fileSuffix);
    end

end

%% Stimuli
if ~exist(fullfile(basePathStimuli, 'Audio', 'audio_merged.wav'),'file')
    % Audio
    %resample audio to target fs
    targetFs = 128;
    resample_audio_files(basePathStimuli, numOfStim, targetFs)

    EEG_merged = pop_loadset(sprintf('Subject%d_ICA_rej_preprocHeilb_merged.set',1), ...
                              fullfile(basePathEEG, sprintf('Subject%d',1)));

    % merge audio and create mask of concatenation points
     stimuliDir = fullfile(basePathStimuli, 'Audio');
     test = false;
    [mergedAudio, fs_check, maskConcat] = merge_audio_files(stimuliDir, basePathMasks, test, sartEnd_span);
    if length(mergedAudio) ~= length(EEG_merged.data)
        error('Length of merged audio does not fit EEG merged dataset')
    end
end
%% Models
% make alternative envelope models
name_suffix = 'Hilbert';
name_env_files = ['audio%d_' sprintf('128Hz_%s.mat', name_suffix)]; %input single run files
name_env_model_file = sprintf('envelope_%s.mat', name_suffix); % output merged file

if ~strcmp(name_suffix,'Heilb') % if the method is Heilbron we already have them
    sprintf('Sample freq choosen for model definition: %d',targetFs) % 128
    numOfStim = 20;
    method = name_suffix;
    for runNum = 1:numOfStim
        EnvFilename = sprintf('audio%02d_resampled.wav', runNum);
        filepath = fullfile(basePathStimuli, 'Audio', EnvFilename);
        [audio_signal, Fs] = audioread(filepath); % load file
        [env, Fs] = compute_envelope(audio_signal, Fs, method); %
        env = env'; % Nx1 not 1xN
        % Save with name of run
        save(fullfile(basePathStimuli, 'Envelopes', sprintf(name_env_files, runNum)), 'env');
    end
end

% concatenate the single run envelope models in the Stimuli folder
env_model_data = concatenate_envelopes(basePathStimuli, name_env_files, numOfStim);
save(fullfile(basePathModels, 'Envelopes', name_env_model_file), 'env_model_data')
sprintf('Envelope saved as %s', name_env_model_file)

        

% %% manual mask inspecting neural signal
% for subjNum = 1:numOfSubj
%     get_manual_mask_from_eegplot(EEG_merged,basePathMasks,subjNum);
% end

%% dRSA
re_run_dRSA = true;
% load EEG
% load model


% Prepare models
env_Hilbert = load(fullfile(basePathModels,'Envelopes_Hilbert_128Hz','envelope_Hilbert.mat'));
env_Heilbron = load(fullfile(basePathModels,'Envelopes_Heilb_128Hz','envelope_Heilb.mat'));

models.data = {env_Hilbert.env_model_data', env_Heilbron.env_model_data'};
models.labels = {'AudioEnvelopeHilbert', 'AudioEnvelopeHeilbron'};



 for subjNum = 1:numOfSubj
    % Define result file path
    resultFile = fullfile(basePathOutput, sprintf('dRSA_%s_subj%02d.mat', preproc_type, subjNum));

    % Check if dRSA already exists and skip unless re_run_dRSA == true
    if exist('re_run_dRSA', 'var') && re_run_dRSA
        fprintf('\nRe-running dRSA for subject %02d (forced by re_run_dRSA flag)...\n', subjNum);
    elseif exist(resultFile, 'file')
        fprintf('\nSkipping subject %02d: dRSA results already exist at %s\n', subjNum, resultFile);
        continue; % skip this subject
    end
    
  % Load data
    EEG_merged = pop_loadset(sprintf('Subject%d_ICA_rej_preproc%s_merged.set',subjNum,preproc_type), ...
                              fullfile(basePathEEG, sprintf('Subject%d',subjNum)));
  % Check signal lengths
     for modelNum = 1:size(models.data)
        model_len = size(models.data{modelNum, 1},2);
        if ~size(EEG_merged.data, 2) == model_len
            error('EEG data is not same length of the saved model.');
        end
     end
    
    %% Load masks

    mask_concat_file = fullfile(basePathMasks,'mask_concat.mat'); % mask start and finish of audio stimuli
    mask_bad_wins_file = fullfile(basePathMasks,'bad_wins',sprintf('mask_bad_wins_Subject%02d.mat',subjNum)); % mask bad windows in EEG data
    if exist(mask_bad_wins_file, 'file')
        tmp = load(mask_bad_wins_file);
        maskBadWins = tmp.maskBadWins;
    else
        warning('File not found: %s. Creating empty mask.', mask_bad_wins_file);
        maskBadWins = zeros(1, size(EEG_merged.data,2));
    end
    maskConcat = load(mask_concat_file);
    mask = {maskConcat.maskConcat; maskBadWins};
    maskLabels = {'6 sec junct pnts';sprintf('Bad EEG windows Subject %d',subjNum)};
%     load(fullfile(paths.repoBase, 'ds004408', 'masks', 'mask_final_sub-001.mat'))
    
    %% Format EEG for dRSA
    Y = reshape(EEG_merged.data, [1, 1, size(EEG_merged.data,1), size(EEG_merged.data,2)]);

    %% Define options
    opt.SubSampleDurSec = 5;
    opt.nSubSamples = 300;
    opt.nIter = 100;
    opt.dRSA.corrMethod = 'corr';
    opt.dRSA.Normalize = 'Rescale';
    opt.distanceMeasureModel = {'euclidean' 'euclidean'};
    opt.distanceMeasureNeural = 'correlation';
    opt.sampleDur = 1/EEG_merged.srate;
    opt.SubSampleDur = round(opt.SubSampleDurSec / opt.sampleDur);
    opt.spaceStartSec = 0;
    opt.spacingSec = 0.1;
    opt.spaceStart = round(opt.spaceStartSec / opt.sampleDur);
    opt.spacing = round(opt.spacingSec / opt.sampleDur);
    opt.allModels = 1:2;
    %opt.models2regressout = {[3 4]  [1 3 4]  [1 4]  [1 3]  [3 4]  [3 4]};%
    %for every model you say which models to regress out
    opt.autocorr = 1; % compute autocorrelation
    opt.mask = mask;
    opt.maskLabels = maskLabels;
    opt.modelsLabels = models.labels;
    cfg.modelVec = [1];
    cfg.modelRegressout = [];
    cfg.condVec = 1;
    cfg.ROINumber = 3;
    cfg.saving = 0;
    cfg.SubSampleDir = paths.subsamples;

    %% Run dRSA
    [dRSA, nRSA, mRSA] = dRSA_coreFunction(Y, models, opt, cfg);

    %% Save results
    save(resultFile, 'dRSA','nRSA','mRSA','opt');

    fprintf('\nAll dRSA computations complete for subject %02d. Results saved to: %s\n', subjNum, basePathOutput);

    
    
    %% build dRSA diagonal average
    if  ~exist('dRSA','var')
        load(fullfile(basePathOutput, sprintf('dRSA_%s_%02d.mat',preproc_type, subjNum)));
    end
    
    for model_num = 1 : size(dRSA, 3)
        figure;
        title_plot = opt.modelsLabels{model_num};
        % -------- overall title from opt.modelsLabels -------------
        sgtitle(title_plot, 'Interpreter', 'none');  % keep literal underscores

        % ----------- subplot 1 : dRSA matrix -----------------------
        subplot(3,2,1)
        imagesc(dRSA(:,:,model_num))
        axis square; colorbar
        title('dRSA', 'Interpreter', 'none');

        % ----------- subplot 2 : mRSA matrix -----------------------
        subplot(3,2,3)
        imagesc(mRSA(:,:,model_num))
        axis square; colorbar
        title('mRSA', 'Interpreter', 'none');

        % ----------- subplot 3 : nRSA matrix -----------------------
        subplot(3,2,5)
        imagesc(nRSA)
        axis square; colorbar
        title('nRSA', 'Interpreter', 'none');

        % ---------- diagonal-average time-courses ------------------
        [dRSA_diag_avg, dRSA_diag_std] = all_diagonal_averages(dRSA(:,:,model_num));
        [mRSA_diag_avg, mRSA_diag_std] = all_diagonal_averages(mRSA(:,:,model_num));
        [nRSA_diag_avg, nRSA_diag_std] = all_diagonal_averages(nRSA);

        Fs = 128;                                % sampling rate (Hz)
        plot_dRSA_subj_avg(dRSA_diag_avg, dRSA_diag_std, ...
                           mRSA_diag_avg, mRSA_diag_std, ...
                           nRSA_diag_avg, nRSA_diag_std, ...
                           Fs,title_plot);
    end
 end
 
 %% build dRSA diagonal average across subjects
 lagWindowSec = [-3 3]; % e.g., set to [-3 3] to limit plots to ±3 seconds

for subjNum = 1:numOfSubj
    resultFile = fullfile(basePathOutput, sprintf('dRSA_%s_subj%02d.mat', preproc_type, subjNum));
    if ~exist(resultFile, 'file')
        warning('Group dRSA: file not found for subject %02d (%s). Skipping.', subjNum, resultFile);
        continue;
    end
    tmp = load(resultFile, 'dRSA', 'mRSA', 'nRSA', 'opt');
    if ~exist('dRSA_group_all', 'var')
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
    dRSA_group_all(:,:,:,subjNum) = tmp.dRSA;
    mRSA_group_all(:,:,:,subjNum) = tmp.mRSA;
    nRSA_group_all(:,:,subjNum) = tmp.nRSA;
    for model_num = 1:nModels
        [dDiagAvg, ~] = all_diagonal_averages(tmp.dRSA(:,:,model_num));
        [mDiagAvg, ~] = all_diagonal_averages(tmp.mRSA(:,:,model_num));
        dRSA_diag_group_all(subjNum,:,model_num) = dDiagAvg;
        mRSA_diag_group_all(subjNum,:,model_num) = mDiagAvg;
    end
    [nDiagAvg, ~] = all_diagonal_averages(tmp.nRSA);
    nRSA_diag_group_all(subjNum,:) = nDiagAvg;
end
if exist('dRSA_group_all', 'var')
    if isfield(optGroup, 'sampleDur') && optGroup.sampleDur > 0
        Fs_group = round(1 ./ optGroup.sampleDur);
    else
        Fs_group = 128;
    end
    validSubsMask = any(~isnan(nRSA_diag_group_all), 2);
    numValidSubs = sum(validSubsMask);
    for model_num = 1:nModels
        if numValidSubs == 0
            warning('Group dRSA: no valid subjects available for plotting model %d.', model_num);
            continue;
        end
        dRSA_mean_matrix = squeeze(mean(dRSA_group_all(:,:,model_num,:), 4, 'omitnan'));
        mRSA_mean_matrix = squeeze(mean(mRSA_group_all(:,:,model_num,:), 4, 'omitnan'));
        nRSA_mean_matrix = squeeze(mean(nRSA_group_all, 3, 'omitnan'));
        dRSA_diag_avg = mean(dRSA_diag_group_all(:,:,model_num), 1, 'omitnan');
        dRSA_diag_std = std(dRSA_diag_group_all(:,:,model_num), 0, 1, 'omitnan');
        mRSA_diag_avg = mean(mRSA_diag_group_all(:,:,model_num), 1, 'omitnan');
        mRSA_diag_std = std(mRSA_diag_group_all(:,:,model_num), 0, 1, 'omitnan');
        nRSA_diag_avg = mean(nRSA_diag_group_all, 1, 'omitnan');
        nRSA_diag_std = std(nRSA_diag_group_all, 0, 1, 'omitnan');
        title_plot = sprintf('Preproc type: %s - Model: %s - Group Average (n=%d)', preproc_type, optGroup.modelsLabels{model_num}, numValidSubs);
        figure;
        sgtitle(title_plot, 'Interpreter', 'none');
        subplot(3,2,1)
        imagesc(dRSA_mean_matrix)
        axis square; colorbar
        title('dRSA (group mean)', 'Interpreter', 'none');
        subplot(3,2,3)
        imagesc(mRSA_mean_matrix)
        axis square; colorbar
        title('mRSA (group mean)', 'Interpreter', 'none');
        subplot(3,2,5)
        imagesc(nRSA_mean_matrix)
        axis square; colorbar
        title('nRSA (group mean)', 'Interpreter', 'none');
        plot_dRSA_subj_avg(dRSA_diag_avg, dRSA_diag_std, ...
                           mRSA_diag_avg, mRSA_diag_std, ...
                           nRSA_diag_avg, nRSA_diag_std, ...
                           Fs_group, title_plot, lagWindowSec);
    end
else
    warning('Group dRSA: no subject results found. Skipping group average plots.');
end
