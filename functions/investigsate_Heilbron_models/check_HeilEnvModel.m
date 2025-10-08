close all; clear all;

eeglabDir = '/Users/damiano/Documents/MATLAB/eeglab2025.0.0'; % e.g., '/path/to/toolboxes/eeglab2025.0.0'
repoBase = '/Users/damiano/Documents/UniTn/Dynamo_ch/dRSA_speech_V02';      % e.g., '/path/to/dRSA_speech_V02'

if isempty(repoBase)
    currentDir = fileparts(mfilename('fullpath'));
    repoBase = fileparts(fileparts(currentDir));
end

if ~exist(repoBase, 'dir')
    error('check_HeilEnvModel:RepoNotFound', 'Repository base directory not found: %s', repoBase);
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
        warning('check_HeilEnvModel:EEGLABNotFound', 'EEGLAB directory not found: %s', eeglabDir);
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
proproc_type = 'Heilb'; % Heilbron et al., 2022

basePathStimuli = paths.dataStimuli;
numOfStim = 20;
method = 'hilbert';
corr_all =[];
pvall = [];
for runNum=1:numOfStim
    
        EnvFilename = sprintf('audio%02d_resampled.wav', runNum);
        filepath = fullfile(basePathStimuli, 'Audio', EnvFilename);
        [audio_signal, Fs] = audioread(filepath);  
        [myEnv, Fs] = compute_envelope(audio_signal, Fs, method);
        save(
        EnvFilename = sprintf('audio%d_128Hz.mat', runNum);
        filepath = fullfile(basePathStimuli, 'Envelopes', EnvFilename);
        data = load(filepath);  % assumes variable is named 'env'
%         figure;
%         t = (0:length(data.env)-1) / Fs;
%         plot(t, data.env);
%         xlabel('Time (s)');
%         ylabel('My Envelope amplitude');
%         title('Audio Envelope');
    
        [corr_curr, pval] = corr(data.env,myEnv','type','Spearman');
        corr_all = [corr_all pvall; corr_curr pval];
end


