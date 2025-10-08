function paths = load_paths_config()
%LOAD_PATHS_CONFIG Load repository paths and configure the MATLAB environment.
%
% The function reads base directories from config_paths.json, verifies that
% they exist, adds the relevant folders to the MATLAB path, and returns a
% structure with the commonly used paths.
%
% Usage:
%   paths = load_paths_config();
%
% Returns:
%   paths (struct) containing the main repository folders (data, results,
%   models, etc.) and the EEGLAB directory, if provided.

    configFile = fullfile(fileparts(mfilename('fullpath')), 'config_paths.json');
    if ~isfile(configFile)
        error('load_paths_config:ConfigNotFound', ...
              'Configuration file not found: %s', configFile);
    end

    rawText = fileread(configFile);
    config = jsondecode(rawText);

    if ~isfield(config, 'repoBase') || isempty(config.repoBase)
        repoBase = fileparts(mfilename('fullpath'));
    else
        repoBase = config.repoBase;
    end

    if ~isfolder(repoBase)
        error('load_paths_config:RepoNotFound', ...
              'Repository base directory not found: %s', repoBase);
    end

    if isfield(config, 'eeglabDir')
        eeglabDir = config.eeglabDir;
    else
        eeglabDir = '';
    end

    if ~isempty(eeglabDir) && ~isfolder(eeglabDir)
        warning('load_paths_config:EEGLABNotFound', ...
                'EEGLAB directory not found: %s', eeglabDir);
        eeglabDir = '';
    end

    addpath(repoBase);
    addpath(fullfile(repoBase, 'functions'));
    addpath(fullfile(repoBase, 'functions', 'dRSA'));
    addpath(fullfile(repoBase, 'functions', 'modelCreation'));
    addpath(fullfile(repoBase, 'functions', 'investigsate_Heilbron_models'));

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
end
