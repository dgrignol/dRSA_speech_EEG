clearvars;
close all;
clc;

paths = load_paths_config();
addpath(paths.eeglab)
addpath(paths.functions)

basePathStimuli = paths.dataStimuli;
basePathModels = paths.models;
numOfStim = 20;
name_suffix = 'Hilbert'; % options include: 'Hilbert', 'Heilb', etc.
targetFs = 128;

if ~isfolder(basePathStimuli)
    error('B1_model_envelope:StimuliPathMissing', 'Stimuli folder not found: %s', basePathStimuli);
end

envelopeRunDir = fullfile(basePathStimuli, 'Envelopes');
if ~isfolder(envelopeRunDir)
    mkdir(envelopeRunDir);
end

if ~strcmpi(name_suffix, 'Heilb')
    fprintf('Computing %s envelopes for %d stimuli at %d Hz...\n', name_suffix, numOfStim, targetFs);
    for runNum = 1:numOfStim
        EnvFilename = sprintf('audio%02d_resampled.wav', runNum);
        filepath = fullfile(basePathStimuli, 'Audio', EnvFilename);
        if ~isfile(filepath)
            error('B1_model_envelope:AudioMissing', 'Audio file not found: %s', filepath);
        end
        [audio_signal, Fs] = audioread(filepath);
        [env, Fs] = compute_envelope(audio_signal, Fs, name_suffix);
        env = env(:)'; % ensure row vector
        save(fullfile(envelopeRunDir, sprintf('audio%d_128Hz_%s.mat', runNum, name_suffix)), 'env');
    end
end

name_env_files = ['audio%d_' sprintf('128Hz_%s.mat', name_suffix)];
name_env_model_file = sprintf('envelope_%s.mat', name_suffix);

fprintf('Concatenating single-run envelopes...\n');
env_model_data = concatenate_envelopes(basePathStimuli, name_env_files, numOfStim);

outputDir = fullfile(basePathModels, 'Envelopes');
if ~isfolder(outputDir)
    mkdir(outputDir);
end

save(fullfile(outputDir, name_env_model_file), 'env_model_data');
fprintf('Envelope model saved to %s\n', fullfile(outputDir, name_env_model_file));
