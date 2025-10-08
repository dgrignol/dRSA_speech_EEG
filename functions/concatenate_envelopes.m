function [env_all] = concatenate_envelopes(basePathStimuli, name_env_files, numOfStim)
% CONCATENATE_ENVELOPES Concatenates envelope files and checks total length.
%
%   concatenate_envelopes(basePathStimuli, numOfStim)
%
%   Inputs:
%       basePathStimuli - path to the stimulus directory (with trailing '/')
%       numOfStim       - number of audio stimuli (e.g., 20)
%
%   This function loads individual envelope files named
%   'audioX_128Hz.mat' from the 'Envelopes' subfolder, concatenates them,
%   and checks the final length against 'audio_lengths.mat'.
%   The result is saved as 'envelope_all.mat'.

    env_all = [];

    for runNum = 1:numOfStim
        EnvFilename = sprintf(name_env_files, runNum);
        filepath = fullfile(basePathStimuli, 'Envelopes', EnvFilename);
        data = load(filepath);  % assumes variable is named 'env'
        env_all = [env_all; data.env];
    end
    
    % Load lengths and check total
    lengthsFile = fullfile(basePathStimuli, 'Envelopes', 'audio_lengths.mat');
    lengthsData = load(lengthsFile);  % assumes variable is named 'lengths'

    if length(env_all) ~= sum(lengthsData.lengths)
        error('Length of concatenated envelope is incorrect.')
    else
        env_model_data = env_all;
        fprintf('Envelope concatenation successful. \n');
    end
end