function compute_and_save_envelopes(basePathStimuli, numOfStim, method, name_suffix)
% COMPUTE_AND_SAVE_ENVELOPES - Computes and saves the envelope of resampled audio files.
%
% Inputs:
%   basePathStimuli      - Base path where 'Audio' and 'Envelopes' folders are located
%   numOfStim            - Number of stimuli/audio files to process
%   method               - Method used in compute_envelope function (e.g., 'hilbert')
%   name_env_model_file  - Name of the output envelope .mat file (e.g., 'audio01_128Hz.mat')

    for runNum = 1:numOfStim
        EnvFilename = sprintf('audio%02d_resampled.wav', runNum);
        filepath = fullfile(basePathStimuli, 'Audio', EnvFilename);
        
        [audio_signal, Fs] = audioread(filepath);  
        [env, Fs] = compute_envelope(audio_signal, Fs, method);
        
        % Create unique name for each envelope file if needed
        save_name = sprintf('audio%d_128Hz_%s.mat', runNum, name_suffix);  % e.g., 'audio%02d_128Hz.mat'
        save(fullfile(basePathStimuli, 'Envelopes', save_name), 'env');
    end
end