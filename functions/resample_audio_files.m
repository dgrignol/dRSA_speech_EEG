function resample_audio_files(basePathStimuli, numOfStim, targetFs)
% RESAMPLE_AUDIO_FILES Resamples and saves audio files to target sampling rate.
%
%   resample_audio_files(basePathStimuli, numOfStim, targetFs)
%
%   Inputs:
%       basePathStimuli - Base directory containing 'Audio' subfolder
%       numOfStim       - Number of audio files to process (assumes names like audio01.wav)
%       targetFs        - Target sampling rate for resampling


    % Load lengths and check total
%     lengthsFile = fullfile(basePathStimuli, 'Envelopes', 'audio_lengths.mat');
    lengthsData = load('audio_lengths.mat');  % assumes variable exist in a loaded path (e.g. functions)

    % Assumes var loaded is named 'lengths' (so lengthsData.lengths)


    for runNum = 1:numOfStim
        % Create filename
        AudioFilename = sprintf('audio%02d.wav', runNum);
        filepath = fullfile(basePathStimuli, 'Audio', AudioFilename);

        % Read original audio
        [audioData, fs] = audioread(filepath);

        % Resample audio
        audioData_resampled = resample(audioData, targetFs, fs);


    if length(audioData_resampled) ~= lengthsData.lengths(runNum)
        error('Length of concatenated audio is incorrect.')
    else
        % Save resampled audio
        ResampledFilename = sprintf('audio%02d_resampled.wav', runNum);
        outputPath = fullfile(basePathStimuli, 'Audio', ResampledFilename);
        audiowrite(outputPath, audioData_resampled, targetFs);
        sprintf('Audio concatenation successful. Saved to audio%02d_resampled.wav file\n',runNum);
    end
    
    end
    
    

    
    
end