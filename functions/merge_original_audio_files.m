function mergedAudio = merge_original_audio_files(stimuliDir)
    % Merge individual resampled audio files into one audio_merged.wav file
    % and build a mask in the concatenation points
    %
    % Inputs:
    %   - baseDir: string, base dataset directory (e.g., '.../ds004408')
    %
    % Outputs:
    %   - mergedAudio: concatenated audio signal


    outputFile = fullfile(stimuliDir, 'audio_original_merged.wav');
    
    % Initialise
    mergedAudio = [];
    numStim = 20;

    for i = 1:numStim
        fileName = sprintf('audio%02d.wav', i);
        filePath = fullfile(stimuliDir, fileName);
        if ~exist(filePath, 'file')
            error('File does not exist: %s', filePath);
        end

        [audioData, fs] = audioread(filePath);

        % Append the new audio
        mergedAudio = [mergedAudio; audioData];

    end
    
    
    % Save merged audio
    audiowrite(outputFile, mergedAudio, fs);


end
