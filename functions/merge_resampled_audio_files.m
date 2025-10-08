function [mergedAudio, fs_check, maskConcat] = merge_resample_audio_files(stimuliDir, maskFolderPath, test, sartEnd_span)
    % Merge individual resampled audio files into one audio_merged.wav file
    % and build a mask in the concatenation points
    %
    % Inputs:
    %   - baseDir: string, base dataset directory (e.g., '.../ds004408')
    %   - test: logical, optional (default=false), whether to run in test mode
    %   - sartEnd_span: span (in seconds) before and after the junctions to mark with 1 in mask_concat
    %
    % Outputs:
    %   - mergedAudio: concatenated audio signal
    %   - fs_check: sampling rate of the merged signal
    %   - mask_concat: binary mask indicating ±sartEnd_span seconds around concatenation points

    if nargin < 2
        test = false;
    end
    if nargin < 3
        sartEnd_span = 3;  % default to 3 seconds if not provided
    end

    outputFile = fullfile(stimuliDir, 'audio_merged.wav');

    % Number of stimuli to merge
    if test
        numStim = 3;
    else
        numStim = 20;
    end

    % Initialise
    mergedAudio = [];
    maskConcat = [];
    fs_check = [];

    for i = 1:numStim
        fileName = sprintf('audio%02d_resampled.wav', i);
        filePath = fullfile(stimuliDir, fileName);
        if ~exist(filePath, 'file')
            error('File does not exist: %s', filePath);
        end

        [audioData, fs] = audioread(filePath);

        if isempty(fs_check)
            fs_check = fs;
        elseif fs ~= fs_check
            error('Sampling rate mismatch at %s', fileName);
        end

        % Append the new audio
        prev_length = length(mergedAudio);
        mergedAudio = [mergedAudio; audioData];

        % Create mask
        audio_len = length(audioData);
        mask_segment = zeros(audio_len, 1);

        if i == 1
            % First segment: just append corresponding zeros
            maskConcat = [maskConcat; mask_segment];
        else
            % Define span in samples
            span_samples = round(sartEnd_span * fs);
            start_idx = max(prev_length - span_samples + 1, 1);
            end_idx = min(prev_length + span_samples, prev_length + audio_len);

            % Extend mask_concat if needed
            if length(maskConcat) < prev_length + audio_len
                maskConcat(prev_length + 1 : prev_length + audio_len) = 0;
            end
            maskConcat(start_idx:end_idx) = 1;
        end
    end
    
    % Add 1s at the start and end of the merged audio
    span_samples = round(sartEnd_span * fs_check);
    maskConcat(1 : span_samples) = 1;
    maskConcat(end - span_samples + 1 : end) = 1;
    
    % Save merged audio
    audiowrite(outputFile, mergedAudio, fs_check);
    % Save concatenation points mask
    if ~exist(maskFolderPath, 'dir')
        mkdir(maskFolderPath);
    end
    maskConcat = logical(maskConcat');
    save(fullfile(maskFolderPath, 'mask_concat.mat'), 'maskConcat');


    %% Summary plot
    % Generate time vector
    t = (0:length(mergedAudio)-1) / fs_check;

    % Plot the audio waveform
    figure;
    plot(t, mergedAudio);
    hold on;
    yl = ylim;

    % Find transitions in mask (start and end of 1s)
    dMask = diff([0; maskConcat'; 0]); % Pad to catch edges
    startIdx = find(dMask == 1);
    endIdx = find(dMask == -1) - 1;  % inclusive

    % Plot shaded regions where mask == 1
    for i = 1:length(startIdx)
        t1 = (startIdx(i)-1) / fs_check;
        t2 = (endIdx(i)-1) / fs_check;
        patch([t1 t2 t2 t1], [yl(1) yl(1) yl(2) yl(2)], ...
              'Blue', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end

    title('Merged Audio with Mask Overlay');
    xlabel('Time (s)');
    ylabel('Amplitude');
    legend({'Audio', 'Masked regions'});



end
