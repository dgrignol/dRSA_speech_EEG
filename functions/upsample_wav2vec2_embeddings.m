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
