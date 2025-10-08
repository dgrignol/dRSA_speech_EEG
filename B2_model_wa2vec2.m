clearvars;
close all;
clc;

paths = load_paths_config();

audioFile = fullfile(paths.dataStimuli, 'Audio', 'audio_original_merged.wav');
outputDir = fullfile(paths.models, 'wav2vec2');
pythonScript = fullfile(paths.repoBase, 'B2_model_wav2vec2.py');

if ~isfile(audioFile)
    error('B2_model_wa2vec2:AudioMissing', ...
        'Merged audio file not found: %s', audioFile);
end

if ~isfile(pythonScript)
    error('B2_model_wa2vec2:ScriptMissing', ...
        'Python helper script not found: %s', pythonScript);
end

if ~isfolder(outputDir)
    mkdir(outputDir);
end

pythonCmdCandidates = {'python3', 'python'};
pythonCmd = '';
for idx = 1:numel(pythonCmdCandidates)
    candidate = pythonCmdCandidates{idx};
    [status, ~] = system(sprintf('"%s" --version', candidate));
    if status == 0
        pythonCmd = candidate;
        break;
    end
end

if isempty(pythonCmd)
    error('B2_model_wa2vec2:PythonNotFound', ...
        'Unable to locate a usable Python interpreter (tried python3, python).');
end

modelName = 'facebook/wav2vec2-large-xlsr-53';
chunkSeconds = 20; % adjust if you want to trade speed for memory usage
outputBaseName = 'wav2vec2_embeddings';

command = sprintf('"%s" "%s" --audio "%s" --output "%s" --model "%s" --chunk-seconds %.3f --output-base "%s"', ...
    pythonCmd, pythonScript, audioFile, outputDir, modelName, chunkSeconds, outputBaseName);

fprintf('Running wav2vec2 embedding extraction using %s...\n', modelName);

[status, cmdout] = system(command);
if status ~= 0
    error('B2_model_wa2vec2:PythonExecutionFailed', ...
        'Python script failed with exit code %d:\n%s', status, cmdout);
end

fprintf('%s\n', strtrim(cmdout));

expectedNpy = fullfile(outputDir, [outputBaseName '.npy']);
expectedMetadata = fullfile(outputDir, [outputBaseName '.json']);

if ~isfile(expectedNpy)
    error('B2_model_wa2vec2:OutputMissing', ...
        'Expected embedding file not found: %s', expectedNpy);
end

if ~isfile(expectedMetadata)
    warning('B2_model_wa2vec2:MetadataMissing', ...
        'Metadata file is missing: %s', expectedMetadata);
else
    fprintf('Metadata saved to %s\n', expectedMetadata);
end

fprintf('wav2vec2 embeddings saved to %s\n', expectedNpy);
