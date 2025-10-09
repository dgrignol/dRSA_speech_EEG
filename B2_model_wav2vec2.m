clearvars;
close all;
clc;

paths = load_paths_config();
audioFile = fullfile(paths.dataStimuli, 'Audio', 'audio_original_merged.wav');
outputDir = fullfile(paths.models, 'wav2vec2');
pythonScript = fullfile(paths.repoBase, 'B2_model_wav2vec2.py');

originalDir = pwd;
cleanupObj = [];
if ~strcmp(originalDir, paths.repoBase)
    cd(paths.repoBase);
    cleanupObj = onCleanup(@() cd(originalDir));
end
if ispc
    venvPython = fullfile(paths.repoBase, '.venv', 'Scripts', 'python.exe');
else
    venvPython = fullfile(paths.repoBase, '.venv', 'bin', 'python');
end

if ~isfile(audioFile)
    error('B2_model_wav2vec2:AudioMissing', ...
        'Merged audio file not found: %s', audioFile);
end

if ~isfile(pythonScript)
    error('B2_model_wav2vec2:ScriptMissing', ...
        'Python helper script not found: %s', pythonScript);
end

if ~isfolder(outputDir)
    mkdir(outputDir);
end

if ~isfile(venvPython)
    error('B2_model_wav2vec2:VenvMissing', ...
        ['Expected repository virtual environment at %s. ' ...
        'Create it by following the setup instructions in README.md.'], venvPython);
end

currentPy = pyenv;
if currentPy.Status == "Loaded"
    if ~strcmp(currentPy.Executable, venvPython)
        error('B2_model_wav2vec2:PythonMismatch', ...
            ['MATLAB already loaded a different Python executable:\n%s\n' ...
            'Restart MATLAB before running this script or ensure pyenv uses %s.'], ...
            currentPy.Executable, venvPython);
    end
else
    pyenv('Version', venvPython);
end

modelName = 'facebook/wav2vec2-large-xlsr-53';
chunkSeconds = 20; % adjust if you want to trade speed for memory usage
outputBaseName = 'wav2vec2_embeddings';

pythonLauncher = sprintf('"%s"', venvPython);
if ismac
    [archStatus, ~] = system('arch -arm64 /usr/bin/true');
    if archStatus == 0
        pythonLauncher = sprintf('arch -arm64 "%s"', venvPython);
    end
end

expectedNpy = fullfile(outputDir, [outputBaseName '.npy']);
expectedMetadata = fullfile(outputDir, [outputBaseName '.json']);
expectedMat = fullfile(outputDir, [outputBaseName '.mat']);
needExtraction = ~(isfile(expectedNpy) && isfile(expectedMat));

command = sprintf('%s "%s" --audio "%s" --output "%s" --model "%s" --chunk-seconds %.3f --output-base "%s"', ...
    pythonLauncher, pythonScript, audioFile, outputDir, modelName, chunkSeconds, outputBaseName);

if needExtraction
    fprintf('Running wav2vec2 embedding extraction using %s...\n', modelName);
    [status, cmdout] = system(command);
    if status ~= 0
        error('B2_model_wav2vec2:PythonExecutionFailed', ...
            'Python script failed with exit code %d:\n%s', status, cmdout);
    end
    fprintf('%s\n', strtrim(cmdout));
else
    fprintf('Found existing wav2vec2 embeddings. Skipping Python extraction.\n');
end

if ~isfile(expectedNpy)
    error('B2_model_wav2vec2:OutputMissing', ...
        'Expected embedding file not found: %s', expectedNpy);
end

if ~isfile(expectedMetadata)
    warning('B2_model_wav2vec2:MetadataMissing', ...
        'Metadata file is missing: %s', expectedMetadata);
else
    fprintf('Metadata saved to %s\n', expectedMetadata);
end

fprintf('wav2vec2 embeddings saved to %s\n', expectedNpy);

if ~isempty(cleanupObj)
    clear cleanupObj; %#ok<CLSCR> ensure directory resets immediately
end
