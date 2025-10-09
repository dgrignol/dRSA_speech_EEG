% Reset MATLAB session to ensure a clean environment.
clearvars;
close all;
clc;

% Resolve repository paths and derive key locations.
paths = load_paths_config();
audioFile = fullfile(paths.dataStimuli, 'Audio', 'audio_original_merged.wav');
outputDir = fullfile(paths.models, 'wav2vec2');
pythonScript = fullfile(paths.repoBase, 'B2_model_wav2vec2.py');

% Temporarily switch into the repository root to simplify relative paths.
originalDir = pwd;
cleanupObj = [];
if ~strcmp(originalDir, paths.repoBase)
    cd(paths.repoBase);
    cleanupObj = onCleanup(@() cd(originalDir));
end

% Locate the repository virtual environment Python interpreter.
if ispc
    venvPython = fullfile(paths.repoBase, '.venv', 'Scripts', 'python.exe');
else
    venvPython = fullfile(paths.repoBase, '.venv', 'bin', 'python');
end

% Validate critical files and folders before running the pipeline.
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

% Configure MATLAB's Python interface to use the repo-local interpreter.
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

% Define wav2vec2 model parameters and chunking behaviour for extraction.
modelName = 'facebook/wav2vec2-large-xlsr-53';
chunkSeconds = 20; % adjust if you want to trade speed for memory usage
outputBaseName = 'wav2vec2_embeddings';

% Build a launcher that enforces ARM execution on Apple Silicon.
pythonLauncher = sprintf('"%s"', venvPython);
if ismac
    [archStatus, ~] = system('arch -arm64 /usr/bin/true');
    if archStatus == 0
        pythonLauncher = sprintf('arch -arm64 "%s"', venvPython);
    end
end

% Determine whether feature extraction needs to be re-run.
expectedMetadata = fullfile(outputDir, [outputBaseName '.json']);
needExtraction = true;
if isfile(expectedMetadata)
    try
        metadata = jsondecode(fileread(expectedMetadata));
        if isfield(metadata, 'layers') && ~isempty(metadata.layers)
            % Confirm that the final layer MAT file is present.
            if isfield(metadata.layers(end), 'mat_file')
                lastLayerFile = metadata.layers(end).mat_file;
                if isfile(fullfile(outputDir, lastLayerFile))
                    needExtraction = false;
                end
            end
        end
    catch
        needExtraction = true;
    end
end

% Invoke the Python helper when no cached embeddings are available.
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

% Confirm that the expected artefacts exist before exiting.
if ~isfile(expectedMetadata)
    warning('B2_model_wav2vec2:MetadataMissing', ...
        'Metadata file is missing: %s', expectedMetadata);
else
    metadata = jsondecode(fileread(expectedMetadata));
    if ~isfield(metadata, 'layers') || isempty(metadata.layers)
        warning('B2_model_wav2vec2:LayerInfoMissing', ...
            'Metadata does not contain layer information. Verify Python output.');
    else
        finalLayerFile = metadata.layers(end).mat_file;
        finalLayerPath = fullfile(outputDir, finalLayerFile);
        if ~isfile(finalLayerPath)
            error('B2_model_wav2vec2:FinalLayerMissing', ...
                'Final layer MAT file not found: %s', finalLayerPath);
        end
        fprintf('wav2vec2 layer files available (%d total). Final layer: %s\n', ...
            numel(metadata.layers), finalLayerPath);
    end
    fprintf('Metadata saved to %s\n', expectedMetadata);
end

% Restore the original working directory if it was changed.
if ~isempty(cleanupObj)
    clear cleanupObj; %#ok<CLSCR> ensure directory resets immediately
end
