% Generate a word2vec-based semantic model aligned to the EEG timeline.
%
% This script:
%   1. Loads textual annotations describing word onset/offset times for each run.
%   2. Collects pretrained word2vec embeddings for the vocabulary.
%   3. Projects those embeddings onto the EEG sampling grid using word durations.
%
% The resulting model is saved under Models/word2vec/ as a matrix of size
% [embedding_dim x time_samples] alongside metadata documenting the procedure.

clearvars;
close all;
clc;

paths = load_paths_config();
if ~isempty(paths.eeglab)
    addpath(paths.eeglab);
end

textDir = fullfile(paths.dataStimuli, 'Text');
if ~isfolder(textDir)
    error('B3_model_word2vec:TextDirMissing', ...
        'Stimulus text directory not found: %s', textDir);
end

outputDir = fullfile(paths.models, 'word2vec');
if ~isfolder(outputDir)
    mkdir(outputDir);
end
outputMatPath = fullfile(outputDir, 'word2vec_model.mat');

preproc_type = '2to20Hz';
numOfSubj = 19;
applyNormalization = false;

% Resolve repository virtual environment Python interpreter.
if ispc
    venvPython = fullfile(paths.repoBase, '.venv', 'Scripts', 'python.exe');
else
    venvPython = fullfile(paths.repoBase, '.venv', 'bin', 'python');
end

if ~isfile(venvPython)
    error('B3_model_word2vec:VenvMissing', ...
        ['Expected repository virtual environment at %s. ' ...
         'Follow the setup instructions in README.md to create it.'], venvPython);
end

helperScript = fullfile(paths.functions, 'python_word2vec_helper.py');
if ~isfile(helperScript)
    error('B3_model_word2vec:HelperMissing', ...
        'Python helper script not found: %s', helperScript);
end

pythonLauncher = sprintf('"%s"', venvPython);
if ismac
    [archStatus, ~] = system('arch -arm64 /usr/bin/true');
    if archStatus == 0
        pythonLauncher = sprintf('arch -arm64 "%s"', venvPython);
    end
end

% Identify a reference EEG dataset to define the time grid.
referenceLen = [];
referenceFs = [];
for subjProbe = 1:numOfSubj
    eegFilename = sprintf('Subject%d_ICA_rej_preproc%s_merged.set', subjProbe, preproc_type);
    eegFolder = fullfile(paths.dataEEG, sprintf('Subject%d', subjProbe));
    eegFullPath = fullfile(eegFolder, eegFilename);
    if isfile(eegFullPath)
        EEG = pop_loadset(eegFilename, eegFolder);
        referenceLen = size(EEG.data, 2);
        referenceFs = EEG.srate;
        fprintf('Using Subject %02d EEG grid (%d samples @ %.2f Hz).\n', ...
            subjProbe, referenceLen, referenceFs);
        clear EEG;
        break;
    end
end

if isempty(referenceLen)
    error('B3_model_word2vec:NoEEGFound', ...
        'Could not locate a merged EEG dataset to define the time grid.');
end

% Gather textual word annotations across runs.
runFiles = dir(fullfile(textDir, 'Run*.mat'));
if isempty(runFiles)
    error('B3_model_word2vec:NoRunFiles', ...
        'No Run*.mat files found under %s', textDir);
end

runIndices = zeros(numel(runFiles), 1);
for idx = 1:numel(runFiles)
    token = regexp(runFiles(idx).name, '^Run(\d+)\.mat$', 'tokens', 'once');
    if isempty(token)
        error('B3_model_word2vec:UnexpectedRunName', ...
            'Unable to parse run index from file %s', runFiles(idx).name);
    end
    runIndices(idx) = str2double(token{1});
end
[~, order] = sort(runIndices);
runFiles = runFiles(order);
runIndices = runIndices(order);

wordEvents = struct('word', {}, 'run', {}, 'onset', {}, 'offset', {});
allTokens = {};
timelineOffset = 0.0;

for runIdx = 1:numel(runFiles)
    runData = load(fullfile(runFiles(runIdx).folder, runFiles(runIdx).name));
    requiredFields = {'wordVec', 'onset_time', 'offset_time'};
    if ~all(isfield(runData, requiredFields))
        error('B3_model_word2vec:MissingFields', ...
            'Run file %s is missing required fields (%s).', ...
            runFiles(runIdx).name, strjoin(requiredFields, ', '));
    end

    words = runData.wordVec(:);
    onsets = double(runData.onset_time(:));
    offsets = double(runData.offset_time(:));

    if ~(numel(words) == numel(onsets) && numel(words) == numel(offsets))
        error('B3_model_word2vec:VectorLengthMismatch', ...
            'Mismatch between word, onset, and offset vectors in %s.', runFiles(runIdx).name);
    end

    for wordIdx = 1:numel(words)
        token = char(words{wordIdx});
        onset = timelineOffset + onsets(wordIdx);
        offset = timelineOffset + offsets(wordIdx);
        if offset < onset
            warning('B3_model_word2vec:NegativeDuration', ...
                'Word %s in run %d has negative duration. Swapping times.', token, runIndices(runIdx));
            tmp = onset;
            onset = offset;
            offset = tmp;
        end
        wordEvents(end + 1).word = token; %#ok<SAGROW>
        wordEvents(end).run = runIndices(runIdx);
        wordEvents(end).onset = onset;
        wordEvents(end).offset = offset;
        allTokens{end + 1} = token; %#ok<SAGROW>
    end

    if isempty(offsets)
        warning('B3_model_word2vec:EmptyRun', ...
            'Run %d contains no word annotations.', runIndices(runIdx));
    else
        timelineOffset = timelineOffset + max(offsets);
    end
end

if isempty(allTokens)
    error('B3_model_word2vec:NoTokensFound', ...
        'No word annotations detected across the provided runs.');
end

uniqueTokens = unique(allTokens);
variantCells = cellfun(@(w) build_token_variants(w), uniqueTokens, 'UniformOutput', false);
embeddingRequests = unique([variantCells{:}]);
embeddingRequests = embeddingRequests(~cellfun('isempty', embeddingRequests));

fprintf('Requesting embeddings for %d unique tokens (including variants).\n', numel(embeddingRequests));

tokenFile = [tempname '.txt'];
embeddingJsonFile = [tempname '.json'];

tokenFID = fopen(tokenFile, 'w');
if tokenFID == -1
    error('B3_model_word2vec:TokenFileWriteFailed', ...
        'Unable to create temporary token file: %s', tokenFile);
end
for idx = 1:numel(embeddingRequests)
    fprintf(tokenFID, '%s\n', embeddingRequests{idx});
end
fclose(tokenFID);

cacheDir = fullfile(outputDir, 'cache');
manualModelPath = fullfile(outputDir, 'GoogleNews-vectors-negative300.bin');
if ~isfile(manualModelPath)
    envManualPath = getenv('WORD2VEC_GOOGLE_NEWS_PATH');
    if ~isempty(envManualPath) && isfile(envManualPath)
        manualModelPath = envManualPath;
    else
        manualModelPath = '';
    end
end

if ~isempty(manualModelPath)
    command = sprintf('%s "%s" --tokens "%s" --output "%s" --cache-dir "%s" --model-path "%s"', ...
        pythonLauncher, helperScript, tokenFile, embeddingJsonFile, cacheDir, manualModelPath);
else
    command = sprintf('%s "%s" --tokens "%s" --output "%s" --cache-dir "%s"', ...
        pythonLauncher, helperScript, tokenFile, embeddingJsonFile, cacheDir);
end
[status, cmdout] = system(command);
if status ~= 0
    if exist(tokenFile, 'file'); delete(tokenFile); end
    if exist(embeddingJsonFile, 'file'); delete(embeddingJsonFile); end
    error('B3_model_word2vec:PythonHelperFailed', ...
        'Python helper failed with exit code %d:\n%s', status, cmdout);
end

embeddingStruct = jsondecode(fileread(embeddingJsonFile));
if exist(tokenFile, 'file'); delete(tokenFile); end
if exist(embeddingJsonFile, 'file'); delete(embeddingJsonFile); end

embeddingMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
fields = fieldnames(embeddingStruct);
missingRequests = {};
for idx = 1:numel(fields)
    key = fields{idx};
    vec = embeddingStruct.(key);
    if isempty(vec)
        missingRequests{end + 1} = key; %#ok<SAGROW>
        embeddingMap(key) = [];
    else
        embeddingMap(key) = single(vec(:));
    end
end

for idx = 1:numel(embeddingRequests)
    key = embeddingRequests{idx};
    if ~embeddingMap.isKey(key)
        embeddingMap(key) = [];
        missingRequests{end + 1} = key; %#ok<SAGROW>
    end
end

sampleVector = [];
for idx = 1:numel(embeddingRequests)
    key = embeddingRequests{idx};
    if embeddingMap.isKey(key)
        vec = embeddingMap(key);
        if ~isempty(vec)
            sampleVector = vec;
            break;
        end
    end
end

if isempty(sampleVector)
    error('B3_model_word2vec:NoEmbeddingsRetrieved', ...
        'No embeddings could be retrieved. Check that gensim downloaded the model correctly.');
end

embeddingDim = numel(sampleVector);
word2vec_model_data = zeros(embeddingDim, referenceLen, 'single');

vectorValues = embeddingMap.values;
vectorCells = vectorValues(~cellfun(@isempty, vectorValues));
if isempty(vectorCells)
    error('B3_model_word2vec:NoEmbeddingsRetrieved', ...
        'No embeddings could be retrieved. Check that the model file is valid.');
end
vectorMatrix = cat(2, vectorCells{:});
noiseMean = single(mean(vectorMatrix, 2));
noiseStd = single(std(vectorMatrix, 0, 2));
noiseStd(noiseStd < sqrt(eps('single'))) = 1;

missingWords = {};
noiseWordCount = 0;
wordMask = false(1, referenceLen);

for idx = 1:numel(wordEvents)
    token = wordEvents(idx).word;
    candidates = build_token_variants(token);
    vector = [];
    for cIdx = 1:numel(candidates)
        candidate = candidates{cIdx};
        if embeddingMap.isKey(candidate)
            candidateVec = embeddingMap(candidate);
            if ~isempty(candidateVec)
                vector = candidateVec;
                break;
            end
        end
    end

    if isempty(vector)
        parts = regexp(lower(token), '[^a-z0-9'']+', 'split');
        parts = parts(~cellfun('isempty', parts));
        if numel(parts) > 1
            partVectors = cell(1, numel(parts));
            for pIdx = 1:numel(parts)
                partToken = parts{pIdx};
                if embeddingMap.isKey(partToken) && ~isempty(embeddingMap(partToken))
                    partVectors{pIdx} = embeddingMap(partToken);
                else
                    partVectors = {};
                    break;
                end
            end
            if ~isempty(partVectors)
                stacked = cat(2, partVectors{:});
                vector = single(mean(stacked, 2));
            end
        end
    end

    if isempty(vector)
        if ~ismember(token, missingWords)
            missingWords{end + 1} = token; %#ok<SAGROW>
        end
        vector = noiseMean + noiseStd .* randn(embeddingDim, 1, 'single');
        noiseWordCount = noiseWordCount + 1;
    end

    onsetSample = max(1, min(referenceLen, floor(wordEvents(idx).onset * referenceFs) + 1));
    offsetSample = max(onsetSample, min(referenceLen, ceil(wordEvents(idx).offset * referenceFs)));
    spanLen = offsetSample - onsetSample + 1;

    vectorSingle = single(vector(:));
    word2vec_model_data(:, onsetSample:offsetSample) = vectorSingle * ones(1, spanLen, 'single');
    wordMask(onsetSample:offsetSample) = true;
end

quietColumns = find(~wordMask);
for idx = 1:numel(quietColumns)
    col = quietColumns(idx);
    word2vec_model_data(:, col) = noiseMean + noiseStd .* randn(embeddingDim, 1, 'single');
end

if applyNormalization
    featureMeanTime = single(mean(word2vec_model_data, 2));
    featureStdTime = single(std(word2vec_model_data, 0, 2));
    featureStdTime(featureStdTime < sqrt(eps('single'))) = 1;
    word2vec_model_data = (word2vec_model_data - featureMeanTime) ./ featureStdTime;
    word2vec_model_data = single(word2vec_model_data);
else
    featureMeanTime = [];
    featureStdTime = [];
end

word2vec_info.embedding_model = 'word2vec-google-news-300';
word2vec_info.embedding_dim = embeddingDim;
word2vec_info.reference_fs = referenceFs;
word2vec_info.reference_len = referenceLen;
word2vec_info.preproc_type = preproc_type;
word2vec_info.missing_request_tokens = unique(missingRequests);
word2vec_info.missing_words = unique(missingWords);
word2vec_info.noise_filled_word_count = noiseWordCount;
word2vec_info.quiet_fill_columns = numel(quietColumns);
word2vec_info.noise_mean = noiseMean;
word2vec_info.noise_std = noiseStd;
word2vec_info.normalization_applied = applyNormalization;
word2vec_info.normalization = struct('mean', featureMeanTime, 'std', featureStdTime);
word2vec_info.created_at = datestr(now, 'yyyy-mm-dd HH:MM:SS');
word2vec_info.total_runs = numel(runFiles);
word2vec_info.num_words = numel(wordEvents);
word2vec_info.pipeline = 'B3_model_word2vec.m';

save(outputMatPath, 'word2vec_model_data', 'word2vec_info', 'wordEvents', '-v7.3');
fprintf('Saved word2vec model to %s (%dx%d).\n', outputMatPath, embeddingDim, referenceLen);

function variants = build_token_variants(word)
    base = char(word);
    trimmed = strtrim(base);
    lowerBase = lower(trimmed);
    variants = {trimmed, lowerBase};

    leadingTrailingStripped = regexprep(lowerBase, '^[^a-z0-9'']+|[^a-z0-9'']+$', '');
    variants{end + 1} = leadingTrailingStripped;
    variants{end + 1} = regexprep(lowerBase, '[^a-z0-9'']+', '');
    variants{end + 1} = strrep(lowerBase, '''', '');

    hyphenParts = regexp(lowerBase, '[-_/]', 'split');
    hyphenParts = hyphenParts(~cellfun('isempty', hyphenParts));
    if numel(hyphenParts) > 1
        variants = [variants, hyphenParts, {strjoin(hyphenParts, ''), strjoin(hyphenParts, '_')}]; %#ok<AGROW>
    end

    apostParts = regexp(lowerBase, '''', 'split');
    apostParts = apostParts(~cellfun('isempty', apostParts));
    if numel(apostParts) > 1
        variants = [variants, apostParts, {strjoin(apostParts, ''), strjoin(apostParts, '_')}]; %#ok<AGROW>
    end

    suffixes = {"'s","'d","'t","'ll","'re","'ve","'m","s","es","ed","ing"};
    for sIdx = 1:numel(suffixes)
        suffix = suffixes{sIdx};
        if length(lowerBase) > length(suffix) && endsWith(lowerBase, suffix)
            variants{end + 1} = lowerBase(1:end - length(suffix)); %#ok<AGROW>
        end
    end

    variants = unique(variants);
    variants = variants(~cellfun('isempty', variants));
end
