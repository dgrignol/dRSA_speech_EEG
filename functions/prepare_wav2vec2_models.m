function [referenceInfo, wav2vec2Models] = prepare_wav2vec2_models(basePathModels, basePathEEG, preprocType, numSubjects, layerIndices)
%PREPARE_WAV2VEC2_MODELS Load and resample wav2vec2 embeddings for dRSA.
%
%   [referenceInfo, wav2vec2Models] = PREPARE_WAV2VEC2_MODELS(basePathModels, ...
%       basePathEEG, preprocType, numSubjects, layerIndices) loads wav2vec2
%   layer metadata, verifies availability of requested layers, identifies a
%   reference EEG dataset to define the shared timeline, and resamples each
%   wav2vec2 embedding onto that grid.
%
%   referenceInfo contains:
%       .length          - number of samples in the reference EEG
%       .fs              - sampling rate of the reference EEG
%       .preloadedEEG    - EEG dataset object for the reference subject
%       .firstSubjectIdx - subject index used to establish the reference grid
%
%   wav2vec2Models contains:
%       .data            - cell array of resampled wav2vec2 matrices
%       .labels          - cell array of labels matching .data

    if nargin < 5
        error('prepare_wav2vec2_models:MissingArguments', ...
            'All inputs (basePathModels, basePathEEG, preprocType, numSubjects, layerIndices) are required.');
    end

    metadataPath = fullfile(basePathModels, 'wav2vec2', 'wav2vec2_embeddings.json');
    if ~isfile(metadataPath)
        error('prepare_wav2vec2_models:MetadataMissing', ...
            'wav2vec2 metadata not found: %s', metadataPath);
    end
    metadata = jsondecode(fileread(metadataPath));
    if ~isfield(metadata, 'layers') || isempty(metadata.layers)
        error('prepare_wav2vec2_models:LayersMissing', ...
            'wav2vec2 metadata does not describe any layer MAT files: %s', metadataPath);
    end

    layersAvailable = arrayfun(@(layer) layer.index, metadata.layers);
    [isLayerPresent, layerPositions] = ismember(layerIndices, layersAvailable);
    if ~all(isLayerPresent)
        missingLayers = layerIndices(~isLayerPresent);
        error('prepare_wav2vec2_models:LayerNotFound', ...
            'Requested wav2vec2 layer indices not available: %s', num2str(missingLayers));
    end
    selectedLayers = metadata.layers(layerPositions);

    referenceLen = [];
    referenceFs = [];
    preloadedEEG = [];
    firstSubjectIdx = NaN;
    for subjProbe = 1:numSubjects
        eegFile = sprintf('Subject%d_ICA_rej_preproc%s_merged.set', subjProbe, preprocType);
        eegFolder = fullfile(basePathEEG, sprintf('Subject%d', subjProbe));
        eegPath = fullfile(eegFolder, eegFile);
        if isfile(eegPath)
            preloadedEEG = pop_loadset(eegFile, eegFolder);
            referenceLen = size(preloadedEEG.data, 2);
            referenceFs = preloadedEEG.srate;
            firstSubjectIdx = subjProbe;
            fprintf('Using Subject %02d to define wav2vec2 resampling grid (%d samples @ %.2f Hz).\n', ...
                subjProbe, referenceLen, referenceFs);
            break;
        end
    end

    if isempty(referenceLen)
        error('prepare_wav2vec2_models:NoEEGFound', ...
            'Unable to locate any merged EEG dataset to define wav2vec2 resampling.');
    end

    requiredFields = {'embeddings', 'time_axis'};
    wav2vec2Resampled = cell(1, numel(selectedLayers));
    wav2vec2Labels = cell(1, numel(selectedLayers));
    for layerIdx = 1:numel(selectedLayers)
        matPath = fullfile(basePathModels, 'wav2vec2', selectedLayers(layerIdx).mat_file);
        if ~isfile(matPath)
            error('prepare_wav2vec2_models:LayerFileMissing', ...
                'wav2vec2 layer file not found: %s', matPath);
        end
        layerData = load(matPath);
        if ~all(isfield(layerData, requiredFields))
            error('prepare_wav2vec2_models:LayerFieldsMissing', ...
                'wav2vec2 layer file %s missing fields: %s', matPath, strjoin(requiredFields, ', '));
        end
        wav2vec2Resampled{layerIdx} = upsample_wav2vec2_embeddings(layerData, referenceLen, referenceFs)';
        layerLabel = selectedLayers(layerIdx).label;
        if isstring(layerLabel) || ischar(layerLabel)
            layerLabel = char(layerLabel);
        else
            layerLabel = sprintf('layer%d', selectedLayers(layerIdx).index);
        end
        wav2vec2Labels{layerIdx} = sprintf('wav2vec2.%s', layerLabel);
    end

    referenceInfo.length = referenceLen;
    referenceInfo.fs = referenceFs;
    referenceInfo.preloadedEEG = preloadedEEG;
    referenceInfo.firstSubjectIdx = firstSubjectIdx;

    wav2vec2Models.data = wav2vec2Resampled;
    wav2vec2Models.labels = wav2vec2Labels;
end
