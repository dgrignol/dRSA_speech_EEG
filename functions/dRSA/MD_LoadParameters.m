function [opt, cfg] = MD_LoadParameters (opt, cfg)

%update the opt Struct
% opt.models2regressout = {[3 4 5];
%                           [1 4 5];
%                           [1 4 5];
%                           [1,3,5];
%                           [1 3 4];
%                            [3 5];
%                            [3 5];
%                            [3 5]
%                           };


opt.models2regressout = {[4 5];
    [1 4 5];
    [1 4 5];
    [1 5];
    [1 4];
    [1 4 5];
    [1 4 5];
    [1 4 5];
    [1 4 5];
    [1 5];
    [1 5];
    [1 5];
    [1 5];
    [1 4 5];
    [1 4 5];
    [1 4 5];
    [5]
    };


opt.peaks2test = {
    1, [0.1, 0.1];
    3, [-0.6, -0.6];
    3 [1.3, 1.3];
    4, [0, 0];
    5, [0.1, 0.1]
    };

opt.AllLabels = {'position', 'direction', 'direction_angle', 'interaction' 'eyeRDM' ,...
    'PositionVector_100', 'PositionVector_150', 'PositionVector_200', 'PositionVector_250'...
    'GeneralDirection_20', 'GeneralDirection_50', 'GeneralDirection_100', 'GeneralDirection_200',...
    'ProbabilityCenter100', 'ProbabilityCenter200', 'ProbabilityCenter300', 'PredictionError' ...
    }; % direction and angle should be similar

opt.distanceMeasureModel    = {'euclidean', 'euclidean', 'difference', 'interaction (1,3)', 'euclidean', ...
    'correlation', 'correlation', 'correlation',  'correlation', ...
    'difference', 'difference', 'difference', 'difference',...
    'euclidean',  'euclidean',  'euclidean', 'euclidean' };

% Default Values if not provided
if ~isfield(opt, 'Var')
    opt.Var = 0.10;  % Set default value if 'Var' is not provided
end

if ~isfield(opt, 'distanceMeasureNeural')
    opt.distanceMeasureNeural = 'correlation';  % Default distance measure
end

if ~isfield(opt, 'AverageTime')
    opt.AverageTime = 2;  % Default averaging time (in sec)
end


opt.nPCAcombs = 15;

cfg.ROIVec = [1 2 3]; %depends on the list of ROIs


opt.Regressionborder = sqrt(opt.Var ); %apparently, explains 10% of variance ?
opt.sampleDur       = 1/100; % temp resolution: 10 ms
opt.SubSampleDur    = opt.SubSampleDurSec/opt.sampleDur;

opt.spaceStartSec = 0.2;  % Default earliest sequence start (in sec)
opt.spacingSec = 0.1;  % Default spacing (in sec)
opt.spacing         = opt.spacingSec/opt.sampleDur; % between subsamples
opt.spaceStart      = opt.spaceStartSec/opt.sampleDur; % earliest start of subsample after NaN



%update the cfg struct
cfg.ROInames = {'occipital','frontal', 'parietal' 'wholeBrain'};%{'allsensors'};%;%
cfg.badSeg  = 1; % 1 = nans in the data, where there are bad segments. 0 = no nans
cfg.ica     = 1; % 1 = ICA applyed. 0 = no ICA
cfg.fs = 100;
cfg.modelNames = {'Position', 'direction', 'direction-angle', 'interaction' 'eyeRDM',...
    'PositionVector_100', 'PositionVector_150', 'PositionVector_200',  'PositionVector_250', ...
    'GeneralDirection_20', 'GeneralDirection_50', 'GeneralDirection_100', 'GeneralDirection_200',...
    'ProbabilityCenter100', 'ProbabilityCenter200', 'ProbabilityCenter300', 'PredictionError' ...
    }; % direction and angle should be similar

cfg.Condnames = {'σ = 0', 'σ = 9', 'σ = 18' , 'σ = 30'};
cfg.color = parula(length(cfg.condVec ) + 2);
cfg.spaceType = 'MNI';


if isfield (cfg ,'ROI')
    % ROI names
    cfg.ROIold = cfg.ROI;
    
    if strcmp(cfg.ROI, 'LOTC')
        cfg.ROI = {'Area_V4t', 'Area_FST', 'Middle_Temporal_Area', 'Medial_Superior_Temporal_Area', 'Area_Lateral_Occipital_1', 'Area_Lateral_Occipital_2', 'Area_Lateral_Occipital_3', 'Area_PH', 'Area_PHT', 'Area_TemporoParietoOccipital_Junction_2', 'Area_TemporoParietoOccipital_Junction_3'};
    elseif strcmp(cfg.ROI, 'aIPL')
        cfg.ROI = {'Area_PF_Complex', 'Area_PFt', 'Anterior_IntraParietal', 'Area_IntraParietal_2'};
    elseif strcmp(cfg.ROI, 'PMv')
        cfg.ROI = {'Area_IFJa', 'Area_IFJp', 'Rostral_Area_6', 'Ventral_Area_6', 'Premotor_Eye_Field', 'Area_IFSp', 'Area_44', 'Area_45'};
    elseif strcmp(cfg.ROI, 'V3+V4')
        cfg.ROI = {'Third_Visual_Area', 'Fourth_Visual_Area'};
    elseif strcmp(cfg.ROI, 'V2')
        cfg.ROI = {'Second_Visual_Area'};
    elseif strcmp(cfg.ROI, 'V1')
        cfg.ROI = {'Primary_Visual_Cortex'};
    elseif strcmp(cfg.ROI, 'SPL')
        cfg.ROI = {
            'Medial_Area_7P', 'Lateral_Area_7A', 'Medial_Area_7A', 'Lateral_Area_7P', 'Area_7PC', ...
            'Area_Lateral_IntraParietal_ventral', 'Ventral_IntraParietal_Complex', 'Medial_IntraParietal_Area', ...
            'Area_Lateral_IntraParietal_dorsal', 'Anterior_IntraParietal_Area'
            };
    end
    
    cfg = MD_changeROINames(cfg);  %if it is part of list, we update ROIs
    
    
    if iscell(cfg.ROI)
        % Join ROIs with hyphens if it's a cell array
        cfg.Name.ROIstr = strjoin(cfg.ROI, '-');
        
        % Clean the string (replace spaces with underscores, but keep hyphens as they are)
        cleanStr = @(x) strrep(x, ' ', '_');  % Only replace spaces with underscores, keep hyphens
        cfg.Name.ROIstr = cleanStr(cfg.Name.ROIstr);
    else
        cfg.Name.ROIstr = cfg.ROI;  % If it's a single ROI, just use it
    end
    
    if iscell(cfg.ROIold)
        % Join ROIs with hyphens if it's a cell array
        cfg.Name.ROIoldStr = strjoin(cfg.ROIold, '-');
        
        % Clean the string (replace spaces with underscores, but keep hyphens as they are)
        cfg.Name.ROIoldStr = cleanStr(cfg.Name.ROIoldStr);
        cfg.ROIold = cfg.ROIold{:};
        
    else
        cfg.Name.ROIoldStr = cfg.ROIold;  % If it's a single ROI, just use it
    end
    
    
    
else
    cfg.Name.ROIstr = '';  % You can assign a default string if desired
    
    if ~isfield (cfg ,'atlas')
        cfg.atlas = '';
        cfg.side = '';
    end
end


if strcmp(opt.dRSA.corrMethod,'PCR')
    cfg.Name.methodname =  MD_createPCR_nameString (opt);
else
    cfg.Name.methodname = '';
end


%names
if isfield(cfg, 'fisher') && cfg.fisher == 1
    cfg.Name.name1 = 'fishertransformed_';
elseif isfield(cfg, 'fisher')
    cfg.Name.name1 = ''; % or you can leave it as '' if you want to leave it blank
end

if isfield(cfg, 'dRSArelative') && cfg.dRSArelative == 1
    cfg.Name.name2 = 'relative';
elseif isfield(cfg, 'dRSArelative')
    cfg.Name.name2 = ''; % or you can leave it as '' if you want to leave it blank
end

cfg.ntests= 1;  %for Statistics



if strcmp(opt.dRSA.corrMethod, 'PCR')
    cfg.Name.nameYaxis = 'beta';
else
    cfg.Name.nameYaxis = 'corr';
end

end