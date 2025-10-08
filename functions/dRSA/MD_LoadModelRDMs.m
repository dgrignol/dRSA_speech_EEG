function [models] = MD_LoadModelRDMs (iSub, iCon, cfg, opt)

if isfield(cfg, 'rootdir') && ~isempty(cfg.rootdir)
    rootdir = cfg.rootdir;
elseif isfield(cfg, 'repoBase') && ~isempty(cfg.repoBase)
    rootdir = cfg.repoBase;
else
    error('MD_LoadModelRDMs:MissingRootDir', ...
          'Provide cfg.rootdir or cfg.repoBase before calling MD_LoadModelRDMs.');
end

toolboxPaths = {};
if isfield(cfg, 'toolboxPaths') && ~isempty(cfg.toolboxPaths)
    toolboxPaths = cfg.toolboxPaths;
elseif isfield(cfg, 'fieldtripPath') && ~isempty(cfg.fieldtripPath)
    toolboxPaths = {cfg.fieldtripPath};
end

if ischar(toolboxPaths) || isstring(toolboxPaths)
    toolboxPaths = {char(toolboxPaths)}; %#ok<ISSTR>
elseif ~iscell(toolboxPaths)
    error('MD_LoadModelRDMs:InvalidToolboxPaths', ...
          'cfg.toolboxPaths must be a char, string, or cell array of char.');
end

toolboxPaths = cellfun(@char, toolboxPaths, 'UniformOutput', false);
for iTool = 1:numel(toolboxPaths)
    toolDir = toolboxPaths{iTool};
    if exist(toolDir, 'dir')
        addpath(toolDir);
    else
        warning('MD_LoadModelRDMs:MissingToolbox', ...
                'Toolbox directory not found and skipped: %s', toolDir);
    end
end



% find predictability level
selectedCond = iCon; % predibility: 0,9,18,30 sigma
condVec = selectedCond:4:16;



%% models
fn = fullfile(rootdir, 'MEG_experiment', 'InputFiles', sprintf('MovDot_Sub%02d.mat', iSub));
load(fn);

for iSeq = 1:size(xySeqs,3)
    position(iSeq,:,:)  = xySeqs(selectedCond,1,iSeq).xy'; % events (4 sequences) x features (xy), time points
    AD(iSeq,:,:)        = xySeqs(selectedCond,1,iSeq).AngleDirection'; % events (4 sequences) x features (xy), time points
    direction(iSeq,:,:) = [0 0; diff(squeeze(position(iSeq,:,:))')]';
end


%% Transform direction into the angle

dirAngle = zeros(size(direction,1), 1, size(direction,3));
mag = [];
for iSeq = 1:size(xySeqs ,3)
    for iT = 1:size(direction,3)
        
        if iT == 1
            dirAngle(iSeq, 1, iT) = 0;
        else
            dist = pdist([0,0;direction(iSeq,:, iT)], 'euclidean');
            
            x = direction(iSeq,1, iT);
            y = direction(iSeq,2, iT);
            
            if dist == 0 | x == 0 & y == 0
                theta = 0;
            else
                xnorm = x / dist;
                ynorm  = y / dist;
                theta  = atan2d(ynorm, xnorm);
            end
            
            %rescaledtheta = rescale(theta, opt.Rescale(1), opt.Rescale(2), 'InputMin', 0, 'InputMax', 180);
            
            dirAngle(iSeq, 1, iT) = theta;
            %mag = [mag, dist];
        end
    end
end

%% Load EyeModels

fn = fullfile(rootdir, 'eyeRDM', sprintf('SUB%02d', iSub), ...
              sprintf('eyeTracking_averageddata_sub%02d.mat', iSub));
load(fn);

%average across both x and y Positions
data_combined = (data_averaged(:, [1, 2], :) + data_averaged(:, [3, 4], :)) / 2;

%choose data for Condition   % iSeq x ModelFeatures x TimePoints
EyeModel = data_combined(condVec, :,:);

%Model 5: EyeModel


%% Probability Maps & Prediction Error

%Model 13 - 16

%load them, per participant

%100ms
datadir  = fullfile(rootdir, 'dRSA', 'ProbabilityMaps', 'ProbabilityMaps');
filename = fullfile(datadir, sprintf('ProbabilityMap_iSub_%02d_%d-times_%dms', iSub, 1000, 100));
load(filename);

idx = find([ProbabilityData.Condition] == iCon);

C = {ProbabilityData(idx).ProbabilityCenter_Now};  % 1×4 cell array, each 1800×2
D = [ProbabilityData(idx).PredictionError]';
bigMat = cell2mat(C');  % Each cell becomes a block vertically stacked
Probability_temp_100 = permute(reshape(bigMat, 1800, [], 2), [2, 3, 1]);
PredictionErr_temp_100 = reshape(D, 4, 1, 1800);

Probability_Map_200 = [];
Probability_Map_300 = [];

%200ms
% clearvars ProbabilityData bigMat C D idx;
% 
% filename = fullfile(datadir, sprintf('ProbabilityMap_iSub_%02d_%d-times_%dms', iSub, 1000, 200));
% load(filename);
% 
% idx = find([ProbabilityData.Condition] == iCon);
% 
% C = {ProbabilityData(idx).ProbabilityCenter_Now};  % 1×4 cell array, each 1800×2
% D = [ProbabilityData(idx).PredictionError]';
% bigMat = cell2mat(C');  % Each cell becomes a block vertically stacked
% Probability_temp_200_ = permute(reshape(bigMat, 1800, [], 2), [2, 3, 1]);
% PredictionErr_temp_200 = reshape(D, 4, 1, 1800);
% 
% 
% %300ms
% clearvars ProbabilityData bigMat C D idx;
% 
% filename = fullfile(datadir, sprintf('ProbabilityMap_iSub_%02d_%d-times_%dms', iSub, 1000, 300));
% load(filename);
% 
% idx = find([ProbabilityData.Condition] == iCon);
% 
% C = {ProbabilityData(idx).ProbabilityCenter_Now};  % 1×4 cell array, each 1800×2
% D = [ProbabilityData(idx).PredictionError]';
% bigMat = cell2mat(C');  % Each cell becomes a block vertically stacked
% Probability_temp_300 = permute(reshape(bigMat, 1800, [], 2), [2, 3, 1]);
% PredictionErr_temp_300 = reshape(D, 4, 1, 1800);
% 
% 


%% interpolate so that neural and model data have the same number of time points

TPmodels = size(position,3);
TPneural = 3000;
timeVecmodels = 1:TPmodels;
timeVecNeural = 1:TPneural;
timeVecDesired = linspace(1,TPmodels,TPneural) ;


for iSeq = 1:size(xySeqs,3)
    video = squeeze(position(iSeq,:,:));
    positionNew(iSeq,:,:) = interp1(timeVecmodels,video',timeVecDesired,'nearest')';
    video = squeeze(direction(iSeq,:,:));
    directionNew(iSeq,:,:) = interp1(timeVecmodels,video',timeVecDesired,'nearest')';
    video = squeeze(dirAngle(iSeq,1,:));
    dirAngleNew(iSeq,1,:) = interp1(timeVecmodels,video',timeVecDesired,'nearest')';
    
    video = squeeze(Probability_temp_100(iSeq,:,:));
    Probability_Map_100(iSeq,:,:) = interp1(timeVecmodels,video',timeVecDesired,'nearest')';
    
    video = squeeze(PredictionErr_temp_100(iSeq,:,:));
    PredictionError_100(iSeq,1,:) =  interp1(timeVecmodels,video',timeVecDesired,'nearest')';
    
    %video = squeeze(AD(iSeq,:,:));
    %ADold = squeeze(interp1(timeVecmodels,video',timeVecDesired,'nearest')');
    %ADNew(iSeq,1,:) = interp1(timeVecmodels,video',timeVecDesired,'nearest')';
    
%     
%     video = squeeze(Probability_temp_200(iSeq,:,:));
%     Probability_Map_200(iSeq,:,:) = interp1(timeVecmodels,video',timeVecDesired,'nearest')';
%     
%       
%     video = squeeze(Probability_temp_300(iSeq,:,:));
%     Probability_Map_300(iSeq,:,:) = interp1(timeVecmodels,video',timeVecDesired,'nearest')';
%    
    
    
end
%Model 1: Position
%Model 2: Old Direction (just euclidean vector)
%Model 3: Direction as angle
%Model 13: ProbabilityMap
%Model 14: prediction error

%Model 4:  empty file for interaction
interaction = zeros(size(xySeqs,3), 1, TPneural);

%% Position Vector:

%Model 6, 7, 8:  (50, 100, 200 Vector of next Postion)

Pos100 = MD_createPositionVector_v2(positionNew, 100);
Pos150 = MD_createPositionVector_v2(positionNew, 150);
Pos200 = MD_createPositionVector_v2(positionNew, 200);
Pos250 = MD_createPositionVector_v2(positionNew, 250);



%% Direction Difference
%Model 10, 11, 12, 13
dir20 = MD_createDirectionMean_v2 (positionNew, 20);
dir50 = MD_createDirectionMean_v2 (positionNew, 50);
dir100 = MD_createDirectionMean_v2 (positionNew, 100);
dir200 = MD_createDirectionMean_v2 (positionNew, 200);




%% Put data into struct

data_mod = {positionNew, directionNew, dirAngleNew, interaction, EyeModel, ...
    Pos100, Pos150, Pos200, Pos250, ...
    dir20, dir50, dir100, dir200,...
    Probability_Map_100, Probability_Map_200, Probability_Map_300, ...
    PredictionError_100};

models.data = data_mod(cfg.oldModels);  %use only Models that we need for later
models.labels = opt.AllLabels;



end
