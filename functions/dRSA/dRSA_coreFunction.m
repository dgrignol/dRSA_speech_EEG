function [dRSA, nRSA, mRSA] = dRSA_coreFunction(Y,models,opt,cfg,debugMode)

% core function for dRSA

% INPUT VARIABLES:
% Y (independent variable, e.g. neural or behavioral data): 4-D num array with dimensions ERFT = (E)vents x (R)epetitions x (F)eatures x (T)ime points
% this can be:
% - 1x1x1xN (a single event with a single feature, e.g. pupil dilation during watching a movie, or a single neuron during a single free moving event)
% - 1x1xMxN (same as above but with >1 features, e.g. xy eye positions, or several neurons)
% - 1xLxMxN (same as above but with >1 repetitions of the same event. Data will be averaged to 1x1xMxN)
% - KxLxMxN (>1 event, e.g. several trials, from which the subsampling will be done. Data will be concatenated to 1x1xMxN)
% - mix of above, e.g. Kx1xMxN (several events without repetition)
%
% models (dependent variable, e.g. video pixels, kinematic markers, DNN
% layers): struct array with the following fields:
% models.data (1xN cell array, with 3-D num arrays with dimensions EFT in each cell that match Y in number of Events and Time points)
% models.labels (1xN cell array names of models corresponding to order of models.data)
%
% Y and models may contain NaN.
%
% opt: struct array with the following options:
%
% opt.nSubSamples (how many subsamples should be selected from Event)
% opt.SubSampleDur (how many time points per subsample. nSubSamples*SubSampleDur must be smaller than number of time points
% opt.nIter (how many subsampling iterations. If opt.SubSampleDur ==  size(KxLxMxN,1): no subsampling, e.g. 14 5s-long trials, dRSA on the full 5s)
%
% opt.spacing (minimum distance (num of time points) between subsamples. Default = 1)
% opt.spaceStart (num of time points to be excluded after start of each Event. Default = 1) CONSIDER EXTENDING IT TO ANY NAN IN MASK
%
% opt.modelVec (index vector for models to be included in dRSA. Default = all --> 1:numel(models))
% opt.distanceMeasureModel (pdist measure for model RDMs, e.g. 'euclidean', 'correlation')
% opt.distanceMeasureNeural (pdist measure for neural RDMs, e.g. 'euclidean', 'correlation')
%
% opt.dRSA.corrMethod ('correlation','PCR')
%
% opt.mask: optional additional mask: 1xN cell array with N masks provided as 2-D logical or num array with dimensions ET (matching Y / models): 0=keep, 1=replace Y and models with NaN for these time points
% opt.maskLabels: label(s) for the user-provided mask(s)
% the mask will also be created/extended in this function using opt.spacing, opt.spaceStart
%
% opt.autocorr: optional: compute also model and neural auto-correlations
%
% OUTPUT VARIABLES:
% dRSA = 3-D NxNxM num array, where N = opt.SubSampleDur and M = num of tested models

origDims    = size(Y);
nEvents     = origDims(1);
nReps       = origDims(2);
nFeatures   = origDims(3);
nTPevents   = origDims(4);
fprintf('data: %d event(s), %d repetition(s), %d feature(s), %d time points\n',nEvents,nReps,nFeatures,nTPevents);

% which models will be used?
if ~isfield(opt,'modelVec')
    opt.modelVec = 1:numel(models.data);
end
disp('we will use these models:')
for i = opt.modelVec
    disp(['  - ' models.labels{i}]);
end

if ~exist('debugMode')
    debugMode = false;
end

%% average across repetitions
if size(Y,2)>1
    disp('average across repetitions');
    Y = mean(Y,2,'omitnan');
end

Y = squeeze(Y);
if nFeatures == 1 % needs to be transposed if only one feature so that Y is 1xN
    Y = Y';
end

%% concatenate events
if nEvents>1 % if there is more than 1 event
    disp('concatenate events in data');
    Y2 = shiftdim(Y,2); % transpose to TP x events x features
    Y = reshape(Y2, nTPevents*nEvents, nFeatures)'; % concatenate (3D --> 2D)
    
    disp('concatenate events in models');
    for i=1:numel(models.data)
        nModelFeatures = size(models.data{i},2); % models can have different num of features
        M = shiftdim(models.data{i},2); % transpose to TP x events x features
        models.data{i} = reshape(M,nTPevents*nEvents,nModelFeatures)';
    end
end
nTP = size(Y,2); % final number of time points
fprintf('total number of TPs: %d\n',nTP);

%% set up mask

% start points of event(s)
if nEvents>1 % if there is more than 1 event
    % create a logical vector with 1=start of event
    eventStartVec = zeros(1,nTP);
    eventStartVec(1:nTPevents:nTP) = 1;
    mask = eventStartVec;
else
    mask = zeros(1,size(Y,2));
    mask(1) = 1; % first TP is start of event
end

% add masking of TPs after event start points
idx = repmat(find(mask),opt.spaceStart,1) + repmat(0:opt.spaceStart-1, nEvents,1)';
mask(idx(:))    = 1;
maskLabels{1}   = 'event start';

% NaNs in Y
mask(2,:) = any(isnan(Y));
maskLabels{2} = 'NaNs in data';

% NaN in models
for i=1:numel(opt.allModels)
%     mask(2+i,:) = any(isnan(models.data{opt.allModels(i)}));
    mask(2+i,:) = any(isnan(models.data{opt.allModels(i)}), 1);
    maskLabels{2+i} = sprintf('NaNs in model (%s)', models.labels{opt.allModels(i)});
end


if isfield(opt,'mask') % if another mask is provided, add it to the existing one
    for i = 1:numel(opt.mask)
        mask = [mask; opt.mask{i}];
        maskLabels{end+1} = sprintf('user mask (%s)',opt.maskLabels{i});
    end
end

%% Mask prep

maskSubsampling = any(mask); % turn into a 1D mask that combines all individual masks
maskSubsamplingStarts = any(mask); % and another mask that is only used to find random start positions of the subsamples

% check if enough samples for planned sequences
TotalDurSeqS = opt.SubSampleDur*opt.nSubSamples + opt.spacing*(opt.nSubSamples-1) + sum(maskSubsampling==1);

if TotalDurSeqS > nTP
    disp('too many subsamples/ subsamples too long');
    %disp('for now, I overwrite the num of subsamples, so that the script can continue:');
    %opt.nSubSamples = 2;
    return;
end

% % exclude timepoints after masked-out samples
% idx = (repmat(find(maskAve)',opt.spaceStart,1) + (0:opt.spaceStart-1)')';
% idx(idx>nSamples)=[];

% make a mask that also considers the duration of subsamples, so that subsamples do not overlap with masked time points:
idxSubsampleStart = (repmat(find(maskSubsampling),opt.SubSampleDur,1) + (-opt.SubSampleDur:-1)')';
idxSubsampleStart(idxSubsampleStart<1)=[];
maskSubsamplingStarts(idxSubsampleStart(:)) = 1;


%% subsample (get start and end points of subsamples)
cfg.sampling='subsampling';
if strcmp(cfg.sampling,'subsampling')
    fn = fullfile(cfg.SubSampleDir, sprintf('Subsamples_%d-Dur_%d-nSubSamples_%d-nIter.mat', opt.SubSampleDurSec, opt.nSubSamples, opt.nIter));
    if exist(fn) == 2 && sum(sum(isnan(Y))) == 0
        load(fn);
        fl = fullfile(cfg.SubSampleDir, sprintf('SubSamplesPlot_%d-Dur_%d-nSubSamples_%d-nIter.mat', opt.SubSampleDurSec, opt.nSubSamples, opt.nIter));
        load(fl);
    else
        [SSIndices SSIndicesPlot] = dRSA_subsampling_v3(maskSubsampling, opt); % SS = subsamples#
        save(fn,'SSIndices');
        fl = fullfile(cfg.SubSampleDir, sprintf('SubSamplesPlot_%d-Dur_%d-nSubSamples_%d-nIter.mat', opt.SubSampleDurSec, opt.nSubSamples, opt.nIter));
        save(fl,'SSIndicesPlot');
    end
elseif strcmp(cfg.sampling,'timelocked')
    [SSIndices SSIndicesPlot] = XXX; % TO DO (just take a vector of onset indices specified as cfg.SampleOnsets)
end



%% plot

% Get the min and max of Y where mask == 1
masked_vals = Y(mask == 0);
clims = [min(masked_vals(:)), max(masked_vals(:))];

% Plot
figure;
subplot(3,1,1);
imagesc(Y, clims); % Apply the color limits here
colorbar;
xlabel('time points');
ylabel('features');
title('data');

subplot(3,1,2);
imagesc(mask);
colorbar;
xlabel('time points');
yticklabels(maskLabels)
title('mask');

subplot(3,1,3);
imagesc(SSIndicesPlot); % all subsampling iterations
colorbar;
xlabel('time points');
ylabel('iterations');
title(sprintf('%d subsamples, dur: %d time points',opt.nSubSamples,opt.SubSampleDur));


%% dRSA

mRDMs = [];
nRDMs = [];
dRSA = [];
%fprintf('\nrun dRSA:      ')

dRSA = zeros(opt.nIter, opt.SubSampleDur, opt.SubSampleDur, numel(opt.modelVec));  %all models should have the same size, so we just pick the first


for i = 1:opt.nIter
    %fprintf('\b\b\b\b\b%04d ',i);
    fprintf('\n run dRSA: %04d ',i);
    
    for iT = 1:opt.SubSampleDur
        
        % for Model Data
        for iModel = opt.allModels
            if  strcmp(opt.distanceMeasureModel{iModel}, 'difference')
                %                     mRDMs{iModel}(:,iT) = pdist(models.data{opt.modelVec(iModel)}(:,SSIndices(:,iT,i))',@calculateAngleDistance);
                mRDMs{iModel}(:,iT) = pdist(models.data{iModel}(:,SSIndices(:,iT,i))',@diff);
                
            elseif contains(opt.distanceMeasureModel{iModel}, 'interaction')
                str = opt.distanceMeasureModel{iModel};
                numbers = regexp(str, 'interaction \((\d+),(\d+)\)', 'tokens');
                if ~isempty(numbers)
                    extractedNumbers = str2double(numbers{1});
                else
                    disp('No match found.');
                end
                
                clearvars model1 model2
                model1 = rescale( mRDMs{extractedNumbers(1)} (:,iT), 1, 2);
                model2 = rescale( mRDMs{extractedNumbers(2)} (:,iT), 1, 2);
                
                mRDMs{iModel}(:,iT) = model1.*model2;
            else
                %                    mRDMs{iModel}(:,iT) = pdist(models.data{opt.modelVec(iModel)}(:,SSIndices(:,iT,i))',opt.distanceMeasureModel{iModel});
                mRDMs{iModel}(:,iT) = pdist(models.data{iModel}(:,SSIndices(:,iT,i))',opt.distanceMeasureModel{iModel});
            end
        end
        
        %for neural data
        nRDMs(:,iT) = pdist(Y(:,SSIndices(:,iT,i))',opt.distanceMeasureNeural);  % 'correlation'
        
    end

    
    %% Debug section
    if debugMode
        % check RDMs across iT
        figure;
        for iT = 1:15
            time_indices = SSIndices(:, iT, 1);  % first iteration
            neural_window = Y(:, time_indices);  % [channels x time]
            neural_rdm = pdist(neural_window', 'correlation');
            subplot(3, 5, iT);
            imagesc(squareform(neural_rdm));
            title(sprintf('Neural RDM iT=%d', iT));
        end
        sgtitle('Neural RDMs across iT (iter 1)');

        figure;
        for iT = 1:15
            time_indices = SSIndices(:, iT, 1);
            model_window = models.data{iModel}(:, time_indices);
            model_rdm = pdist(model_window', 'euclidean');
            subplot(3, 5, iT);
            imagesc(squareform(model_rdm));
            title(sprintf('RDM iT=%d', iT));
        end
        sgtitle('Model RDMs across iT (iter 1)');
    end
    
    %%  correlation-based dRSA
    if strcmp(opt.dRSA.corrMethod,'corr')
        
        iMod = 0;
        for iModel = opt.modelVec
            iMod = iMod+1;
            dRSA(i,:,:,iMod) = corr(mRDMs{iModel},nRDMs); % dRSA; x:model, y:neural
        end
        
                 if isfield(opt,'autocorr') && opt.autocorr==1
                     % model autocorrelations
                    for iModel = 1:numel(mRDMs)
                        mRSA(i,:,:,iModel) = corr(mRDMs{iModel}, mRDMs{iModel}); % model x model
                    end
        
                    % model cross-correlations
                    if numel(mRDMs) > 1
                        allCombos = nchoosek(1:numel(mRDMs),2);
                        for iCombs = 1:size(allCombos,1)
                            crossRSA(i,:,:,iCombs) = corr(mRDMs{allCombos(iCombs,1)}, mRDMs{allCombos(iCombs,2)}); % model1 x model2
                        end
                    end
                    % neural autocorrelations
                    nRSA(i,:,:) = corr(nRDMs, nRDMs); % neural x neural
                 end
        
        %% PCA based dRSA
    elseif strcmp(opt.dRSA.corrMethod,'PCR')
        %% Prepare models
        
        for iModel = opt.allModels
             if strcmp(opt.dRSA.Normalize,'Standardize')
                tempmodel = mRDMs{iModel};
                %center the model
                tempmodel = tempmodel - repmat(mean(tempmodel,'omitnan'),size(tempmodel,1),1);
                %standardize the model
                mRDMs{iModel} = tempmodel ./ std(tempmodel(:));
                
            elseif strcmp(opt.dRSA.Normalize,'Rescale')
                % rescale all RDMs to same [0 2] interval.
                % Unscaled might be problematic for PCA or regression (i.e., larger scale = more variance = higher component)
                ResizedModel = reshape(mRDMs{iModel}, size(mRDMs{iModel}, 1)* size(mRDMs{iModel}, 2), 1); %reshape to a vector
                RescaledModel = rescale(ResizedModel, 0, 1 );  %rescale
                
                TempModel = reshape(RescaledModel, size(mRDMs{iModel}, 1), size(mRDMs{iModel}, 2));%put it back into the shape we need
                           
                mRDMs{iModel} =   TempModel-repmat(nanmean(TempModel),size(TempModel,1),1);
             end  
        end
        
        
        if strcmp(opt.dRSA.Normalize,'Standardize')
            %center neural model: RDMs need to be centered per individual time point for PCA and regression
            nRDMs = nRDMs - repmat(nanmean(nRDMs, 1), size(nRDMs, 1), 1);
            %standardize
            nRDMs = nRDMs ./ std(nRDMs(:));
        elseif strcmp(opt.dRSA.Normalize,'Rescale')
            ResizedModel = reshape(nRDMs, size(nRDMs, 1)* size(nRDMs, 2), 1); %reshape to a vector
            RescaledModel = rescale(ResizedModel, 0, 1);  %rescale
            nRDMs = reshape(RescaledModel, size(nRDMs, 1), size(nRDMs, 2));%put it back into the shape we need
            % neuralRDM is already rescaled above across the whole time range, but it still needs to be centered per individual time point
            nRDMs = nRDMs - repmat(nanmean(nRDMs),size(nRDMs,1),1);
        end
        
        
        %% regression boarders
        regborder = opt.dRSA.regborder;
        
        
        %% dRSA: loop through (1) each model and (2) each time point
        dRSA_Iter = zeros(size(mRDMs{1}, 2), size(nRDMs, 2), numel(opt.modelVec));  %all models should have the same size, so we just pick the first
        
        
        iMod = 0;
        %Loop through all Models to test
        for iModel = opt.modelVec
            iMod = iMod+1;
            fprintf('\n Model: %04d ',iModel);
            
            %Load models to be excluded
            models2regressout = opt.models2regressout{iModel};
            
            % Loop through all time points
            for iT = 1:opt.SubSampleDur
                
                % define our current model
                XTest = mRDMs{iModel}(:,iT ); %our Model, all features, at time point iT
                
                %recenter it (but not necessary, because done automatically by PCA function)
                %XTest = XTest - nanmean(XTest, 1);
                
                
                %%  1) Autocorrelation based on Regression Borders
                if isnan(regborder(iModel)) == 0
                    %LeftSide = iT - opt.AverageTime/opt.sampleDur : iT - opt.AverageTime/opt.sampleDur - regborder_NewTime(iModel) ;
                    
                    LeftSide = iT - opt.AverageTime/opt.sampleDur : iT - opt.AverageTime/opt.sampleDur + regborder(iModel) ;
                    
                    RightSide = iT + opt.AverageTime/opt.sampleDur - regborder(iModel) : iT + opt.AverageTime/opt.sampleDur;
                    %LeftSide = iT - opt.AverageTime/opt.sampleDur : iT - regborder_NewTime(iModel);
                    Autocorrelation_indexes = [LeftSide  RightSide];
                    Autocorrelation_indexes(logical((Autocorrelation_indexes<1) + (Autocorrelation_indexes > opt.SubSampleDur ))) = [];
                else
                    Autocorrelation_indexes = [];  %if we don't have a border, better to not regress out
                end
                
                if isnan(regborder(iModel)) == 0
                    xAutocorrelation = mRDMs{iModel}(:,Autocorrelation_indexes );  %our model to test, at time points to regress out
                    
                    if opt.dRSA.Additional_PCA  %if we have large models, we perform a second PCA
                        [~, score, ~, ~, exp, ~] = pca(xAutocorrelation);
                        imax = sum(exp>.1); %only the first with a high explained variance
                        xAutocorrelation  = score(:,1:imax); %reduce only to most important components
                    end
                    
                else
                    xAutocorrelation = [];
                end
                
                
                %% 2)  Regressing out the other models at different time points
                
                % for averaging across time we need the indexes:
                regressout_indexes = iT - opt.AverageTime/opt.sampleDur : iT + opt.AverageTime/opt.sampleDur;
                
                %Delete indexes outside
                regressout_indexes(logical((regressout_indexes<1) + (regressout_indexes > opt.SubSampleDur ))) = [];
                
                
                %prepare matrix for PCA
                xModelRegressout = zeros (size(mRDMs{iModel},1), 5000); % fill it with our data
                
                %for loop for models to regress out
                for iReg = models2regressout
                    Regressmodel = mRDMs{iReg}(:, regressout_indexes);
                    
                    if opt.dRSA.Additional_PCA
                        [~, score, ~, ~, exp, ~] = pca(Regressmodel);
                        imax = sum(exp>.1);% only components with minimum variance of X%
                        Regressmodel = score(:,1:imax);
                    end
                    
                    Index_NewData = nnz(xModelRegressout(1,:)); %Index where to add our new data to
                    xModelRegressout(:, Index_NewData + 1 : Index_NewData + size(Regressmodel, 2) ) = Regressmodel;
                end
                
                %delete the 0s
                NotZeros = nnz(xModelRegressout(1,:));
                xModelRegressout = xModelRegressout(:,1:NotZeros);
                xRegressout = [xModelRegressout xAutocorrelation];
                
                
                %clean workspace
                
                if opt.dRSA.Additional_PCA
                    if strcmp(opt.dRSA.Normalize,'Standardize')
                        %Standardize by dividing through sd
                        xRegressout = xRegressout/std(xRegressout(:));
                    elseif strcmp(opt.dRSA.Normalize,'Rescale')
                        ResizedModel = reshape(xRegressout, size(xRegressout, 1)* size(xRegressout, 2), 1); %reshape to a vector
                        RescaledModel = rescale(ResizedModel, 0, 1);  %rescale
                        xRegressout = reshape(RescaledModel, size(xRegressout, 1), size(xRegressout, 2));%put it back into the shape we need
                        xRegressout = xRegressout-repmat(nanmean(xRegressout),size(xRegressout,1),1);
                    end
                    
                end
                
                clearvars NotZeros Index_NewData Regressmodel regressout_indexes LeftSide RightSide Autocorrelation_indexes ResizedModel RescaledModel;
                
                
                Xx = [XTest xRegressout];  %put it together
                
                
                %to get the minimum amount of features
                [nFeature, nPredictors] = size(Xx);
                MinComp = min(nFeature,nPredictors);
                
                %% Principal Component Regression (PCR)
                % a) First we run a PCA is run on X, resulting in the PCAscores (components)
                
                
                if strcmp(opt.dRSA.PCRMethod,'FixedComp')
                    [PCALoadings,PCAScores,~,~,explained] = pca(Xx,'NumComponents',opt.nPCAcombs);
                                        
                elseif strcmp(opt.dRSA.PCRMethod,'MinCompPCR')
                    MinComp = floor(opt.percPCA*MinComp);
                    [PCALoadings,PCAScores,~,~,explained] = pca(Xx,'NumComponents',MinComp);
                                       
                elseif strcmp(opt.dRSA.PCRMethod,'ExplainedVar')
                    [PCALoadings,PCAScores,~,~,explained] = pca(Xx);
                    imax = sum(explained>.1); %only the first with a high explained variance
                    
                    if imax >= nFeature
                        error('More Predictors than observations.  This will lead to overfitting.')
                    end
                                        
                    PCAScores  = PCAScores(:,1:imax); %reduce only to most important components
                    PCALoadings = PCALoadings(1,1:imax);
                   
                elseif strcmp(opt.dRSA.PCRMethod,'CumulativeVar')
                    [PCALoadings,PCAScores,~,~,explained] = pca(Xx);
                    threshold = opt.nPCACumVar *  sum(explained);
                    imax = find(cumsum(explained) >= threshold, 1);
                    
                    
                    if imax >= nFeature
                        error('More Predictors than observations.  This will lead to overfitting.')
                    end
                    
                    PCAScores  = PCAScores(:,1:imax); %reduce only to most important components
                    PCALoadings = PCALoadings(1,1:imax);
       
                end
               
                
                % b) We use these scores as  predictor variables in a least-squares regression, with our neural RDM as response variable
                betaPCR = PCAScores\(nRDMs);  % The \ operator performs a least-squares regression to calculate the slopes or regression coefficient
                
                
                % c) the principal component regression weights (betaPCR) are projected back onto the original variable space using the PCA loadings,
                %    to extract a single regression weight corresponding to the original X
                temporarydRSA = PCALoadings*betaPCR;
                
                
                %d) Select the first weight, which corresponds to XTest. The others represent xRegressout and xAutocorrelation
                dRSA_Iter(iT, :, iMod) = temporarydRSA(1,:)';
                
            end % end of Time Loop
        end %end of Model Loop
        
        dRSA(i,:,:,:) = dRSA_Iter;
    end %end of if loop
    

end  %end of Iterarions

nRSA = mean(nRSA, 1); 
mRSA = mean(mRSA, 1);
dRSA = mean(dRSA, 1); % 1 = Fmean across first dim
%we can not use squeeze, in case we only have 1 model we would loose the fourth dimension
dRSA = reshape(dRSA, size(dRSA,2), size(dRSA,3), size(dRSA,4));
mRSA = reshape(mRSA, size(mRSA,2), size(mRSA,3), size(mRSA,4));
nRSA = reshape(nRSA, size(nRSA,2), size(nRSA,3), size(nRSA,4));






