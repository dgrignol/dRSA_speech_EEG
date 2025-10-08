function dRSA_border (models, opt, cfg, iSub, iCon)

fprintf('Calculate autocorrelation borders for < %d %% of variance \n', opt.Regressionborder*100)

%%  *********************************Autocorrelation model by model
mRDMs=[];


for iModel = opt.allModels
    
    fprintf('\n New Model: %d \n', iModel);

    %% First load input variables:
    
    origDims    = size(models.data{(iModel)});
    nEvents     = origDims(1);
    nFeatures   = origDims(2);
    nTPevents   = origDims(3);
    
    
    %% concatenate events
    
    if nEvents>1
        disp('concatenate events in models');
        M = shiftdim(models.data{iModel},2); % transpose to TP x events x features
        models.data{iModel} = reshape(M,nTPevents*nEvents,nFeatures)';
    end
    
    nTP = size(models.data{iModel},2);
fprintf('total number of TPs: %d \n', nTP);
    
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
    
    
    mask(1+iModel,:) = any(isnan(models.data{(iModel)}));
    maskLabels{1+iModel} = sprintf('NaNs in model (%s)', models.labels{(iModel)});
    
    maskSubsampling = any(mask); % turn into a 1D mask that combines all individual masks
    maskSubsamplingStarts = any(mask); % and another mask that is only used to find random start positions of the subsamples
    
    
    % check if enough samples for planned sequences
    TotalDurSeqS = opt.SubSampleDur*opt.nSubSamples + opt.spacing*(opt.nSubSamples-1) + sum(maskSubsampling==1);
    if TotalDurSeqS > nTP
        disp('too many subsamples/ subsamples too long');
        return;
    end
    
    
    % make a mask that also considers the duration of subsamples, so that subsamples do not overlap with masked time points:
    idxSubsampleStart = (repmat(find(maskSubsampling),opt.SubSampleDur,1) + (-opt.SubSampleDur:-1)')';
    idxSubsampleStart(idxSubsampleStart<1)=[];
    maskSubsamplingStarts(idxSubsampleStart(:)) = 1;
    
    
    %% Start with the Simulations (= Subsampling)
    cfg.sampling='subsampling';
    
    fn = fullfile(cfg.SubSampleDir, sprintf('Subsamples_%d-Dur_%d-nSubSamples_%d-nIter.mat', opt.SubSampleDurSec, opt.nSubSamples, 1000));
    
    if exist(fn) == 2
        load(fn);
    else
        opt.nIter = 1000;
        [SSIndices SSIndicesPlot] = dRSA_subsampling_v3(maskSubsampling, opt); % SS = subsamples
        save(fn,'SSIndices');
        fl = fullfile(cfg.SubSampleDir, sprintf('SubSamplesPlot_%d-Dur_%d-nSubSamples_%d-nIter.mat', opt.SubSampleDurSec, opt.nSubSamples, 1000));
        save(fl,'SSIndicesPlot');
    end
    fprintf('\nrun Border:      ')


    %% Autocorrelation
    for iSim = 1:1000
    fprintf('\b\b\b\b\b%04d ', iSim);  
    
        for iTime = 1:opt.SubSampleDur
            if  strcmp(opt.distanceMeasureModel{iModel}, 'difference')
                mRDMs{iModel}(:,iTime) = pdist(models.data{(iModel)}(:,SSIndices(:,iTime,iSim))',@calculateAngleDistance);
                
            elseif contains(opt.distanceMeasureModel{iModel}, 'interaction')
                str = opt.distanceMeasureModel{iModel};
                numbers = regexp(str, 'interaction \((\d+),(\d+)\)', 'tokens');
                if ~isempty(numbers)
                    extractedNumbers = str2double(numbers{1});
                else
                    disp('No match found.');
                    return;
                end
                
                clearvars model1 model2
                model1 = rescale( mRDMs{extractedNumbers(1)} (:,iTime), 1, 2);
                model2 = rescale( mRDMs{extractedNumbers(2)} (:,iTime), 1, 2);
                
                mRDMs{iModel}(:,iTime) = model1.*model2;
            else
                mRDMs{iModel}(:,iTime) = pdist(models.data{(iModel)}(:,SSIndices(:,iTime,iSim))',opt.distanceMeasureModel{iModel});
            end
        end
        
        %
        
        % model autocorrelations
        mRSA(iSim,:,:,iModel) = corr(mRDMs{iModel}, mRDMs{iModel}); % model x model
    end
    
end  %end of iModel loop


%% Averaging over a Time Window

%average across Simulations
mRSA_averaged = nanmean(mRSA, 1);

%we can not use squeeze, in case we only have 1 model we would loose the fourth dimension
mRSA_averaged = reshape(mRSA_averaged, size(mRSA_averaged,2), size(mRSA_averaged,3), size(mRSA_averaged,4));



% create the new time vector:
tRange = opt.SubSampleDur/2;

rstack = zeros(size(mRSA_averaged,3), size(mRSA_averaged,1), tRange*2+1); %saving the sliding window of time
Averaged_Autocorr = zeros(size(mRSA_averaged,3), size(mRSA_averaged,1));   %saving the autocorrelation averaged acriss time, n Models and m DataPoints


%sliding window over time
for jModel = 1:size(mRSA_averaged,3)  %how many models?
    for iModelTime = 1:size(mRSA_averaged,1)  %first time points
        timeindex = iModelTime - tRange:iModelTime + tRange;
        OutsideSample = logical((timeindex < 1)+(timeindex > size(mRSA_averaged,1)));  %outside of our Subsample
        timeindex(OutsideSample) = 1;%remove indices that fall before or after video
        
        slice = squeeze(mRSA_averaged(iModelTime, timeindex, jModel)); %slice of our time over which we average
        slice(OutsideSample) = NaN;%remove indices that fall outside our sample
        
        rstack(jModel, iModelTime, :) = slice;
    end
end

% % average across time:
Averaged_Autocorr = squeeze(nanmean(rstack,2));




%% defining the Border

regborder = 0; %zeros(size(mRSA_averaged,3)); %our regression borders

%finding the 0 lag of our averaged data:
TimVec = [-(size(Averaged_Autocorr,2)-1)*opt.sampleDur / 2:opt.sampleDur:(size(Averaged_Autocorr,2)-1)*opt.sampleDur / 2];  %Time Vector used for averaging

Lag0 = dsearchn(TimVec',0);


for iModel = 1:size(Averaged_Autocorr,1)  %how many models?
    peak = max(Averaged_Autocorr(iModel,:));  %the highest autocorrelation for this model
    
    %Define the left and right side of the peak
    LeftSide = squeeze(Averaged_Autocorr(iModel,1:Lag0));
    RightSide = squeeze(Averaged_Autocorr(iModel,Lag0:end));
    
    %find the regression borders where less than xx% of variance has been explained
    LeftBorder = find((LeftSide) < opt.Regressionborder*peak,1, 'last');  %flip to find the first from the left side
    RightBorder = length(RightSide) - find((RightSide) < opt.Regressionborder*peak, 1, 'first');
    
    regborder(iModel,:) = ceil(nanmean([LeftBorder RightBorder ])); %take the average
    
    if  isnan(regborder(iModel,:))
        fprintf('ERROR: For Model "%s" the regression border could not be calculated! \n', models.labels{iModel});
    end
    
end



%for figure later
figuredir = sprintf('%sfigures/', cfg.RegressionBorderDir);
mkdir(figuredir);




for iModel = 1:size(Averaged_Autocorr,1)  %how many models?
    
    %% Plot Autocorrelation
    figure('Visible', 'off');
    
    
    % hold on
    plot(TimVec, Averaged_Autocorr(iModel,:))
    title(sprintf('Regression Borders: %s',models.labels{iModel}));
    xlabel('Time [s]');
    ylabel('Autocorrelation');
    set(gca,'ylim',[-0.1 1]);
    Title_Name = sprintf('S%02d Con%d Mod %s', iSub, iCon, models.labels{iModel} );
    title(Title_Name, 'FontSize', 8);
    if  ~isnan(regborder(iModel,:))
        
        leftborder = TimVec(regborder(iModel));
        rightborder = TimVec(end-  regborder(iModel) + 1);
        
        xline([leftborder], '--r', {'Left Border'});
        xline([rightborder], '--r', {'Right Border'});
    end
    %
    hold off
    
    outputFilename = sprintf('RegressionBorders_Sub%d_Con%d_Model-%s.png', iSub, iCon, opt.AllLabels{iModel});
    saveas(gcf, fullfile(figuredir, outputFilename)); % Change to your save directory
    
    
    close(gcf);
    
    regborder_model = regborder(iModel);
    AutoCor_Name_model = Averaged_Autocorr(iModel);
    
    %% save data separately from Models
    fn = fullfile(cfg.RegressionBorderDir, sprintf('Regborders_sub%02d_cond%02d_model-%s_%02d-features_%02d-iter_%.2f-Var.mat', iSub, iCon, opt.AllLabels{iModel}, opt.nSubSamples, 1000, opt.Var));
    save(fn,'regborder_model');
    AutoCor_Name = fullfile(cfg.RegressionBorderDir, sprintf('Autocorrelation_sub%02d_cond%02d_model-%s_%02d-features_%02d-iter_%.2f-Var.mat', iSub, iCon, opt.AllLabels{iModel}, opt.nSubSamples, 1000, opt.Var));
    save(AutoCor_Name,'AutoCor_Name_model');
    
    
end  %of iModel



end






