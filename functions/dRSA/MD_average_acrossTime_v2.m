function MD_average_acrossTime_v2(cfg, opt)

%% Paremeteeers

[opt, cfg] = MD_LoadParameters (opt, cfg);
[opt, cfg] = MD_RemapModels (opt, cfg);  %Change Parameters so that they fit to Modelnumber

saveDir = opt.saveDir;


tRange = opt.AverageTime * (1/opt.sampleDur); % How many data points fit in the time span?
numModels = length(cfg.modelVec);
TimeVec = [-opt.AverageTime:opt.sampleDur:opt.AverageTime];  %Time Vector used for averaging


dRSA_allCon = cell(1,length(cfg.condVec));
dRSA_allCon_sd  = cell(1,length(cfg.condVec));



%% Load & Prepare
for iCon = cfg.condVec

    if cfg.sensor  == 0 %source
        
         filename = fullfile(saveDir, sprintf('Source/%s_dRSA_allSub_con%02d_%02d-features_%ds_%02d-iter_%.2f-Var_%s_%s_SourceSpace_atlas-%s_ROI-%s_hm-%s.mat', ...
                opt.dRSA.corrMethod, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, cfg.Name.modelsStr,cfg.Name.methodname, cfg.atlas, cfg.ROIold, cfg.side));

         savename = fullfile(saveDir, sprintf('Source/%s_dRSA_allSub_averagedtime_con%02d_%02d-features_%ds_%02d-iter_%.2f-Var_%s_%s_SourceSpace_atlas-%s_ROI-%s_hm-%s.mat', ...
                opt.dRSA.corrMethod, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, cfg.Name.modelsStr,cfg.Name.methodname, cfg.atlas, cfg.ROIold, cfg.side));
        
       
        
    elseif cfg.sensor  == 1  % sensor
       
        filename = fullfile(saveDir, sprintf('Sensor/%s_dRSA_allSub_cond%02d_%02d-features_%ds_%02d-iter_%.2f-Var_%s_%s_Sensor_%s.mat', ...
                opt.dRSA.corrMethod, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, cfg.Name.modelsStr, cfg.Name.methodname, cfg.ROInames{cfg.ROINumber }));

        savename = fullfile(saveDir, sprintf('Sensor/%s_dRSA_allSub_averagedtime_cond%02d_%02d-features_%ds_%02d-iter_%.2f-Var_%s_%s_Sensor_%s.mat', ...
                opt.dRSA.corrMethod, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, cfg.Name.modelsStr, cfg.Name.methodname, cfg.ROInames{cfg.ROINumber }));
          
    end
    
       
    load (filename);

    nsub =  size(dRSA_SUB_Con,1);
    numTimePoints = size(dRSA_SUB_Con,2);

    rstack = zeros(nsub, numModels, numTimePoints, tRange*2+1);
    %dRSA_averaged_con =   zeros(nsub, numModels, numTimePoints);
    clearvars dRSA_averaged_con  dRSA_temp1 dRSA_temp2 dRSA_temp3

    iMod = 0;
    for iModel = cfg.modelVec
        iMod = iMod+1;
        for iModelTime = 1:numTimePoints
            timeindex = iModelTime - tRange:iModelTime + tRange;  % Get the index of our time window
            OutsideSample = logical((timeindex < 1) + (timeindex > numTimePoints));  % Outside of our subsample
            timeindex(OutsideSample) = 1; % Remove indices that are before or after video

            slice = (dRSA_SUB_Con(:, iModelTime, timeindex, iMod)); % Slice of our time over which we average
            slice = reshape(slice, size(slice, 1), []);  % Reshape to N x 401 (or 401 x 1 if there's only 1 participant)
            slice( :, OutsideSample) = NaN; % Remove indices that are outside our sample

            rstack(:, iMod, iModelTime, :) = slice;
        end
    end

    dRSA_temp1 = nanmean(rstack,3);
    dRSA_temp2 =  reshape(dRSA_temp1, [size(dRSA_temp1,1), length(cfg.modelVec), size(dRSA_temp1,4)]);

    dRSA_data = dRSA_temp2;
    save(savename, 'dRSA_data');
    clearvars dRSA_data;

end


end






