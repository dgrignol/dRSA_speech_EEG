function MD_run_dRSA_v4 (cfg, opt, iSub)



close all;
clc;

%Paths
globalCfg = MD_function.config();

if isfield(cfg, 'rootdir') && ~isempty(cfg.rootdir)
    rootdir = cfg.rootdir;
elseif isfield(cfg, 'repoBase') && ~isempty(cfg.repoBase)
    rootdir = cfg.repoBase;
else
    error('MD_run_dRSA_v4:MissingRootDir', ...
          'Provide cfg.rootdir or cfg.repoBase before calling MD_run_dRSA_v4.');
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
    error('MD_run_dRSA_v4:InvalidToolboxPaths', ...
          'cfg.toolboxPaths must be a char, string, or cell array of char.');
end

toolboxPaths = cellfun(@char, toolboxPaths, 'UniformOutput', false);
for iTool = 1:numel(toolboxPaths)
    toolDir = toolboxPaths{iTool};
    if exist(toolDir, 'dir')
        addpath(toolDir);
    else
        warning('MD_run_dRSA_v4:MissingToolbox', ...
                'Toolbox directory not found and skipped: %s', toolDir);
    end
end

saveDir = fullfile(rootdir, 'dRSA', 'dRSAData', 'Sensor');
if ~isfield(cfg, 'SubSampleDir') || isempty(cfg.SubSampleDir)
    cfg.SubSampleDir = fullfile(rootdir, 'dRSA', 'SubSamples');
end
if ~isfield(cfg, 'RegressionBorderDir') || isempty(cfg.RegressionBorderDir)
    cfg.RegressionBorderDir = fullfile(rootdir, 'dRSA', 'RegressionBorders');
end


%Defining Variables for building cDRSA
[opt, cfg] = MD_LoadParameters (opt, cfg); %Load fixed Parameters
[opt, cfg] = MD_RemapModels (opt, cfg);  %Change Parameters so that they fit to Modelnumber



% Convert opt.allModels to a string with underscores instead of spaces
modelsStr = strrep(mat2str(cfg.oldModels), ' ', '-');
modelsStr = modelsStr(2:end-1); % Remove the square brackets



%% neural data
if cfg.badSeg == 1 && cfg.ica == 1
    fn = sprintf('%s/preproc_data/SUB%02d/BadSeg-ICA-preproc-data-final-%dhz-sub%02d_%s.mat', rootdir, iSub, globalCfg.resamplefs100, iSub, cfg.ROInames{cfg.ROINumber });
elseif cfg.badSeg == 0 && cfg.ica == 1
    fn = sprintf('%s/preproc_data/SUB%02d/NoBadSeg-ICA-preproc-data-final-%dhz-sub%02d_%s.mat', rootdir, iSub, globalCfg.resamplefs100, iSub, cfg.ROInames{cfg.ROINumber });
elseif cfg.badSeg == 1 && cfg.ica == 0
    fn = sprintf('%s/preproc_data/SUB%02d/BadSeg-NoICA-preproc-data-final-%dhz-sub%02d_%s.mat', rootdir, iSub, globalCfg.resamplefs100, iSub, cfg.ROInames{cfg.ROINumber });
elseif cfg.badSeg == 0 && cfg.ica == 0
    fn = sprintf('%s/preproc_data/SUB%02d/NoBadSeg-NoICA-preproc-data-final-%dhz-sub%02d_%s.mat', rootdir, iSub, globalCfg.resamplefs100, iSub, cfg.ROInames{cfg.ROINumber });
end
load(fn);

cDRSA = zeros(opt.SubSampleDur, opt.SubSampleDur, length(cfg.modelVec), globalCfg.nCond);  %all models should have the same size, so we just pick the first

for iCon = cfg.condVec
    clearvars Y
    
    % find predictability level
    selectedCond = iCon; % predibility: 0,9,18,30 sigma
    condVec = selectedCond:4:16;
    
    % re-organize into required format
    for i = 1:numel(condVec)
        Y(i, 1, :, :) = data_final.trial{condVec(i)}; % events, (repetitions/runs), features, time points
    end
    
    %Load the model RDMs
    [models] = MD_LoadModelRDMs (iSub, iCon, cfg, opt);
        
   
    %% Regression Border
    
    if strcmp(opt.dRSA.corrMethod,'PCR')
        
        %check if I have all regborders & autocorrelation
       regborder = MD_LoadRegborders (models, opt, cfg, iSub, iCon);
               
        opt.dRSA.regborder = regborder;
         methodname =  MD_createPCR_nameString (opt);
         
    else
         methodname = '';

    end
    
    
    % %run dRSA
%     tStart = tic;           % pair 2: tic

    [mDRSA] = dRSA_coreFunction(Y,models,opt,cfg);
    
    %to measure the time
%     tEnd = toc(tStart);      % pair 2: toc
%     timename = fullfile(saveDir, sprintf('%s_dRSA_sub%02d_%02d-features_%02ds_%02d-iter_%.2f-Var_Sensor_%s_TIME_dRSA.mat', opt.dRSA.corrMethod, iSub, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, cfg.ROInames{cfg.ROINumber}));
%     
%     save(timename,'tEnd');
%     
%     
    
    cDRSA(:,:,:,iCon) =  mDRSA;  %save based on condition
    
    filename = fullfile(saveDir, sprintf('%s_dRSA_sub%02d_cond%02d_%02d-features_%ds_%02d-iter_%.2f-Var_%s_%s_Sensor_%s.mat', ...
        opt.dRSA.corrMethod, iSub, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, modelsStr, methodname, cfg.ROInames{cfg.ROINumber }));
    
    if cfg.saving
        save(filename,'mDRSA');
    end
    
end

filename = fullfile(saveDir, sprintf('%s_dRSA_sub%02d_%02d-features_%02ds_%02d-iter_%.2f-Var_%s_%s_Sensor_%s.mat',...
    opt.dRSA.corrMethod, iSub, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, modelsStr, methodname, cfg.ROInames{cfg.ROINumber}));
if cfg.saving
    save(filename,'cDRSA');
end






end % end of function
