function MD_acrossSubjects(cfg, opt)




%% Load & Change Parameters
[opt, cfg] = MD_LoadParameters (opt, cfg);
[opt, cfg] = MD_RemapModels (opt, cfg);  %Change Parameters so that they fit to Modelnumber

%iROI = 1;
globalCfg = MD_function.config();
if isfield(cfg, 'rootdir') && ~isempty(cfg.rootdir)
    rootdir = cfg.rootdir;
elseif isfield(cfg, 'repoBase') && ~isempty(cfg.repoBase)
    rootdir = cfg.repoBase;
else
    error('MD_acrossSubjects:MissingRootDir', ...
          'Provide cfg.rootdir or cfg.repoBase before calling MD_acrossSubjects.');
end

saveDir = fullfile(rootdir, 'dRSA', 'dRSAData');  % Proper path creation


%% Load & Put together

for iCon = cfg.condVec
    dRSA_SUB_Con = zeros(length(cfg.SubVec), opt.SubSampleDurSec*globalCfg.resamplefs100,  opt.SubSampleDurSec*globalCfg.resamplefs100, length(cfg.modelVec));
    VP = 0;
    for iSub = cfg.SubVec
        VP = VP +1;
        clearvars mDRSA
        
        %source vs sensor space
        if cfg.sensor  == 0 %source
            %             filename = fullfile(saveDir, sprintf('Source/%s_dRSA_sub%02d_con%02d_%02d-features_%02ds_%02d-iter_%.2f-Var_SourceSpace_atlas-%s_ROI-%s_hm-%s.mat', opt.dRSA.corrMethod, ...
            %                 iSub, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, cfg.atlas, cfg.Name.ROIstr, cfg.side));
            %
            %             filename = fullfile(saveDir, sprintf('Source/%s_dRSA_sub%02d_con%02d_%02d-features_%02ds_%02d-iter_%.2f-Var_%d-comps_%d-AdditionalPCA_%s_SourceSpace_atlas-%s_ROI-%s_hm-%s.mat', ...
            %             opt.dRSA.corrMethod, iSub, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, opt.nPCAcombs, opt.Additional_PCA, cfg.Name.modelsStr, cfg.atlas, cfg.Name.ROIstr, cfg.side));
            %
            
%             filename = fullfile(saveDir, sprintf('Source/%s_dRSA_sub%02d_con%02d_%02d-features_%02ds_%02d-iter_%.2f-Var_%s_%s_SourceSpace_atlas-%s_ROI-%s_hm-%s.mat', ...
%                 opt.dRSA.corrMethod, iSub, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, cfg.Name.modelsStr,cfg.Name.methodname, cfg.atlas, cfg.Name.ROIstr, cfg.side));
%             

filename = fullfile(saveDir, sprintf('Source/%s_dRSA_sub%02d_con%02d_%02d-features_%02ds_%02d-iter_%.2f-Var_%s_%s_SourceSpace_atlas-%s_ROI-%s_hm-%s.mat', ...
        opt.dRSA.corrMethod, iSub, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, cfg.Name.modelsStr,cfg.Name.methodname, cfg.atlas, cfg.ROIold, cfg.side));

            
            savename = fullfile(saveDir, sprintf('Source/%s_dRSA_allSub_con%02d_%02d-features_%ds_%02d-iter_%.2f-Var_%s_%s_SourceSpace_atlas-%s_ROI-%s_hm-%s.mat', ...
                opt.dRSA.corrMethod, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, cfg.Name.modelsStr,cfg.Name.methodname, cfg.atlas, cfg.ROIold, cfg.side));
            
            
            
        elseif cfg.sensor  == 1  % sensor
            % Define filename template outside the loop
            %
            %             filename = fullfile(saveDir, sprintf('Sensor/%s_dRSA_sub%02d_cond%02d_%02d-features_%ds_%02d-iter_%.2f-Var_%s_Sensor_%s.mat', ...
            %                 opt.dRSA.corrMethod, iSub, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, cfg.Name.modelsStr, cfg.ROInames{cfg.ROINumber }));
            %          	filename = fullfile(saveDir, sprintf('Sensor/%s_dRSA_sub%02d_cond%02d_%d-features_%ds_%d-iter_%.2f-Var_%d-comps_%d-AdditionalPCA_%s_Sensor_%s.mat', ...
            %         opt.dRSA.corrMethod, iSub, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, opt.nPCAcombs, opt.Additional_PCA, cfg.Name.modelsStr, cfg.ROInames{cfg.ROINumber }));
            
            
            filename = fullfile(saveDir, sprintf('Sensor/%s_dRSA_sub%02d_cond%02d_%02d-features_%ds_%02d-iter_%.2f-Var_%s_%s_Sensor_%s.mat', ...
        opt.dRSA.corrMethod, iSub, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, cfg.Name.modelsStr, cfg.Name.methodname, cfg.ROInames{cfg.ROINumber }));
                
            savename = fullfile(saveDir, sprintf('Sensor/%s_dRSA_allSub_cond%02d_%02d-features_%ds_%02d-iter_%.2f-Var_%s_%s_Sensor_%s.mat', ...
                opt.dRSA.corrMethod, iCon, opt.nSubSamples, opt.SubSampleDurSec, opt.nIter, opt.Var, cfg.Name.modelsStr, cfg.Name.methodname, cfg.ROInames{cfg.ROINumber}));
            
        end
        
        load(filename);
        
        dRSA_SUB_Con(VP,:,:,:) = mDRSA;
    end
    
    save(savename, 'dRSA_SUB_Con');
    
end

end
