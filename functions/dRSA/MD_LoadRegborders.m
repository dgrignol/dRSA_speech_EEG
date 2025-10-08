function regborder = MD_LoadRegborders (models, opt, cfg, iSub, iCon)

Regborder = @(iModel) fullfile(cfg.RegressionBorderDir, sprintf('Regborders_sub%02d_cond%02d_model-%s_%02d-features_%02d-iter_%.2f-Var.mat', iSub, iCon, opt.AllLabels{iModel}, opt.nSubSamples, 1000, opt.Var));
AutoCor_Name = @(iModel) fullfile(cfg.RegressionBorderDir, sprintf('Autocorrelation_sub%02d_cond%02d_model-%s_%02d-features_%02d-iter_%.2f-Var.mat', iSub, iCon, opt.AllLabels{iModel}, opt.nSubSamples, 1000, opt.Var));

% Check for existence of each file in opt.modelVec without an explicit loop
RegborderExistCheck = arrayfun(@(iModel) exist(Regborder(iModel), 'file') == 2, opt.allModels);
AutoCorExistCheck = arrayfun(@(iModel) exist(AutoCor_Name(iModel), 'file') == 2, opt.allModels);


if any(~RegborderExistCheck) | (~AutoCorExistCheck)
    dRSA_border  (models, opt, cfg, iSub, iCon);
end


%put the regborder together by loading the files
regborder = NaN(1, length(opt.allModels));  % Use NaN as placeholder

% Load the files that exist and store them in regborder(iModel)
for iModel = opt.allModels
    tempData = load(Regborder(iModel));  % Load the file
    regborder(iModel) = tempData.regborder_model';  % Store the loaded data in regborder(iModel)
end



end
