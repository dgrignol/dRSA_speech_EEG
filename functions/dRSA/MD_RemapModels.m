function [opt, cfg] = MD_RemapModels (opt, cfg)

% Input:
% cfg.modelVec = [1 3 5];
% cfg.modelRegressout = [1 3 4 5];
% cfg.Modelplot ;


%Output:
% opt.distanceMeasureModel    = {'euclidean', 'euclidean', 'difference', 'interaction (1,3)', 'euclidean'}; --> also numbers change!!
% opt.AllLabels  --> adapt so that works
% opt.allModels
% opt.modelVec
% opt.modelOut // models2regressout


Modelsused = unique([cfg.modelVec, cfg.modelRegressout]);
ModelsNewNum = 1:length(Modelsused);


% Generate a mapping that goes from original model numbers to their new indices
% This will map the original model indices
modelMapping = containers.Map(Modelsused, ModelsNewNum) ;


%Adjust numbers by using temp structs that can be changed later
Modelplot_temp = [];

if isfield(opt, 'SimulatedModel')
    newvar = modelMapping(opt.SimulatedModel);
    opt.SimulatedModel = newvar;
end


if isfield (cfg, 'Modelplot')
    cfg.ModelplotOld = cfg.Modelplot;
    newvar = arrayfun(@(k) modelMapping(k), cfg.Modelplot);
    cfg.Modelplot = newvar;
end

for i = ModelsNewNum
    
    %get original number
    OriginalNumb = Modelsused(i);
    
    %opt.distanceMeasureModel
    distanceMeasureModel_temp(i) =    opt.distanceMeasureModel (OriginalNumb);
    
    if isfield(cfg, 'Modelplot') && ismember(OriginalNumb, cfg.Modelplot)
        Modelplot_temp = [Modelplot_temp, modelMapping(OriginalNumb)];
    end
    
    
    
    %change also numbers within interaction
    if contains(distanceMeasureModel_temp(i), 'interaction')
        str = distanceMeasureModel_temp(i);
        numbers = regexp(str, 'interaction \((\d+),(\d+)\)', 'tokens');
        if ~isempty(numbers)
            extractedNumbers = str2double(numbers{1}{:});
            extractedNumbers_new(1) = modelMapping(extractedNumbers(1));
            extractedNumbers_new(2) = modelMapping(extractedNumbers(2));
            updatedStr = sprintf('interaction (%d,%d)', extractedNumbers_new(1), extractedNumbers_new(2));
            distanceMeasureModel_temp{i} = updatedStr;
        end
    end
    
    %opt.AllLabels
    Label_Temp(i) = opt.AllLabels(OriginalNumb);
    
    %opt.modelVec
    if ismember(OriginalNumb, cfg.modelVec)
        modelVec_temp(i) = i;
    else
        modelVec_temp(i) = NaN;
    end
    
    if strcmp(opt.dRSA.corrMethod,'PCR')
        
        %opt.models2regressout
        originalModels = opt.models2regressout{OriginalNumb};
        newModels = nan(size(originalModels));
        
        % Check if each model is present in the mapping and update
        for j = 1:length(originalModels)
            if isKey(modelMapping, originalModels(j))
                newModels(j) = modelMapping(originalModels(j));
            else
                % Optionally handle missing keys, e.g., remove or set to NaN
                newModels(j) = nan; % This sets missing models to NaN
            end
        end
        
        
        newModels = newModels(~isnan(newModels));
        modelsout {i} = newModels;
    end
end

if strcmp(opt.dRSA.corrMethod,'PCR')
    
    opt.models2regressout = modelsout;
end

%    if isfield(cfg, 'Modelplot')
%        cfg.Modelplot_new = Modelplot_temp ;
%     end



%change original
opt.AllLabels  = Label_Temp;
opt.distanceMeasureModel = distanceMeasureModel_temp;
opt.allModels = ModelsNewNum;
opt.modelVec = modelVec_temp(~isnan(modelVec_temp));

cfg.oldModels = Modelsused;

%Names
cfg.Name.modelsStr = strrep(mat2str(cfg.oldModels), ' ', '-');
cfg.Name.modelsStr = cfg.Name.modelsStr(2:end-1); % Remove the square brackets

end