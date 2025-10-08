function EEG = ICA(EEG, rejection)
% Input: EEG structure; rejection can be ICLabel or manual
% Output: EEG with ICA

        % ICA: on a 1 Hz filtered copy
        EEG_ica = pop_eegfiltnew(EEG, 1, []);
        EEG_ica = pop_runica(EEG_ica, 'icatype', 'runica');

        % Transfer ICA to the original dataset
        EEG.icaact     = EEG_ica.icaact;
        EEG.icawinv    = EEG_ica.icawinv;
        EEG.icasphere  = EEG_ica.icasphere;
        EEG.icaweights = EEG_ica.icaweights;
        EEG.icachansind = EEG_ica.icachansind;

        EEG = iclabel(EEG);
        
%         % Save dataset before component rejection
%         preRejectFile = sprintf('%s_eeg_ICA.set', subjectID);
%         pop_saveset(EEG, 'filename', preRejectFile, 'filepath', targetPath);
%         
        % Save the preprocessed EEG
        [~, nome, ~] = fileparts(EEG.filename);
        EEG.filename = [nome '_ICA.set']; % 
        EEG = pop_saveset(EEG, 'filename', EEG.filename, 'filepath', EEG.filepath, 'savemode', 'onefile');
        
        labels_prob = EEG.etc.ic_classification.ICLabel.classifications;
        artifact_classes = {'Eye', 'Muscle', 'Heart', 'Line Noise', 'Channel Noise'};
        [~, artifact_indices] = ismember(artifact_classes, EEG.etc.ic_classification.ICLabel.classes);
        artifact_indices(artifact_indices == 0) = [];
        comp2reject = find(any(labels_prob(:, artifact_indices) >= 0.90, 2));
        EEG.ICLabelled_comp.comp2reject = comp2reject;
        
        if strcmp(rejection,'ICLabel') 
            if ~isempty(comp2reject)
                
                EEG = pop_subcomp(EEG, comp2reject, 0); % reject ICLabelled components

                EEG.ICLabelled_comp.rejected = true;
                % Save the ICA rej EEG
                [~, nome, ~] = fileparts(EEG.filename);
                EEG.filename = [nome '_rej.set']; % 
                EEG = pop_saveset(EEG, 'filename', EEG.filename, 'filepath', EEG.filepath, 'savemode', 'onefile');
            else
                EEG.ICLabelled_comp.rejected = false;
                % Save the ICA rej EEG
                [~, nome, ~] = fileparts(EEG.filename);
                EEG.filename = [nome '_rej.set']; % 
                EEG = pop_saveset(EEG, 'filename', EEG.filename, 'filepath', EEG.filepath, 'savemode', 'onefile');
      
            end
        else
            
            error('Manual ICA component rejection not implemented yet')
%             % to do
%             EEG.ICLabelled_comp.rejected = false;
% 
%             % Save the ICA rej EEG
%             [~, nome, ~] = fileparts(EEG.filename);
%             EEG.filename = [nome '_rej.set']; % 
%             EEG = pop_saveset(EEG, 'filename', EEG.filename, 'filepath', EEG.filepath, 'savemode', 'onefile');
%    
        end


        
    
end