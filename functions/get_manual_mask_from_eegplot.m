function [maskBadWins] = get_manual_mask_from_eegplot(EEG, maskFolderPath, subjNum)
% GET_MANUAL_MASK_FROM_EEGPLOT - Open eegplot() for manual selection,
% wait until plot is closed with Figure->Accept and close, then optionally saves
% and returns a binary timepoint mask.
%
% Inputs:
%   EEG            - EEGLAB EEG struct
%   maskFolderPath - path to save the output mask
%   subjNum        - subject number used for filename
%
% Output:
%   maskBadWins    - binary vector (1 x EEG.pnts), 1 = bad timepoint

    if nargin < 1 || ~isstruct(EEG) || ~isfield(EEG, 'data') || ~isfield(EEG, 'srate')
        error('Input must be a valid EEGLAB EEG struct.');
    end

    fprintf('[INFO] Opening EEG plot for manual selection...\n');
    eegplot(EEG.data, 'srate', EEG.srate);

    hFig = findall(0, 'type', 'figure', 'tag', 'EEGPLOT');  % Locate the eegplot figure
    uiwait(hFig);

    input('[INFO] Press Enter to produce mask and plot...', 's');

    if evalin('base', 'exist(''TMPREJ'', ''var'')')
        winrej = evalin('base', 'TMPREJ');  % Get TMPREJ from base workspace
    else
        warning('[WARN] No time windows were selected.');
        winrej = [];
    end

    % Create mask
    maskBadWins = false(1, EEG.pnts);

    if ~isempty(winrej)
        fprintf('[INFO] %d time window(s) selected.\n', size(winrej,1));

        startIdx = floor(winrej(:,1));
        endIdx   = ceil(winrej(:,2));

        startIdx = max(1,   startIdx);
        endIdx   = min(endIdx, EEG.pnts);

        idx_cells = arrayfun(@(s,e) s:e, startIdx, endIdx, 'UniformOutput', false);
        idx = unique([idx_cells{:}]);
        maskBadWins(idx) = true;
    end

    %% Visualise summary
    figure;
    time = (0:EEG.pnts - 1) / EEG.srate;
    plot(time, EEG.data(:, :)); % Plot all channels
    hold on;

    yl = ylim;
    if ~isempty(winrej)
        for i = 1:size(winrej, 1)
            t1 = winrej(i, 1) / EEG.srate;
            t2 = winrej(i, 2) / EEG.srate;
            patch([t1 t2 t2 t1], [yl(1) yl(1) yl(2) yl(2)], ...
                  'red', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        end
    end

    title('EEG Channels with Mask Overlay');
    xlabel('Time (s)');
    ylabel('Amplitude (\muV)');
    legend({'EEG', 'Masked region'});

    %% Ask user whether to skip saving
    userResponse = lower(strtrim(input('[QUERY] Mask will be saved. Type ''no'' to cancel: ', 's')));
    if ~strcmp(userResponse, 'no')
        savePath = fullfile(maskFolderPath, sprintf('mask_bad_wins_Subject%02d.mat', subjNum));
        save(savePath, 'maskBadWins');
        fprintf('[INFO] Mask saved to: %s\n', savePath);
    else
        fprintf('[INFO] Mask not saved.\n');
    end

    %% Clean TMPREJ from base workspace
    evalin('base', 'clear TMPREJ');
end