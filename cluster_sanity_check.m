% cluster_sanity_check.m
% Quick sanity check for the cluster using parfor; prints worker path info and probes mounts.

fprintf('Cluster sanity check started at %s\n', datestr(now));

% Candidate locations for code and toolboxes as mounted on different machines.
basePathCandidates = {
    '/Volumes/MORWUR/Projects/DAMIANO/SpeDiction/dRSA_speech_V02', ...
    '/mnt/storage/tier2/morwur/Projects/DAMIANO/SpeDiction/dRSA_speech_V02'
};

eeglabPathCandidates = {
    '/Volumes/MORWUR/Projects/SHARED/shared_toolboxes/matlab_toolboxes/eeglab2025.0.0', ...
    '/mnt/storage/tier2/morwur/Projects/SHARED/shared_toolboxes/matlab_toolboxes/eeglab2025.0.0'
};

% Dataset configuration.
subjects = [1 1];
runId = 4;
fprintf('Subjects to check: %s (run %d).\n', mat2str(subjects), runId);

numSubjects = numel(subjects);
subjectSummaries = cell(numSubjects, 1);

parfor idx = 1:numSubjects
    subj = subjects(idx);
    fprintf('[Subject %d] Worker starting; pwd=%s\n', subj, pwd);

    % Build summary scaffold so we can stash debug info even on failure.
    localSummary = struct( ...
        'subject', subj, ...
        'dataset', sprintf('Subject%d_Run%d.set', subj, runId), ...
        'datasetPath', '', ...
        'success', false, ...
        'message', '', ...
        'setname', '', ...
        'nbchan', NaN, ...
        'pnts', NaN, ...
        'srate', NaN, ...
        'trials', NaN, ...
        'sizeMB', NaN, ...
        'workerPwd', pwd, ...
        'selectedBasePath', '', ...
        'selectedEEGLABPath', '', ...
        'signalToolboxVersion', '', ...
        'probeReports', {{}}, ...
        'popLoadsetAvailable', false);

    % Add EEGLAB path on the worker by testing each candidate.
    for pathIdx = 1:numel(eeglabPathCandidates)
        candidate = eeglabPathCandidates{pathIdx};
        if exist(candidate, 'dir')
            addpath(genpath(candidate));
            localSummary.selectedEEGLABPath = candidate;
            break;
        end
    end
    if ~isempty(localSummary.selectedEEGLABPath)
        fprintf('[Subject %d] EEGLAB path added: %s\n', subj, localSummary.selectedEEGLABPath);
    else
        fprintf(2, '[Subject %d] No EEGLAB path found on worker.\n', subj);
    end

    % Check toolbox availability on the worker.
    toolboxInfo = ver('signal');
    if ~isempty(toolboxInfo)
        localSummary.signalToolboxVersion = toolboxInfo(1).Version;
        fprintf('[Subject %d] Signal Processing Toolbox version %s detected.\n', subj, toolboxInfo(1).Version);
    else
        fprintf(2, '[Subject %d] Signal Processing Toolbox not detected on worker.\n', subj);
    end

    % Probe candidate mounts by attempting to cd into each.
    probeTargets = [basePathCandidates, ...
        {'/mnt/storage/tier2/morwur/Projects/DAMIANO/', '/Volumes/MORWUR/Projects/DAMIANO/'}];
    origDir = pwd;
    probeReports = cell(numel(probeTargets), 1);
    for pIdx = 1:numel(probeTargets)
        target = probeTargets{pIdx};
        status = sprintf('Probe %s: ', target);
        if exist(target, 'dir')
            status = [status 'exists'];
            try
                cd(target);
                status = sprintf('%s, cd ok (pwd=%s)', status, pwd);
            catch probeErr
                status = sprintf('%s, cd failed (%s)', status, probeErr.message);
            end
        else
            status = [status 'missing'];
        end
        probeReports{pIdx} = status;
        cd(origDir);
    end
    localSummary.probeReports = probeReports;

    % Determine which base path is reachable on this worker.
    for pathIdx = 1:numel(basePathCandidates)
        candidate = basePathCandidates{pathIdx};
        if exist(candidate, 'dir')
            localSummary.selectedBasePath = candidate;
            break;
        end
    end
    if isempty(localSummary.selectedBasePath)
        localSummary.message = 'No accessible base path detected on worker.';
        subjectSummaries{idx} = localSummary;
        continue;
    end

    subjectDir = fullfile(localSummary.selectedBasePath, 'data', 'EEG', sprintf('Subject%d', subj));
    datasetPath = fullfile(subjectDir, localSummary.dataset);
    localSummary.datasetPath = datasetPath;

    if exist('pop_loadset', 'file') == 2
        localSummary.popLoadsetAvailable = true;
    else
        localSummary.message = 'pop_loadset not available after addpath.';
        subjectSummaries{idx} = localSummary;
        continue;
    end

    if ~exist(datasetPath, 'file')
        localSummary.message = 'Dataset not found on disk.';
        subjectSummaries{idx} = localSummary;
        continue;
    end

    fprintf('[Subject %d] Attempting to load %s\n', subj, localSummary.dataset);
    try
        EEG = pop_loadset('filename', localSummary.dataset, 'filepath', subjectDir);
        localSummary.success = true;
        if isfield(EEG, 'setname') && ~isempty(EEG.setname)
            localSummary.setname = EEG.setname;
        end
        localSummary.nbchan = EEG.nbchan;
        localSummary.pnts = EEG.pnts;
        localSummary.srate = EEG.srate;
        if isfield(EEG, 'trials')
            localSummary.trials = EEG.trials;
        end
        if isfield(EEG, 'data')
            dataBytes = numel(EEG.data) * 8; % double precision estimate
            localSummary.sizeMB = dataBytes / (1024^2);
        end
        fprintf('[Subject %d] Loaded successfully.\n', subj);
    catch ME
        localSummary.message = ME.message;
        fprintf(2, '[Subject %d] Failed to load dataset: %s\n', subj, ME.message);
    end

    subjectSummaries{idx} = localSummary;
end

% Summarize results once workers finish.
for idx = 1:numSubjects
    summary = subjectSummaries{idx};
    fprintf('--- Summary for subject %d ---\n', summary.subject);
    fprintf('  Worker pwd: %s\n', summary.workerPwd);
    fprintf('  Selected base path: %s\n', summary.selectedBasePath);
    fprintf('  Selected EEGLAB path: %s\n', summary.selectedEEGLABPath);
    fprintf('  pop_loadset available: %d\n', summary.popLoadsetAvailable);
    if ~isempty(summary.signalToolboxVersion)
        fprintf('  Signal toolbox version: %s\n', summary.signalToolboxVersion);
    end
    for pIdx = 1:numel(summary.probeReports)
        fprintf('  %s\n', summary.probeReports{pIdx});
    end

    if summary.success
        fprintf('  Dataset: %s\n  Path: %s\n  Channels: %d\n  Time points: %d\n  Sampling rate: %.2f Hz\n  Trials: %.0f\n  Approximate data size: %.2f MB\n', ...
            summary.dataset, summary.datasetPath, summary.nbchan, summary.pnts, summary.srate, ...
            summary.trials, summary.sizeMB);
    else
        fprintf(2, '  Failure reason: %s\n', summary.message);
    end
end

fprintf('Cluster sanity check completed at %s\n', datestr(now));
