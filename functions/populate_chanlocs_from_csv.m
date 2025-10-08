function EEG = populate_chanlocs_from_csv(EEG, csvFile)
% POPULATE_CHANLOCS_FROM_BIOSEMI128
% Populate EEGLAB EEG.chanlocs field using Biosemi 128-channel CSV file.
%
% Inputs:
%   EEG      - EEGLAB EEG structure (must have EEG.nbchan matching the CSV)
%   csvFile  - Path to the biosemi128.csv file
%
% Output:
%   EEG      - Updated EEG structure with populated chanlocs

    % Read the CSV
    T = readtable(csvFile);

    % Sanity check
    if 128 ~= height(T)
        error('Mismatch: EEG has %d channels, but CSV has %d entries.', '128', height(T));
    end

    % Required columns
    requiredCols = {'labels', 'X', 'Y', 'Z'};
    if ~all(ismember(requiredCols, T.Properties.VariableNames))
        error('CSV must contain columns: labels, X, Y, Z');
    end

    % Assign channel locations
    EEG.chanlocs = struct([]);
    for i = 1:EEG.nbchan
        EEG.chanlocs(i).labels = T.labels{i};
        EEG.chanlocs(i).X      = -T.X(i);
        EEG.chanlocs(i).Y      = -T.Y(i);
        EEG.chanlocs(i).Z      = T.Z(i);
        EEG.chanlocs(i).type   = 'EEG';
    end

    % Set standard field for EEGLAB
    EEG.nbchan = height(T);
    

    fprintf('[INFO] Channel labels and positions successfully loaded from CSV.\n');
end