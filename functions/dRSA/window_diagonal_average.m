function avg_diag = window_diagonal_average(dRSA, window_size)
% WINDOW_DIAGONAL_AVERAGE - Averages over diagonals centered on main diagonal
% using a symmetric window of odd length (e.g., 257 = 2*128 + 1).
%
% Inputs:
%   dRSA        - Square matrix (N x N)
%   window_size - Odd number (e.g., 257) defining number of diagonals to include
%
% Output:
%   avg_diag    - Vector of length window_size, average values across diagonals
%                 from -half_window to +half_window

    if size(dRSA,1) ~= size(dRSA,2)
        error('Matrix must be square.');
    end
    if mod(window_size,2) ~= 1
        error('Window size must be odd (2k+1 for symmetric centering).');
    end

    N = size(dRSA, 1);
    half_win = floor(window_size / 2);
    lags = -half_win : half_win;

    avg_diag = zeros(1, length(lags));

    for i = 1:length(lags)
        k = lags(i); % lag (offset from main diagonal)
        diag_values = diag(dRSA, k);
        avg_diag(i) = mean(diag_values, 'omitnan');
    end
end