function [avg_diag, std_diag] = all_diagonal_averages(dRSA)
% ALL_DIAGONAL_AVERAGES - Computes the mean of all diagonals in a square matrix
%
% Inputs:
%   dRSA      - Square matrix (N x N)
%
% Output:
%   avg_diag  - Vector of length (2N - 1), containing the mean of each diagonal
%               ordered from the bottom-left (-N+1) to top-right (+N-1)

    if size(dRSA,1) ~= size(dRSA,2)
        error('Matrix must be square.');
    end

    N = size(dRSA, 1);
    lags = -(N-1):(N-1); % from -max offset (bottom-left) to +max offset (top-right)
    avg_diag = zeros(1, length(lags));

    for i = 1:length(lags)
        k = lags(i);
        d = diag(dRSA, k); % get the diagonal with offset k
        avg_diag(i) = mean(d, 'omitnan');
        std_diag(i) = std(d, 'omitnan');
    end
end