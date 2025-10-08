function plot_dRSA_subj_avg(dRSA_diag_avg, dRSA_diag_std,...
                        mRSA_diag_avg, mRSA_diag_std,...
                        nRSA_diag_avg, nRSA_diag_std,...
                        Fs, title_plot, lagWindowSec)
    % PLOT_DRSA_SUBJ_AVG Draws diagonal-average time courses with optional lag limits.
    % lagWindowSec: [min max] lag window in seconds (e.g., [-3 3]). Empty -> full range.

    if nargin < 8 || isempty(Fs)
        error('Sampling rate Fs must be provided.');
    end
    if nargin < 9 || isempty(lagWindowSec)
        lagWindowSec = [-Inf Inf];
    elseif numel(lagWindowSec) ~= 2
        error('lagWindowSec must be a 1x2 vector, e.g., [-3 3].');
    else
        lagWindowSec = sort(lagWindowSec(:)');
    end

    % ensure row vectors for downstream processing
    dRSA_diag_avg = dRSA_diag_avg(:)';
    dRSA_diag_std = dRSA_diag_std(:)';
    mRSA_diag_avg = mRSA_diag_avg(:)';
    mRSA_diag_std = mRSA_diag_std(:)';
    nRSA_diag_avg = nRSA_diag_avg(:)';
    nRSA_diag_std = nRSA_diag_std(:)';

    % derive x axis values
    N = (numel(dRSA_diag_avg)+1)/2;
    lags_samples = -(N-1):(N-1);                    % diagonal offsets in samples
    lags_sec = lags_samples / Fs;                   % in seconds
    lags_ms = lags_sec * 1000;                      % convert to milliseconds

    lagMask = lags_sec >= lagWindowSec(1) & lags_sec <= lagWindowSec(2);
    if ~any(lagMask)
        lagMask = true(size(lags_sec));
    end

    % apply window mask and drop NaNs for plotting
    [lags_dRSA, dMeanSel, dStdSel] = window_and_clean(lags_ms, dRSA_diag_avg, dRSA_diag_std, lagMask);
    [lags_mRSA, mMeanSel, mStdSel] = window_and_clean(lags_ms, mRSA_diag_avg, mRSA_diag_std, lagMask);
    [lags_nRSA, nMeanSel, nStdSel] = window_and_clean(lags_ms, nRSA_diag_avg, nRSA_diag_std, lagMask);

    % set title
    sgtitle(title_plot, 'Interpreter', 'none');  % keep literal underscores

    % ----------- subplot 1 : dRSA diagonal average -----------------------
    subplot(3,2,2)
    plot_with_shaded_error(lags_dRSA, dMeanSel, dStdSel);
    hold on

    % highlight the maximum correlation point
    if ~isempty(dMeanSel)
        [peakVal, idxPeak] = max(dMeanSel, [], 'omitnan');
    else
        peakVal = NaN;
        idxPeak = [];
    end
    if ~isempty(idxPeak) && ~isnan(peakVal)
        peakLag = lags_dRSA(idxPeak);
        plot(peakLag, peakVal, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
        labelTxt = sprintf('%.3f @ %.0f ms', peakVal, peakLag);
        peakLabel = text(peakLag, peakVal, [' ' labelTxt], 'Color', 'r', ...
             'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
             'BackgroundColor', 'none', 'Margin', 1);
        set(peakLabel, 'FontUnits', 'normalized', 'FontSize', 0.1);
    end

    format_diag_axes('Mean dRSA value across all diagonals', 'Mean dRSA value');

    % ----------- subplot 2 : mRSA diagonal average -----------------------
    subplot(3,2,4)
    plot_with_shaded_error(lags_mRSA, mMeanSel, mStdSel);
    format_diag_axes('Mean mRSA value across all diagonals', 'Mean mRSA value');

    % ----------- subplot 3 : nRSA diagonal average -----------------------
    subplot(3,2,6)
    plot_with_shaded_error(lags_nRSA, nMeanSel, nStdSel);
    format_diag_axes('Mean nRSA value across all diagonals', 'Mean nRSA value');

    function [lags_ms_out, meanValsOut, stdValsOut] = window_and_clean(lags_ms_all, meanVals, stdVals, mask)
        lags_ms_out = lags_ms_all(mask);
        meanValsOut = meanVals(mask);
        stdValsOut = stdVals(mask);

        valid = ~isnan(meanValsOut);
        lags_ms_out = lags_ms_out(valid);
        meanValsOut = meanValsOut(valid);
        stdValsOut = stdValsOut(valid);
        stdValsOut(isnan(stdValsOut)) = 0;
    end

    function plot_with_shaded_error(xVals, meanVals, stdVals)
        if isempty(xVals)
            return;
        end
        curveUpper = meanVals + stdVals;
        curveLower = meanVals - stdVals;
        fill([xVals, fliplr(xVals)], [curveUpper, fliplr(curveLower)], ...
             'c', 'FaceAlpha', 0.1, 'EdgeAlpha', 0.2, 'LineStyle', 'none');
        hold on
        plot(xVals, meanVals, 'LineWidth', 2, 'Color', [0 0.4470 0.7410]);
    end

    function format_diag_axes(titleStr, yLabelStr)
        ax = gca;
        ax.YRuler.Exponent = 0;
        xlabel('Lag (ms)');
        ylabel(yLabelStr);
        title(titleStr);
        grid on;
    end
end
