function [env_out, Fs] = compute_envelope(audio_signal, Fs, method)
% COMPUTE_ENVELOPE Compute audio envelope using 'rms', 'peak', or 'hilbert' method
%
% Inputs:
%   - audio_signal : [samples x channels], typically from audioread
%   - Fs           : sampling frequency (Hz)
%   - method       : string, 'rms', 'peak', or 'hilbert'
%
% Output:
%   - env_out      : envelope [1 x time]
%   - Fs           : sampling rate (unchanged)

    if nargin < 3
        error('Usage: compute_envelope(audio_signal, Fs, method)');
    end

    % Convert to mono if stereo
    if size(audio_signal, 2) > 1
        audio_signal = mean(audio_signal, 2);
    end

    switch lower(method)
        case 'rms'
            % RMS envelope with 500-sample window
            env_out = envelope(audio_signal, 500, 'rms');

        case 'peak'
            % Peak envelope with 500-sample window
            env_out = envelope(audio_signal, 500, 'peak');

        case 'hilbert'
            % Hilbert transform envelope + low-pass filter
            analytic_signal = hilbert(audio_signal);
            raw_env = abs(analytic_signal);

            % FIR Low-pass filter below 50 Hz with 12.5 Hz transition band
            d = designfilt('lowpassfir', ...
                'PassbandFrequency', 50, ...
                'StopbandFrequency', 56.25, ...
                'PassbandRipple', 0.5, ...
                'StopbandAttenuation', 53, ...
                'DesignMethod', 'equiripple', ...
                'SampleRate', Fs);

            env_out = filtfilt(d, raw_env);

        otherwise
            error('Invalid method. Choose ''rms'', ''peak'', or ''Hilbert''.');
    end

    env_out = env_out(:)';  % Ensure row vector
    
    % Plot result
    t = (0:length(env_out)-1) / Fs;
    plot(t, env_out);
    xlabel('Time (s)');
    ylabel('Envelope amplitude');
    title('Audio Envelope');
end