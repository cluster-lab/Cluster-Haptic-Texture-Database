import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

class LMSNoiseCanceller:
    """
    A class that processes the main signal (= desired signal + noise) and noise reference signal with an LMS filter,
    and outputs the noise-cancelled audio (error signal).
    
    Supports two filter modes:
    - Causal mode: Uses only past information (conventional LMS filter)
    - Noncausal mode: Uses both past and future information (extended LMS filter)
    
    Noncausal mode is not suitable for real-time processing, but enables
    higher quality noise cancellation for offline processing.
    """
    def __init__(self, audio_path: str, 
                 filter_length: int = 32, 
                 mu: float = 0.01,
                 algorithm: str = 'nlms',
                 epsilon: float = 1e-6,
                 leakage: float = 0.0,
                 prewhiten: bool = True,
                 filter_mode: str = 'causal',
                 future_taps: int = None):
        """
        Parameters
        ----------
        primary_wav_path : str
            Path to 2-channel WAV file containing main signal + sub signal
        filter_length : int
            Number of taps in the LMS filter
        mu : float
            Learning rate (step size)
        algorithm : str
            'lms': Standard LMS algorithm
            'nlms': Normalized LMS algorithm (faster convergence)
        epsilon : float
            Small value to prevent division by zero in NLMS algorithm
        leakage : float
            Leakage coefficient (0-1). Setting a value greater than 0 improves stability
        prewhiten : bool
            Whether to perform prewhitening on the input signal
        filter_mode : str
            'causal': Uses only past information (conventional causal filter)
            'noncausal': Uses both past and future information (noncausal filter)
        future_taps : int or None
            Number of future taps for noncausal mode. If None, half of filter_length is used
        """
        # read WAV file
        audio_signal, fs = sf.read(audio_path)    # stereo signal
        
        # read WAV file
        self.d, self.fs_d = audio_signal[:, 0], fs    # 主信号 d(n)
        self.x, self.fs_x = audio_signal[:, 1], fs  # リファレンス x(n)

        # check sampling rate (strictly the same is desirable)
        if self.fs_d != self.fs_x:
            print("[Warning] Sampling frequencies of Primary and Reference signals are different.")

        # cut out the original signal at the specified timing
        start_clip = 0
        self.d = self.d[start_clip:]
        self.x = self.x[start_clip:]

        # align the lengths of the two signals (LMS algorithm, easier to handle if the lengths are the same)
        min_len = min(len(self.d), len(self.x))
        self.d = self.d[:min_len]
        self.x = self.x[:min_len]
        self.N = min_len
            
        # prewhitening (reduce autocorrelation of input signal)
        if prewhiten:
            self.x = self._prewhiten_signal(self.x)

        # filter length, learning rate, etc.
        self.M = filter_length
        self.mu = mu
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.leakage = leakage

        # set filter mode
        self.filter_mode = filter_mode
        
        # number of future taps for noncausal filter
        if future_taps is None and filter_mode == 'noncausal':
            self.future_taps = self.M // 2
            self.past_taps = self.M - self.future_taps
        elif filter_mode == 'noncausal':
            self.future_taps = future_taps
            self.past_taps = self.M - self.future_taps
        else:
            self.future_taps = 0
            self.past_taps = self.M
            
        # initialize filter coefficients (not zero but small random values)
        self.w = np.random.randn(self.M) * 0.01

        # for storing outputs
        self.e = np.zeros(self.N, dtype=np.float64)  # error (= audio after noise cancellation)
        self.y = np.zeros(self.N, dtype=np.float64)  # filter output (= estimated noise)

        # for storing filter coefficients
        self.w_history = np.zeros((self.N, self.M), dtype=np.float64)
        
        # for monitoring convergence state
        self.error_power_history = np.zeros(self.N)

    def _prewhiten_signal(self, signal):
        """
        Performs prewhitening on the signal (reduces autocorrelation and improves convergence)
        
        Prewhitening reduces the autocorrelation of the input signal and improves the convergence 
        performance of the LMS filter. The following methods are implemented:
        - 'diff': First-order difference filter (simple but effective)
        - 'ar': Whitening using autoregressive model (more advanced)
        - 'decorr': Decorrelation filter (low computational cost)
        
        Returns
        -------
        np.ndarray
            Prewhitened signal
        """
        # select prewhitening method ('diff', 'ar', 'decorr')
        method = 'diff'
        
        if method == 'diff':
            # simple prewhitening (take first-order difference)
            # y[n] = x[n] - x[n-1]
            # this is a high-pass filter and reduces low-frequency autocorrelation
            return np.concatenate([[0], np.diff(signal)])
            
        elif method == 'ar':
            # prewhitening using autoregressive model
            # estimate autocorrelation of signal and apply its inverse filter
            from scipy import linalg, signal
            
            # order of AR model
            ar_order = 10
            
            # calculate autocorrelation matrix
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(signal)-1:len(signal)+ar_order]
            
            # estimate AR parameters using Levinson-Durbin algorithm
            ar_coeffs = linalg.solve_toeplitz(autocorr[:ar_order], -autocorr[1:ar_order+1])
            
            # apply AR filter (inverse filter)
            ar_coeffs = np.concatenate([[1], ar_coeffs])
            whitened = signal.lfilter(ar_coeffs, [1], signal)
            
            return whitened
            
        elif method == 'decorr':
            # decorrelation filter (low computational cost)
            # y[n] = x[n] - alpha * x[n-1]
            # alpha is estimated from autocorrelation coefficient
            
            # calculate autocorrelation coefficient of 1 sample delay
            if len(signal) > 1:
                alpha = np.sum(signal[1:] * signal[:-1]) / np.sum(signal[:-1]**2)
                alpha = np.clip(alpha, -0.99, 0.99)  # for stability
            else:
                alpha = 0
                
            # apply decorrelation filter
            whitened = np.zeros_like(signal)
            whitened[0] = signal[0]
            whitened[1:] = signal[1:] - alpha * signal[:-1]
            
            return whitened
            
        else:
            # no prewhitening
            return signal
        
    def run_lms(self):
        """
        Uses the LMS algorithm to remove noise from the main signal.
        The result is stored in self.e.
        
        Performs causal (past only) or noncausal (past + future) processing depending on the filter mode.
        """
        
        for n in range(self.N):
            # create x_vec according to filter mode
            if self.filter_mode == 'causal':
                # causal mode (past only)
                if n < self.M:
                    x_vec = np.concatenate([self.x[n::-1], 
                                            np.zeros(self.M - (n+1))])
                else:
                    x_vec = self.x[n : n - self.M : -1]
            else:
                # noncausal mode (past + future)
                # past samples
                if n < self.past_taps:
                    past_samples = np.concatenate([self.x[n::-1], 
                                                  np.zeros(self.past_taps - (n+1))])
                else:
                    past_samples = self.x[n : n - self.past_taps : -1]
                
                # future samples
                if n + self.future_taps >= self.N:
                    # if the end of the signal is exceeded
                    future_idx = np.arange(n+1, self.N)
                    if len(future_idx) > 0:
                        future_samples = np.concatenate([self.x[future_idx], 
                                                        np.zeros(self.future_taps - len(future_idx))])
                    else:
                        future_samples = np.zeros(self.future_taps)
                else:
                    future_samples = self.x[n+1 : n+1+self.future_taps]
                
                # concatenate past and future samples
                x_vec = np.concatenate([past_samples, future_samples])

            # filter output y(n) = w^T x_vec
            self.y[n] = np.dot(self.w, x_vec)

            # error e(n) = d(n) - y(n)
            self.e[n] = self.d[n] - self.y[n]
            
            # record error signal power (for monitoring convergence)
            self.error_power_history[n] = self.e[n]**2

            # update filter coefficients
            if self.algorithm == 'nlms':
                # normalized LMS (NLMS) algorithm
                # normalize by input signal power to speed up convergence
                signal_power = np.dot(x_vec, x_vec) + self.epsilon
                step_size = self.mu / signal_power
                self.w = (1 - self.leakage * self.mu) * self.w + step_size * self.e[n] * x_vec
            else:
                # standard LMS algorithm
                self.w = (1 - self.leakage * self.mu) * self.w + self.mu * self.e[n] * x_vec
            
            # save filter coefficients history
            self.w_history[n] = self.w.copy()

    def amplify_error_signal(self, factor: float = 10):
        """
        Amplifies the error signal by a specified factor.
        """
        self.e = self.e * factor

    def get_error_signal(self) -> np.ndarray:
        """
        Returns the error signal e(n) after noise cancellation.
        This signal contains the "desired component (after noise removal)".

        Returns
        -------
        np.ndarray
            Audio waveform after noise cancellation (LMS error signal)
        """
        return self.e

    def save_error_wav(self, out_path: str):
        """
        Saves the error signal e(n) after noise cancellation as a WAV file.

        Parameters
        ----------
        out_path : str
            Path to the output WAV file
        """
        # When saving as an audio file, it's common to convert to float32
        sf.write(out_path, self.e.astype(np.float32), self.fs_d)
        print(f"Saved noise-cancelled signal to: {out_path}")

    def plot_filter_coefficients(self, interval=1000):
        """
        Plots the time series change of filter coefficients.
        
        Parameters
        ----------
        interval : int
            Plotting interval (number of samples). Larger values make processing lighter.
        """
        # Downsample the plot points
        plot_indices = np.arange(0, self.N, interval)
        
        plt.figure(figsize=(12, 8))
        
        # Plot time series change of each filter coefficient
        for i in range(self.M):
            plt.plot(plot_indices, self.w_history[plot_indices, i], 
                     label=f'w[{i}]' if i < 10 or i == self.M-1 else None)
        
        # For readability, only show labels for the first 10 coefficients and the last one
        plt.title('Time Series Change of Filter Coefficients')
        plt.xlabel('Sample Number')
        plt.ylabel('Coefficient Values')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        
        # Visualize convergence state
        plt.figure(figsize=(12, 4))
        plt.semilogy(np.arange(0, self.N, interval), self.error_power_history[::interval])
        plt.title('Convergence State')
        plt.xlabel('Sample Number')
        plt.ylabel('Error Signal Power')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_signals_time_domain(self, start_sample=0, num_samples=None):
        """
        Visualizes input and output signals in the time domain
        
        Parameters
        ----------
        start_sample : int
            Starting sample for display
        num_samples : int or None
            Number of samples to display. If None, all samples are displayed
        """
        if num_samples is None:
            end_sample = self.N
        else:
            end_sample = min(start_sample + num_samples, self.N)
            
        time_axis = np.arange(start_sample, end_sample) / self.fs_d
        
        plt.figure(figsize=(10, 6))
        
        # Overlay input and output signals
        plt.plot(time_axis, self.d[start_sample:end_sample], label='Main Signal')
        plt.plot(time_axis, self.x[start_sample:end_sample], label='Sub Signal')
        plt.plot(time_axis, self.e[start_sample:end_sample], label='Noise Cancelled Signal')
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_signals_frequency_domain(self, nfft=2048):
        """
        Visualizes input and output signals in the frequency domain
        
        Parameters
        ----------
        nfft : int
            FFT size
        """
        from scipy import signal
        
        # Calculate frequency domain
        f, Pxx_d = signal.welch(self.d, self.fs_d, nperseg=nfft)
        _, Pxx_e = signal.welch(self.e, self.fs_d, nperseg=nfft)
        
        # Convert to dB scale
        Pxx_d_db = 10 * np.log10(Pxx_d)
        Pxx_e_db = 10 * np.log10(Pxx_e)
        
        plt.figure(figsize=(14, 10))
        
        # Main signal spectrum
        plt.subplot(3, 1, 1)
        plt.semilogx(f, Pxx_d_db, label='Input Signal', alpha=0.8)
        plt.semilogx(f, Pxx_e_db, label='Noise Cancelled Signal', alpha=0.8)
        plt.title('Signal Spectrum')
        plt.ylabel('Power [dB/Hz]')
        plt.legend()
        plt.grid(True)
        plt.xlim([20, self.fs_d/2])
        
        # Transfer function from input to output
        plt.subplot(2, 1, 2)
        plt.semilogx(f, Pxx_e_db - Pxx_d_db)
        plt.title('Transfer Function from Input to Output')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power [dB/Hz]')
        plt.grid(True)
        plt.xlim([20, self.fs_d/2])
        
        plt.tight_layout()
        plt.show()
        
    def plot_noise_reduction_comparison(self, nfft=2048):
        """
        Visualizes before and after noise reduction using spectrograms
        
        Parameters
        ----------
        nfft : int
            FFT size
        """
        from scipy import signal
        
        plt.figure(figsize=(14, 10))
        
        # Spectrogram of main signal (before noise reduction)
        plt.subplot(2, 1, 1)
        f, t, Sxx = signal.spectrogram(self.d, self.fs_d, nperseg=nfft, noverlap=nfft//2)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.title('Spectrogram of Main Signal (Before Noise Reduction)')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Power [dB]')
        
        # Spectrogram of output signal (after noise reduction)
        plt.subplot(2, 1, 2)
        f, t, Sxx = signal.spectrogram(self.e, self.fs_d, nperseg=nfft, noverlap=nfft//2)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.title('Spectrogram of Output Signal (After Noise Reduction)')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Power [dB]')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_causal_noncausal(causal_canceller, noncausal_canceller, start_sample=0, num_samples=None):
        """
        Static method to compare results of causal and noncausal modes
        
        Parameters
        ----------
        causal_canceller : LMSNoiseCanceller
            LMSNoiseCanceller instance processed in causal mode
        noncausal_canceller : LMSNoiseCanceller
            LMSNoiseCanceller instance processed in noncausal mode
        start_sample : int
            Starting sample for display
        num_samples : int or None
            Number of samples to display. If None, all samples are displayed
        """
        import matplotlib.pyplot as plt
        from scipy import signal
        
        # Time domain comparison
        if num_samples is None:
            end_sample = min(causal_canceller.N, noncausal_canceller.N)
        else:
            end_sample = min(start_sample + num_samples, 
                            causal_canceller.N, 
                            noncausal_canceller.N)
            
        time_axis = np.arange(start_sample, end_sample) / causal_canceller.fs_d
        
        plt.figure(figsize=(12, 8))
        
        # Time domain comparison
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, causal_canceller.d[start_sample:end_sample], 
                label='Main Signal', alpha=0.7)
        plt.plot(time_axis, causal_canceller.e[start_sample:end_sample], 
                label='Causal Filter Output', alpha=0.7)
        plt.plot(time_axis, noncausal_canceller.e[start_sample:end_sample], 
                label='Noncausal Filter Output', alpha=0.7)
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude')
        plt.title('Time Domain Comparison')
        plt.legend()
        plt.grid(True)
        
        # Frequency domain comparison
        plt.subplot(2, 1, 2)
        nfft = 2048
        
        # Calculate frequency domain
        f, Pxx_d = signal.welch(causal_canceller.d, causal_canceller.fs_d, nperseg=nfft)
        _, Pxx_causal = signal.welch(causal_canceller.e, causal_canceller.fs_d, nperseg=nfft)
        _, Pxx_noncausal = signal.welch(noncausal_canceller.e, noncausal_canceller.fs_d, nperseg=nfft)
        
        # Convert to dB scale
        Pxx_d_db = 10 * np.log10(Pxx_d)
        Pxx_causal_db = 10 * np.log10(Pxx_causal)
        Pxx_noncausal_db = 10 * np.log10(Pxx_noncausal)
        
        plt.semilogx(f, Pxx_d_db, label='Main Signal', alpha=0.7)
        plt.semilogx(f, Pxx_causal_db, label='Causal Filter Output', alpha=0.7)
        plt.semilogx(f, Pxx_noncausal_db, label='Noncausal Filter Output', alpha=0.7)
        plt.title('Frequency Domain Comparison')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power [dB/Hz]')
        plt.legend()
        plt.grid(True)
        plt.xlim([20, causal_canceller.fs_d/2])
        
        plt.tight_layout()
        plt.show()
        
        # Transfer function comparison
        plt.figure(figsize=(12, 6))
        plt.semilogx(f, Pxx_causal_db - Pxx_d_db, label='Causal Filter')
        plt.semilogx(f, Pxx_noncausal_db - Pxx_d_db, label='Noncausal Filter')
        plt.title('Transfer Function Comparison from Input to Output')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Relative Power [dB]')
        plt.legend()
        plt.grid(True)
        plt.xlim([20, causal_canceller.fs_d/2])
        
        plt.tight_layout()
        plt.show()
        
        # Error signal power comparison
        plt.figure(figsize=(12, 6))
        
        # Smooth with moving average
        window_size = 1000
        causal_error_power_smooth = np.convolve(
            causal_canceller.error_power_history, 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        noncausal_error_power_smooth = np.convolve(
            noncausal_canceller.error_power_history, 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        
        plt.semilogy(causal_error_power_smooth, label='Causal Filter')
        plt.semilogy(noncausal_error_power_smooth, label='Noncausal Filter')
        plt.title('Error Signal Power Comparison (Convergence State)')
        plt.xlabel('Sample Number')
        plt.ylabel('Error Signal Power (Moving Average)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# テスト用
if __name__ == "__main__":
    
    audio_path = ".wav"
    output_path = audio_path.replace(".wav", "_noise_cancelled.wav")

    import time
    start_time = time.time()
    canceller = LMSNoiseCanceller(audio_path, 
                                  filter_length=700, 
                                  mu=1.0,
                                  algorithm='nlms',  # Use normalized LMS
                                  leakage=0.001,     # Set leakage coefficient
                                  prewhiten=False,    # Enable prewhitening
                                  filter_mode='noncausal')    # Use causal filter
    
    # Run noise cancellation
    canceller.run_lms()

    # save
    canceller.save_error_wav(output_path)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    # Visualization (optional)
    canceller.plot_signals_time_domain(start_sample=0, num_samples=10000)
    canceller.plot_signals_time_domain()
    

    canceller.plot_signals_frequency_domain()
    
    # canceller.plot_noise_reduction_comparison()
    
