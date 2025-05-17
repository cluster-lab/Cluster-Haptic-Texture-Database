from scipy.signal import butter, lfilter, freqz

class BandPassFilter:
    def __init__(self, lowcut, highcut, fs, order=5):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.b, self.a = butter(self.order, [self.lowcut / (0.5 * self.fs), self.highcut / (0.5 * self.fs)], btype='band', analog=False)

    def apply(self, data):
        return lfilter(self.b, self.a, data)
    
    def plot_frequency_response(self):
        w, h = freqz(self.b, self.a, worN=8000)
        plt.plot(0.5 * self.fs * w / np.pi, np.abs(h), label='order = %d' % self.order)

# test
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    
    fs = 44100
    lowcut = 400
    highcut = 500
    duration = 1
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    data = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)

    order = 3
    filter_instance = BandPassFilter(lowcut, highcut, fs, order)
    filter_instance.plot_frequency_response()

    order = 4
    filter_instance = BandPassFilter(lowcut, highcut, fs, order)
    filter_instance.plot_frequency_response()

    order = 5
    filter_instance = BandPassFilter(lowcut, highcut, fs, order)
    filter_instance.plot_frequency_response()

    order = 6
    filter_instance = BandPassFilter(lowcut, highcut, fs, order)
    filter_instance.plot_frequency_response()
    
    plt.axvline(lowcut, color='k', linestyle='--')
    plt.axvline(highcut, color='k', linestyle='--')
    plt.title('Bandpass Filter Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.xlim(0, 5000)
    plt.grid()
    plt.legend()
    plt.show()
