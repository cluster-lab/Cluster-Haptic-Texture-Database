from scipy.signal import butter, lfilter, freqz

class LowPassFilter:
    def __init__(self, cutoff, fs, order=5):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.b, self.a = butter(self.order, self.cutoff / (0.5 * self.fs), btype='low', analog=False)

    def apply(self, data):
        return lfilter(self.b, self.a, data)
    
    def plot_frequency_response(self):
        w, h = freqz(self.b, self.a, worN=8000)
        plt.plot(0.5 * self.fs * w / np.pi, np.abs(h), label='order = %d' % self.order)
        #plt.plot(self.cutoff, 0.5 * np.sqrt(2), 'ko')
        

# test
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    
    fs = 44100
    cutoff = 1000
    duration = 1
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    data = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)


    order = 5
    filter_instance = LowPassFilter(cutoff, fs, order)
    filter_instance.plot_frequency_response()

    order = 10
    filter_instance = LowPassFilter(cutoff, fs, order)
    filter_instance.plot_frequency_response()

    order = 12
    filter_instance = LowPassFilter(cutoff, fs, order)
    filter_instance.plot_frequency_response()

    order = 13
    filter_instance = LowPassFilter(cutoff, fs, order)
    filter_instance.plot_frequency_response()
    
    plt.axvline(cutoff, color='k', linestyle='--')
    plt.title('Lowpass Filter Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.xlim(0, 5000)
    plt.grid()
    plt.legend()
    plt.show()
