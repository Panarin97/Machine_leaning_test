import matplotlib.pyplot as plt
import numpy as np


def main():
    length = 50
    m = np.zeros(1000)
    n = np.arange(length)
    y = np.cos(2 * np.pi * 0.1 * n)

    signal = np.concatenate((m, y, m))  # create a sinusoid
    noisy_signal = signal + np.sqrt(0.5) * np.random.randn(signal.size)  # create a sinusoid with random noise

    h = np.exp(-2 * np.pi * 1j * 0.1 * n)

    deter_detect = np.convolve(noisy_signal, signal, 'same')
    stoch_detect = np.abs(np.convolve(h, noisy_signal, 'same'))

    fig, ax = plt.subplots(4, 1)

    ax[0].plot(signal)
    ax[1].plot(noisy_signal)
    ax[2].plot(deter_detect)
    ax[3].plot(stoch_detect)

    plt.show()


main()
