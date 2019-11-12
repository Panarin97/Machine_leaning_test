import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np


scores = []
frequencies = []
values = []
f0 = 0.015
x = np.zeros(100)

w = np.sqrt(0.3) * np.random.randn(100)

for i in range (0, 100):

    x[i] = (np.sin(2*np.pi*f0*i) + w[i])

plt.plot(x, 'ro')
plt.show()

for f in np.linspace(0, 0.5, 1000):

    n = np.arange(100)
    z = -2*np.pi*1j*f*n
    e = np.exp(z)
    score = abs(np.dot(x, e))
    scores.append(score)
    frequencies.append(f)
fHat = frequencies[np.argmax(scores)]

print(fHat)

