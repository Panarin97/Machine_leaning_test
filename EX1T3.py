from scipy.io import loadmat
import matplotlib.pyplot as plt



def main():
    mat = loadmat("twoClassData.mat")
    print(mat.keys())
    X = mat["X"]  # Collect the two variables.
    y = mat["y"].ravel()

    P = X[y == 0, :]
    E = X[y == 1, :]

    plt.plot(P[:, 0], P[:, 1], 'ro')
    plt.plot(E[:, 0], E[:, 1], 'bo')

    plt.show()

main()