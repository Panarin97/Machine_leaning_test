from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.model_selection as tst

def evaluation(pred, result):
    count = 0
    index = 0
    for i in pred:
        if i == result[index]:
            count += 1
            index += 1
        else:
            index += 1

    return count/len(pred)


def main():
    mat = loadmat("twoClassData2.mat")

    X = mat["X"]  # Collect the two variables.
    Y = mat["y"].ravel()


    X_train, X_test, y_train, y_test = tst.train_test_split(X, Y, test_size=0.33)

    model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    y_hat2 = clf.predict(X_test)

    print(evaluation(y_hat, y_test), " part of correctly guessed samples")
    print(evaluation(y_hat2, y_test), " part of correctly guessed samples")

main()