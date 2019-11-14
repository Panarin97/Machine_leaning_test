from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import sklearn.model_selection as tst
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    digits = load_digits()
    print(digits.keys())

    plt.gray()
    plt.imshow(digits.images[0])
    plt.show()

    print(digits.target[0])  # check correspondence

    x_train, x_test, y_train, y_test = tst.train_test_split(digits.data, digits.target, test_size=0.2)

    model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')  # train KK
    model.fit(x_train, y_train)
    y_pred_kk = model.predict(x_test)

    clf = LinearDiscriminantAnalysis()  # train lda
    clf.fit(x_train, y_train)
    y_pred_lda = clf.predict(x_test)

    clf_svc = SVC(gamma='auto')  # train SVM
    clf_svc.fit(x_train, y_train)
    y_pred_svc = clf_svc.predict(x_test)

    clf_lr = LogisticRegression()  # train LR
    clf_lr.fit(x_train, y_train)  # LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    y_pred_lr = clf_lr.predict(x_test)

    print("kk: ", accuracy_score(y_test, y_pred_kk), "lda: ", accuracy_score(y_test, y_pred_lda))
    print("svc: ", accuracy_score(y_test, y_pred_svc), "lr: ", accuracy_score(y_test, y_pred_lr))


main()
