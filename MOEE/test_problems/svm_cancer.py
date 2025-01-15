import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class svm_wdbc:
    """
    Breast Cancer Wisconsin (Diagnostic)
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

    https://www.openml.org/search?type=data&sort=runs&id=1510&status=active
    """

    def __init__(self):
        self.dim = 2
        # https://jmlr.org/papers/volume20/18-444/18-444.pdf (Table 1)
        self.lb = np.array([2 ** -10, 2 ** -10])
        self.ub = np.array([2 ** 10, 2 ** 10])

        # self.xopt = np.loadtxt('._bbob_problem_best_parameter.txt')
        self.yopt = 0
        self.cf = None

        # load data
        self.wdbc = load_breast_cancer()
        self.X = self.wdbc.data
        self.y = self.wdbc.target

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=1, stratify=self.y)

    def __call__(self, x):
        x = np.atleast_2d(x)
        y = np.zeros((x.shape[0], 1))
        for idx, i in enumerate(x):
            model = SVC(C=i[0], gamma=i[1])
            model.fit(self.X_train, self.y_train)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            y[idx] = 1-np.mean(scores)  # minimise

        return y.ravel()

    def test_score(self, x):
        x = np.atleast_2d(x)
        y = np.zeros((x.shape[0], 1))
        for idx, i in enumerate(x):
            model = SVC(C=i[0], gamma=i[1])
            model.fit(self.X_train, self.y_train)
            scores = model.score(self.X_test, self.y_test)
            # y[idx] = 1-scores
            y[idx] = scores
        return y.ravel()


if __name__ == "__main__":
    a = np.array([[0.01, 0.1],
                  [22, 26],
                  [8, 4],
                  [10, 0.01]])

    print(svm_wdbc().__call__(a))
