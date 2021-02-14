import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets

from utils import normalize, make_diagonal, train_test_split
from utils import accuracy_score
from utils import Plot
from regression import LinearRegression, LassoRegression
from logistic_regression import LogisticRegression

def test_0():
    advertising = pd.read_csv('data/Advertising.csv')
    print(advertising.info())
    sns.regplot(advertising.TV, advertising.sales, order=1, ci=None, scatter_kws={'color':'r', 's':9})
    plt.xlim(-10,310)
    plt.ylim(ymin=0)
    plt.show()

def test_linearRegression():
    data = pd.read_csv('data/TempLinkoping2016.txt', sep="\t")
    print(data.shape)

    X = np.array(data['time']).reshape((-1, 1))
    y = data['temp'].values

    print (X[:10])

    # Color map
    cmap = plt.get_cmap('viridis')

    plt.scatter(X, y)

    model = LassoRegression(degree=15,
                            reg_factor=0.05,
                            learning_rate=0.001,
                            n_iterations=4000)
    #model = LinearRegression()
    model.fit(X,y)
    y_pred_line = model.predict(X)

    plt.plot(X, y_pred_line, color='black', linewidth=2)
    plt.show()


def test_logistic_regression():
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    clf = LogisticRegression(gradient_descent=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the result
    Plot().plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)


if __name__ == "__main__":
    # test_linearRegression()
    test_logistic_regression()

