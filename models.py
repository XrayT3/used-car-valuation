import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


def get_model(df, num_parameters):
    data_y = df[["price"]]  # image
    data_x = df[num_parameters]  # arguments

    # split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.45, random_state=0)

    # create first model
    m_linear_model = LinearRegression()  # multiple linear model
    m_linear_model.fit(x_train, y_train)  # train the model
    y_pred = m_linear_model.predict(x_test)  # get results

    # display model results
    width = 7
    height = 5
    plt.figure(figsize=(width, height))
    xs = np.linspace(1, x_test.shape[0], x_test.shape[0])  # represent every car
    plt.plot(xs, y_test)  # original prices(blue)
    plt.plot(xs, y_pred)  # prediction prices(yellow)
    plt.title("The first model")
    plt.show()

    # create second model with standardization data and polynomial features
    pipeline_parameters = [("scale", StandardScaler()),
                           ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                           ("model", LinearRegression())]
    pipe = Pipeline(pipeline_parameters)  # multiple linear model with normalization and polynomial features
    pipe.fit(x_train, y_train)  # train the model
    y_pred = pipe.predict(x_test)  # get results

    # display model results
    plt.figure(figsize=(width, height))
    plt.plot(xs, y_test)  # original prices(blue)
    plt.plot(xs, y_pred)  # prediction prices(yellow)
    plt.title("The second model")
    plt.show()
    plt.close()

    # check the precision of models
    print("\nThe R-square of the first model:", m_linear_model.score(x_test, y_test))
    print("The R-square of the second model:", pipe.score(x_test, y_test))

    # the second model has worse result because of over fitting
    # let's fight with it

    # on the beginning transform data
    pr = PolynomialFeatures(degree=2, include_bias=False)
    x_train_pr = pr.fit_transform(x_train)
    x_test_pr = pr.fit_transform(x_test)

    scores = []
    alphas = 10 * np.array(range(1, 500))
    # collect scores for different alphas
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(x_train_pr, y_train)
        scores.append(model.score(x_test_pr, y_test))

    # find alpha that maximizes R squared
    max_r = max(scores)
    best_alpha = scores.index(max_r) * 10

    # create final model with maximum accuracy
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(x_train_pr, y_train)
    print("The R-square of the third model:", ridge_model.score(x_test_pr, y_test))
    return ridge_model
