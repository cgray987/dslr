import argparse
import numpy as np
import pandas
import utils.dslr_math as dslr


def parse_args():
    """Handle command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Create predicted weights to classify'
        'students scores into appropriate houses',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('dataset', type=str,
                        help='path to training dataset CSV file')
    args = parser.parse_args()

    df = pandas.read_csv(args.dataset)
    return df


def fitter(X):
    """Scales columns to unit variance
    mean becomes = 0, variance becomes = 1"""
    X = np.array(X)
    n_features = X.shape[1]
    fitted = np.zeros_like(X)

    for i in range(n_features):
        col = X[:, i]
        mean = dslr.mean(col[~np.isnan(col)])
        std = dslr.std(col[~np.isnan(col)])

        if std == 0:
            fitted[:, i] = 0
        else:
            fitted[:, i] = (col - mean) / std

    return fitted


def sigmoid(guess):
    """returns probability (array) between 0-1 using 
    sigmoid function"""
    guess = np.clip(guess, -100, 100)
    return (1 / (1 + np.exp(-guess)))


def one_vs_all(x: np.array, y: np.array):
    """ create weights for probability that student will be sorted
    into one house, vs being sorted into any of the others """
    n_iterations = 100
    learning_rate = 0.1

    n_values, n_classes = x.shape
    houses = np.unique(y)
    n_houses = len(houses)

    weights = np.zeros((n_houses, n_classes))
    bias = np.zeros_like(y)

    for i_x, current_house in enumerate(houses):
        # hufflepuff: [1, 0, 0, 0] etc
        np.where(current_house == y, 0, 1)
        y_bin = np.array([1 if label == current_house
                          else 0 for label in y])

        weights_per_class = np.zeros(n_classes)
        bias_per_class = 0

        # gradient descent (minimize error)
        for _ in range(n_iterations):
            # guess = weights*data + bias
            guess = np.dot(x, weights_per_class) + bias_per_class
            prob = sigmoid(guess)
            
            error = prob - y_bin
            weight_grad = (1 / n_values) * np.dot(x.T, error)
            bias_grad = (1 / n_values) * np.sum(error)

            weights_per_class -= learning_rate * weight_grad
            bias_per_class -= learning_rate * bias_grad

        weights[i_x] = weights_per_class
        bias[i_x] = bias_per_class
    return weights, bias


def main():
    df = parse_args()

    x = df.select_dtypes(include=[np.number]).iloc[:, 1:]
    y = df['Hogwarts House'].to_numpy()

    x = x.fillna(x.mean()).to_numpy()  # fill nan values with mean of dataset, possibly should be mean of column
    x = fitter(x)  # standard scaling -- all values normed to 0-1

    weights, bias = one_vs_all(x, y)
    with open('datasets/weights2.csv', 'w', ) as file:
        np.savetxt(file, weights)




if __name__ == "__main__":
    main()