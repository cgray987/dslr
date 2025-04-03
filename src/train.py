import argparse
import numpy as np
import pandas
import utils.log_reg as log_reg


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


def one_vs_all(x, y):
    """ create weights for probability that student will be sorted
    into one house, vs being sorted into any of the others """
    n_iterations = 100
    learning_rate = 0.1

    n_values, n_classes = x.shape
    houses = np.unique(y)
    n_houses = len(houses)

    weights = np.zeros((n_houses, n_classes))
    bias = np.zeros(n_houses)

    for i_x, current_house in enumerate(houses):
        # hufflepuff: [0, 1, 0, 0] etc
        y_bin = np.array([1 if label == current_house
                          else 0 for label in y])  # [G, H, R, S]

        weights_per_class = np.zeros(n_classes)
        bias_per_class = 0

        # gradient descent (minimize error)
        for _ in range(n_iterations):
            # guess = weights*data + bias
            guess = np.dot(x, weights_per_class) + bias_per_class
            prob = log_reg.sigmoid(guess)

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

    x = df.iloc[:, 6:]
    y = df['Hogwarts House'].to_numpy()

    # fill nan values with mean of dataset, possibly should be mean of column
    x = x.fillna(x.mean()).to_numpy()
    x = log_reg.fitter(x)  # standard scaling -- all values normed to 0-1

    weights, bias = one_vs_all(x, y)
    with open('datasets/weights.csv', 'w', ) as file:
        np.savetxt(file, weights, delimiter=' ', fmt="%1.9f")


if __name__ == "__main__":
    main()