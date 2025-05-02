import argparse
import numpy as np
import pandas
import utils.log_reg as log_reg
from utils.colors import c
from logreg_predict import predict  # only for confusion matrix
from utils.plotting import plot_confusion_matrix


def parse_args():
    """Handle command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Create predicted weights to classify'
        ' students scores into appropriate houses',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('dataset', type=str,
                        help='path to training dataset CSV file')
    parser.add_argument('-c', '--confusion', action='store_true',
                        help='display a confusion matrix from test data')
    parser.add_argument('-o', '--optimizer', type=str,
                        choices=['batch', 'stochastic'],
                        default='batch', help='Choose optimization algorithm')
    parser.add_argument('-n', '--normalization', action='store_true',
                        help='Use normalization rather than standardization'
                        'for fitting algorithm')
    args = parser.parse_args()

    df = pandas.read_csv(args.dataset)
    return df, args


def show_confusion(x, y, weights, bias):
    houses = np.array(['Gryffindor', 'Hufflepuff',
                       'Ravenclaw', 'Slytherin'])
    bin_predictions = predict(x, weights, bias)
    predictions = houses[bin_predictions]
    conf_matrix = pandas.crosstab(
        predictions,  # rows (predicted)
        y,           # columns (actual)
        rownames=['Predicted'],
        colnames=['Actual']
    )
    print(f"\n{c.BOLD}Confusion Matrix: {c.RST}")
    print(conf_matrix)
    accuracy = np.sum(predictions == y) / len(y) * 100
    print(f"\nAccuracy: {accuracy:.3f}%")
    plot_confusion_matrix(conf_matrix)


def batch_gd(x, y, learning_rate=0.5, n_iterations=100):
    """Uses batch gradient descent to determine weights and bias"""
    n_values, n_features = x.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for i in range(n_iterations):
        # guess = weights*data + bias
        guess = np.dot(x, weights) + bias
        prob = log_reg.sigmoid(guess)

        error = prob - y
        weight_grad = np.dot(x.T, error) / n_values
        bias_grad = np.sum(error) / n_values

        weights -= learning_rate * weight_grad
        bias -= learning_rate * bias_grad
        if i % 10 == 0:
            print(f"\tIteration {i}, Error: {sum(error):.4f}")

    return weights, bias


def stochastic_gd(x, y, learning_rate=.1, epochs=25, batches=1):
    """Uses batches (epochs) to determine weights and bias"""
    n_values, n_features = x.shape
    weights = np.random.randn(n_features) * 0.01  # start with random weights
    bias = 0.0

    for epoch in range(epochs):  # each iteration
        epoch_loss = 0

        shuf_index = np.random.permutation(n_values)
        x_shuf = x[shuf_index]
        y_shuf = y[shuf_index]

        for i in range(0, n_values, batches):
            x_s = x_shuf[i:i + batches]
            y_s = y_shuf[i:i + batches]
            z = np.dot(x_s, weights) + bias
            predict = log_reg.sigmoid(z)

            error = predict - y_s
            dw = np.dot(x_s.T, error) / batches
            db = np.mean(error)

            weights -= learning_rate * dw
            bias -= learning_rate * db

            z_s = np.dot(x_s, weights) + bias
            pred_s = log_reg.sigmoid(z_s)
            loss = -np.mean(y_s * np.log(pred_s + 1e-15) +
                            (1-y_s) * np.log(1 - pred_s + 1e-15))
            epoch_loss += loss
        if epoch % 10 == 0:
            print(f"\tEpoch {epoch}, Loss: {epoch_loss:.4f}")

    return weights, bias


def one_vs_all(x, y, n_iterations=100, learning_rate=0.5, optimizer='batch'):
    """ create weights for probability that student will be sorted
    into one house, vs being sorted into any of the others """

    n_values, n_classes = x.shape
    houses = np.unique(y)
    n_houses = len(houses)

    weights = np.zeros((n_houses, n_classes))
    bias = np.zeros(n_houses)

    for i_x, current_house in enumerate(houses):
        # hufflepuff: [0, 1, 0, 0] etc
        y_bin = np.array([1 if label == current_house
                         else 0 for label in y])  # [G, H, R, S]

        print(f"{c.BOLD}{current_house}{c.RST}")
        if optimizer == 'stochastic':
            w, b = stochastic_gd(x, y_bin, learning_rate)
        else:  # batch
            w, b = batch_gd(x, y_bin, learning_rate, n_iterations)

        weights[i_x] = w
        bias[i_x] = b
    return weights, bias


def main():
    try:
        df, args = parse_args()

        x = df.iloc[:, 6:]
        y = df['Hogwarts House'].to_numpy()

        # fill nan values with mean of dataset, maybe should be mean of column
        x = x.fillna(x.mean()).to_numpy()
        if args.normalization:
            x = log_reg.fitter(x)
        else:
            x = log_reg.fitter_standardization(x)

        weights, bias = one_vs_all(x, y, optimizer=args.optimizer)
        with open('datasets/weights.csv', 'w', ) as file:
            np.savetxt(file, weights, delimiter=' ',
                       fmt="%1.9f", header='weights')
            np.savetxt(file, bias, delimiter=' ',
                       fmt="%1.9f", header='bias')
        print(f"Training weights written to {c.BOLD}{file.name}{c.RST}.")

        if args.confusion:
            show_confusion(x, y, weights, bias)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
