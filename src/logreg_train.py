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
        'students scores into appropriate houses',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('dataset', type=str,
                        help='path to training dataset CSV file')
    parser.add_argument('-c', '--confusion', action='store_true',
                        help='display a confusion matrix from test data')
    parser.add_argument('-s', '--stochastic', action='store_true',
                        help='Use stochastic gradient descent to'
                        'create training weights')
    args = parser.parse_args()

    df = pandas.read_csv(args.dataset)
    return df, args


def show_confusion(x, y, weights):
    houses = np.array(['Gryffindor', 'Hufflepuff',
                       'Ravenclaw', 'Slytherin'])
    bin_predictions = predict(x, weights)
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


def stochastic_gd(x, y, learning_rate=3, epochs=1, batches=1):
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
        if epoch % 25 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    return weights, bias


def one_vs_all(x, y, n_iterations=100, learning_rate=0.1, stochastic=False):
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
        weights_per_class = np.zeros(n_classes)
        bias_per_class = 0

        if stochastic:
            w, b = stochastic_gd(x, y_bin, learning_rate)
            weights[i_x] = w
            bias[i_x] = b
        else:
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
    df, args = parse_args()

    try:
        x = df.iloc[:, 6:]
        y = df['Hogwarts House'].to_numpy()

        # fill nan values with mean of dataset, maybe should be mean of column
        x = x.fillna(x.mean()).to_numpy()
        x = log_reg.fitter(x)  # standard scaling -- all values normed to 0-1

        weights, bias = one_vs_all(x, y, stochastic=args.stochastic)
        with open('datasets/weights.csv', 'w', ) as file:
            np.savetxt(file, weights, delimiter=' ', fmt="%1.9f")
        print(f"Training weights written to {c.BOLD}{file.name}{c.RST}.")

        if args.confusion:
            show_confusion(x, y, weights)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
