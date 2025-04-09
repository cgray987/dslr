import argparse
import numpy as np
import pandas
import utils.log_reg as log_reg
from utils.colors import c


def parse_args():
    """Handle command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Create predicted weights to classify'
        'students scores into appropriate houses',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('dataset', type=str,
                        help='path to test dataset CSV file')
    parser.add_argument('weights', type=str,
                        help='path to training weights file')

    args = parser.parse_args()

    df = pandas.read_csv(args.dataset)
    return df, args


def predict(x, weights, bias):
    """ multiplies features and training weights, converts to probabilities
     and returns array of max probability for each row """
    z = np.dot(x, weights.T) + bias[np.newaxis, :]
    prob = log_reg.sigmoid(z)
    return (np.argmax(prob, axis=1))


def main():
    test_df, args = parse_args()

    try:
        x_test = test_df.iloc[:, 6:]
        x_test = x_test.fillna(x_test.mean()).to_numpy()
        x_test = log_reg.fitter(x_test)

        weights = np.loadtxt(args.weights, skiprows=1, max_rows=4)
        bias = np.loadtxt(args.weights, skiprows=6)

        houses = ['Gryffindor', 'Hufflepuff',
                  'Ravenclaw', 'Slytherin']
        house_index = predict(x_test, weights, bias)
        predictions = [houses[i] for i in house_index]

        with open('datasets/houses.csv', 'w') as file:
            file.write("Index,Hogwarts House\n")
            for i, house in enumerate(predictions):
                file.write(f"{i},{house}\n")
        print(f"House predictions written to {c.BOLD}{file.name}{c.RST}.")

        truth = pandas.read_csv('datasets/dataset_truth.csv')
        truth = truth['Hogwarts House'].to_numpy()
        compare_array = np.array(truth == predictions)
        accuracy = np.mean(compare_array)

        print(f"accuracy score: {accuracy}")

    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
