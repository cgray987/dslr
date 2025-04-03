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
                        help='path to test dataset CSV file')
    parser.add_argument('weights', type=str,
                        help='path to training weights file')
    args = parser.parse_args()

    df = pandas.read_csv(args.dataset)
    return df, args


def predict(x, weights):
    """ multiplies features and training weights, converts to probabilities
     and returns array of max probability for each row """
    # z = np.dot(x, weights.T) + bias[np.newaxis, :]
    z = np.dot(x, weights.T)
    prob = log_reg.sigmoid(z)
    return (np.argmax(prob, axis=1))


def main():
    test_df, args = parse_args()

    x_test = test_df.iloc[:, 6:]
    x_test = x_test.fillna(x_test.mean()).to_numpy()
    x_test = log_reg.fitter(x_test)

    weights = np.loadtxt(args.weights)
    # print(weights)

    # print(x_test.shape)
    # print(weights.shape)
    house_index = predict(x_test, weights)
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    predictions = [houses[i] for i in house_index]

    with open('datasets/houses.csv', 'w') as file:
        file.write("Index,Hogwarts House\n")
        for i, house in enumerate(predictions):
            file.write(f"{i},{house}\n")
            # print(f"{i},{house}")


if __name__ == "__main__":
    main()
