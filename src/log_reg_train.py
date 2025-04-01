import argparse
import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression
import utils.dslr_math as dslr
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split


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


def main():
    """Create predicted weights to classify students scores into
    appropriate houses"""
    try:
        df = parse_args()

        x = df.select_dtypes(include=[np.number]).iloc[:, 1:]
        y = df['Hogwarts House']
 
        x = x.fillna(x.mean())
        x_scaled = fitter(x)

        model = LogisticRegression(solver='liblinear', random_state=0)
        model.fit(x_scaled, y)
        # print(model.classes_)
        # print(model.intercept_)
        print("coefficients", model.coef_)
        with open('datasets/weights.csv', 'w', ) as file:
            file.write(str(model.coef_))

        print(model.predict_proba(x_scaled))
        print(model.score(x_scaled, y))

        test = pandas.read_csv('datasets/dataset_test.csv')
        x_test = test.iloc[:, 6:]
        x_test = x_test.fillna(x.mean())
        x_test = fitter(x_test)

        predicted_house = model.predict(x_test)
        with open('datasets/predictions.csv', 'w', ) as file:
            for i, house in enumerate(predicted_house):
                print(f"{i},{house}")
                file.write(f"{i},{house}\n")
        truth = pandas.read_csv('datasets/dataset_truth.csv')
        predict = pandas.read_csv('datasets/predictions.csv')
        print(f"accuracy score: {accuracy_score(truth, predict, normalize=False)}")

    # print("Before scaling:")
    # print(f"Mean: {dslr.mean(x.values.flatten()):.6f}")
    # print(f"Std: {dslr.std(x.values.flatten()):.6f}")

    # print("\nAfter scaling:")
    # print(f"Mean: {dslr.mean(x_scaled.flatten()):.10f}")
    # print(f"Std: {dslr.std(x_scaled.flatten()):.10f}")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
