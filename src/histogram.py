import matplotlib.pyplot as plt
import pandas
import numpy as np
import argparse
from utils.clean_data import Data
import csv


def plot_hist(ax: plt.Axes, df: pandas.DataFrame, house_name: str):
    house = df[df["hogwarts_house"] == house_name]
    print(house)
    scores = house.loc[:, "arithmancy":"flying"]
    norm = (scores - scores.min()) / (scores.max() - scores.min())

    ax.hist(
        norm.to_numpy().flatten(),
        bins=40, rwidth=0.8,
        stacked=True,
        alpha=0.3,
        label=house_name
        )
    ax.legend(["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"])
    ax.set_title(house_name)


def histogram(X, legend, title, xlabel, ylabel):
    h1 = X[:327]
    h1 = h1[~np.isnan(h1)]
    plt.hist(h1, color='red', alpha=0.5, edgecolor='black')

    h2 = X[327:856]
    h2 = h2[~np.isnan(h2)]
    plt.hist(h2, color='yellow', alpha=0.5, edgecolor='black')

    h3 = X[856:1299]
    h3 = h3[~np.isnan(h3)]
    plt.hist(h3, color='blue', alpha=0.5, edgecolor='black')

    h4 = X[1299:]
    h4 = h4[~np.isnan(h4)]
    plt.hist(h4, color='green', alpha=0.5, edgecolor='black')

    plt.legend(legend, loc='upper right', frameon=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# def load_csv(filename):
#     dataset = list()
#     with open(filename) as csvfile:
#         reader = csv.reader(csvfile)
#         try:
#             for _ in reader:
#                 row = list()
#                 for value in _:
#                     try:
#                         value = float(value)
#                     except:
#                         if not value:
#                             value = np.nan
#                     row.append(value)
#                 dataset.append(row)
#         except csv.Error as e:
#             print(f'file {filename}, line {reader.line_num}: {e}')
#     return np.array(dataset, dtype=object)


def main():
    parser = argparse.ArgumentParser(
        description='Display histograms from dataset'
    )
    parser.add_argument('dataset', type=str, help='path to dataset CSV file')
    parser.add_argument('-c', '--compare', action='store_true',
                        help='compare with pandas describe() output')

    # try:
    args = parser.parse_args()
    df = pandas.read_csv(args.dataset)
    # dataset = Data(df)
    
    # fig, axs = plt.subplots()
    # plot_hist(axs, dataset.df, "Gryffindor")
    # plot_hist(axs, dataset.df, "Ravenclaw")
    # plot_hist(axs, dataset.df, "Slytherin")
    # plot_hist(axs, dataset.df, "Hufflepuff")
    # plt.show()

    # df = load_csv(args.dataset)

    data = df.to_numpy()[1:, :]
    print(data)
    data = data[data[:, 1].argsort()]

    X = np.array(data[:, 16], dtype=float)
    legend = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    histogram(X, legend=legend, title=data[0, 16], xlabel='Marks', ylabel='Number of student')


    # except Exception as e:
    #     print(f"{Exception.__name__}: {e}")

if __name__ == "__main__":
    main()