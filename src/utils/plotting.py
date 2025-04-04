import matplotlib.pyplot as plt
import numpy as np
import utils.dslr_math as dslr


def plot_confusion_matrix(df_confusion, title='Confusion matrix'):
    plt.matshow(df_confusion, cmap='Purples')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()


def histogram(ax, df, value_col, legend, stats=False):
    """Plots histogram of given course grades. Also prints stats of
    said grades per house."""

    if stats:
        print(f"Stats for {value_col}:")
    for house in legend:
        house_scores = df[df["Hogwarts House"] == house][value_col].to_numpy()
        house_scores = house_scores[~np.isnan(house_scores)]
        ax.hist(
            house_scores,
            color=legend[house],
            alpha=0.5,
            stacked=True,
            edgecolor='black',
            label=house,
            )
        # Stats
        if len(house_scores) > 0 and stats:
            print(f"\t{house:10} | max: {dslr.max(house_scores):8.1f} "
                  f"| min: {dslr.min(house_scores):8.1f} "
                  f"| mean: {dslr.mean(house_scores):8.1f} "
                  f"| std: {dslr.std(house_scores):8.1f}")

    ax.set_xlabel('Grade')
    ax.set_title(value_col)
    ax.legend(legend, loc='upper right', frameon=False)


def scatter(ax, df, x_col, y_col, legend, scatter_size=0.5):
    """Creates scatter plot comparing two course scores, colored by house"""

    for house, color in legend.items():
        mask = df["Hogwarts House"] == house
        x_scores = df[mask][x_col].to_numpy()
        y_scores = df[mask][y_col].to_numpy()
        # Remove rows where either score is NaN
        valid = ~(np.isnan(x_scores) | np.isnan(y_scores))
        ax.scatter(x_scores[valid],
                   y_scores[valid],
                   color=color,
                   alpha=0.5,
                   s=scatter_size)
