import matplotlib.pyplot as plt
import pandas
import numpy as np
import argparse
import utils.dslr_math as dmath


def parse_arguments():
    """Handle command line argument parsing"""

    courses = [
        'Arithmancy', 'Astronomy', 'Herbology',
        'Defense Against the Dark Arts', 'Divination',
        'Muggle Studies', 'Ancient Runes', 'History of Magic',
        'Transfiguration', 'Potions', 'Care of Magical Creatures',
        'Charms', 'Flying'
    ]
    parser = argparse.ArgumentParser(
        description='Display histograms from dataset',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('dataset', type=str,
                        help='path to dataset CSV file')
    parser.add_argument('-c', '--course', type=str,
                        choices=courses,
                        metavar='COURSE',
                        help='show marks for specific course\n' +
                        ', '.join(f'{c}' for c in courses))
    args = parser.parse_args()

    df = pandas.read_csv(args.dataset)
    return df, args.course


def histogram(ax, df, value_col, legend, stats=False):
    """Plots histogram of given course grades. Also prints stats of
    said grades per house."""

    if stats: print(f"Stats for {value_col}:")
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
            print(f"\t{house:10} | max: {dmath.max(house_scores):8.1f} "
                  f"| min: {dmath.min(house_scores):8.1f} "
                  f"| mean: {dmath.mean(house_scores):8.1f} "
                  f"| std: {dmath.std(house_scores):8.1f}")

    ax.set_xlabel('Grade')
    ax.set_title(value_col)
    ax.legend(legend, loc='upper right', frameon=False)


def main():
    """Program to plot histograms of students' scores from given csv.
    Care of Magical Creatures has the most uniform score distribution."""

    try:
        df, course = parse_arguments()

        data = df.to_numpy()
        data = data[1:][data[1:, 1].argsort()]  # Sort by house column

        plt.figure(figsize=(20, 10))
        legend = {'Gryffindor': 'red', 'Hufflepuff': 'yellow',
                  'Ravenclaw': 'blue', 'Slytherin': 'green'}
        course_scores = df.columns[6:]

        # Create histogram for single plot
        if course:
            ax = plt.subplot(111)
            histogram(ax, df, course, legend, True)
            ax.set_ylabel('Number of students')
        # Plot histogram for every course
        else:
            for i, course in enumerate(course_scores):
                ax = plt.subplot(3, 5, i + 1)
                histogram(ax, df, course, legend, True)

                if i % 5 == 0:  # leftmost plots
                    ax.set_ylabel('Number of students')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
