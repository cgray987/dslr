import matplotlib.pyplot as plt
import pandas
import argparse
from utils.plotting import histogram


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


def main():
    """Program to plot histograms of students' scores from given csv.
    Care of Magical Creatures has the most uniform score distribution."""

    try:
        df, course = parse_arguments()

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
