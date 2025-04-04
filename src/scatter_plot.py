import matplotlib.pyplot as plt
import pandas
import numpy as np
import argparse
from utils.plotting import scatter


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
        description='Display scatter plots comparing course scores',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('dataset', type=str,
                        help='path to dataset CSV file')
    parser.add_argument('-c', '--courses', type=str, nargs=2,
                        choices=courses, metavar=('COURSE1', 'COURSE2'),
                        help='compare scores between two specific courses')
    args = parser.parse_args()

    df = pandas.read_csv(args.dataset)
    return df, args.courses


def main():
    """Program to plot scatter plots of students' scores from given csv.
    Courses "Astronomy" and "Defense against the Dark Arts" have the most
    correlated scores."""

    try:
        df, selected_courses = parse_arguments()

        legend = {'Gryffindor': 'red', 'Hufflepuff': 'yellow',
                  'Ravenclaw': 'blue', 'Slytherin': 'green'}

        course_scores = df.select_dtypes(include=[np.number]).columns[1:]
        n_courses = len(course_scores)

        if selected_courses:
            # Single scatter plot comparing two specified courses
            fig = plt.figure(figsize=(20, 20))
            ax = plt.subplot(111)
            scatter(ax, df, selected_courses[0],
                    selected_courses[1], legend, scatter_size=100)
            ax.set_xlabel(selected_courses[0])
            ax.set_ylabel(selected_courses[1])
        else:
            # Grid of scatter plots comparing each course with others
            fig, axes = plt.subplots(n_courses, n_courses, figsize=(20, 20))
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            for row in range(n_courses):
                for col in range(n_courses):
                    if col < row:
                        plot_num = row * n_courses + col + 1
                        ax = plt.subplot(n_courses, n_courses, plot_num)
                        scatter(ax, df, course_scores[row],
                                course_scores[col], legend)

                        if row == n_courses - 1:  # bottom row
                            ax.set_xlabel(course_scores[col][:15])
                        else:
                            ax.set_xlabel('')
                        if col == 0:  # leftmost column
                            ax.set_ylabel(course_scores[row][:10], fontsize=8)
                        else:
                            ax.set_ylabel('')
                    else:
                        axes[row, col].set_visible(False)
                    plt.xticks([], [])
                    plt.yticks([], [])

            plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
