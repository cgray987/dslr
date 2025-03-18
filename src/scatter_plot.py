import matplotlib.pyplot as plt
import pandas
import numpy as np
import argparse


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

    plt.xticks([], [])
    plt.yticks([], [])


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
            for i in range(n_courses):
                for j in range(n_courses):
                    if j < i:
                        plot_num = i * n_courses + j + 1
                        ax = plt.subplot(n_courses, n_courses, plot_num)
                        scatter(ax, df, course_scores[i],
                                course_scores[j], legend)

                        if i == n_courses - 1:  # bottom row
                            ax.set_xlabel(course_scores[j][:15])
                        else:
                            ax.set_xlabel('')
                        if j == 0:  # leftmost column
                            ax.set_ylabel(course_scores[i][:10], fontsize=8)
                        else:
                            ax.set_ylabel('')
                    else:
                        axes[i, j].set_visible(False)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
