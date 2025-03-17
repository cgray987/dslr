import matplotlib.pyplot as plt
import pandas
import numpy as np
import argparse
import histogram as h
import scatter_plot as s


def parse_arguments():
    """Handle command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Display a grid of pair plots comparing course scores',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('dataset', type=str,
                        help='path to dataset CSV file')
    args = parser.parse_args()

    df = pandas.read_csv(args.dataset)
    return df

def main():
    """Program to plot scatter plots of students' scores from given csv.
    Courses "Astronomy" and "Defense against the Dark Arts" have the most
    correlated scores."""

    # try:
    df = parse_arguments()

    legend = {'Gryffindor': 'red', 'Hufflepuff': 'yellow',
              'Ravenclaw': 'blue', 'Slytherin': 'green'}

    course_scores = df.select_dtypes(include=[np.number]).columns[1:]
    n_courses = len(course_scores)

    # Grid of scatter plots comparing each course with others
    fig, axes = plt.subplots(n_courses, n_courses, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i in range(n_courses):
        for j in range(n_courses):
            plot_num = i * n_courses + j + 1
            ax = plt.subplot(n_courses, n_courses, plot_num)
            if i == j:
                # pass
                h.histogram(ax, df, course_scores[i], legend, False)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.get_legend().remove()
                ax.set_title("")
                plt.xticks([], [])
                plt.yticks([], [])
            else:
                s.scatter(ax, df, course_scores[i],
                          course_scores[j], legend)

            if i == n_courses - 1:  # bottom row
                ax.set_xlabel(course_scores[j][:15])
            else:
                ax.set_xlabel('')
            if j == 0:  # leftmost column
                ax.set_ylabel(course_scores[i][:10], fontsize=8)
            else:
                ax.set_ylabel('')

    plt.tight_layout()
    plt.show()

    # except Exception as e:
    #     print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
