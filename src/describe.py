import pandas
import numpy as np
import utils.dslr_math as dslr_math
import argparse


def describe(df):
    """Prints table of statistical calculations on given dataframe"""
    numeric_df = df.select_dtypes(include=[np.number])

    print(f"{'':5}", end=" ")
    for col in numeric_df.columns:
        print(f"|{col[0:12]:^12}", end=" ")
    # width of row above
    width = 5 + sum(2 + 12 for _ in numeric_df.columns)
    print()
    print("-" * width)

    stats = {
        'Count': lambda x: dslr_math.count(x),
        'Mean': lambda x: dslr_math.mean(x),
        'Std': lambda x: dslr_math.std(x),
        'Min': lambda x: dslr_math.min(x),
        '25%': lambda x: dslr_math.quantile(x, 25),
        '50%': lambda x: dslr_math.quantile(x, 50),
        '75%': lambda x: dslr_math.quantile(x, 75),
        'Max': lambda x: dslr_math.max(x)
    }
    for stat_name, stat_func in stats.items():
        print(f"{stat_name:5}", end=" ")
        for col in numeric_df.columns:
            data = numeric_df[col]
            val = stat_func(data)
            print(f"|{val:12.4f}", end=" ")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Describe numerical statistics of a CSV dataset'
    )
    parser.add_argument('dataset', type=str, help='path to dataset CSV file')
    parser.add_argument('-c', '--compare', action='store_true',
                        help='compare with pandas describe() output')

    try:
        args = parser.parse_args()
        df = pandas.read_csv(args.dataset)
        describe(df)

        if args.compare:
            print("\nPandas describe() output:")
            print(df.describe())
    except Exception as e:
        print(f"{Exception.__name__}: {e}")


if __name__ == "__main__":
    main()
