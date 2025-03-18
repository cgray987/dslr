import describe as d
import histogram as h
import pair_plot as p
import scatter_plot as s
import argparse


def main():
    """Runs each script at once"""
    parser = argparse.ArgumentParser(
        description="42 DSLR Project"
        )
    parser.add_argument('dataset', type=str, help='path to dataset CSV file')
    try:
        d.main()
        h.main()
        s.main()
        p.main()
    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
