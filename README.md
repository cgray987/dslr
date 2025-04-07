# dslr
    - Advanced Curriculum 42 Datascience project

# Project Objectives
    - Visualize and classify a dataset
    - Classify using logistic regression and gradient descent
    - Learn more about Pandas/Numpy/Scikit
        - Recreate Scikit Logistic Regression functionality
        - Recreate pandas .describe() functionality

# Usage
    - requirements
        - Python 3.x
        - pandas/numpy/matplotlib pyplot/argparse
    - Visualization
        - describe
            python describe.py datasets/dataset_train.csv [--compare]
        - histogram
            python histogram.py [-h] [-c COURSE] dataset
        - scatter plot
            python scatter_plot.py [-h] [-c COURSE1 COURSE2] dataset
        - pair plot
            python pair_plot.py [-h] dataset
    - Prediction
        - Training (Create data weights)
            python logreg_train.py [-h] [-c] dataset
        - Predictions (Create house predictions)
            python logreg_predict.py [-h] dataset weights

# references
https://web.archive.org/web/20180618211933/http://cs229.stanford.edu/notes/cs229-notes1.pdf
https://scikit-learn.org/stable/modules/sgd.html
https://en.wikipedia.org/wiki/Logistic_regression