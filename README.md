# dslr
- Advanced Curriculum 42 Datascience project

# Project Objectives
- Visualize and classify a dataset
- Classify using logistic regression and gradient descent
- Learn more about Pandas/Numpy/Scikit
    - Recreate Scikit Logistic Regression functionality
    - Recreate pandas .describe() functionality

# Usage
###    - requirements
- Python 3.x
- pandas/numpy/matplotlib pyplot/argparse
###    - Visualization
- describe
    - python describe.py dataset [--compare]
- histogram
    - python histogram.py [-h] [-c COURSE] dataset
- scatter plot
    - python scatter_plot.py [-h] [-c COURSE1 COURSE2] dataset
- pair plot
    - python pair_plot.py [-h] dataset
###    - Prediction
- Training (Create data weights)
    - logreg_train.py [-h] [-c] [-o {batch,stochastic,minibatch}] [-n] dataset
    - positional arguments:
        - dataset - path to training dataset CSV file

    - options:
        - **-h**, --help            show this help message and exit
        - **-c**, --confusion       display a confusion matrix from test data
        - **-o** {batch,stochastic,minibatch}, --optimizer {batch,stochastic,minibatch}
                        Choose optimization algorithm
        - **-n**, --normalization   Use normalization rather than standardizationfor fitting algorithm
- Predictions (Create house predictions)
    - logreg_predict.py [-h] [-n] dataset weights

    - positional arguments:
        - dataset - path to test dataset CSV file
        - weights - path to training weights file

    - options:
        - **-h**, --help           show this help message and exit
        - **-n**, --normalization  Use normalization rather than standardization for fitting algorithm

# references
https://web.archive.org/web/20180618211933/http://cs229.stanford.edu/notes/cs229-notes1.pdf
https://scikit-learn.org/stable/modules/sgd.html
https://en.wikipedia.org/wiki/Logistic_regression