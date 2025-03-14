import pandas


class Data:
    """cleans given dataframe by removing index column and
    rows with NaN values"""
    def __init__(self, df: pandas.DataFrame):
        self.df = df
        # self.df.drop(columns=['Index'], inplace=True)
        # self.df.dropna(axis=1, how='all', inplace=True)
        # self.df.dropna(inplace=True)

        self.df.columns = df.columns.str.lower()
        self.df.columns = df.columns.str.replace(' ', '_')

    @property
    def df_scores(self):
        return self.df.loc[:, 'arithmancy':'flying']
