import pandas as pd
import quandl as Quandl


def singleton(cls):
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance


@singleton
class DataSources:
    def remote_google_actions(self):
        df = Quandl.get("WIKI/GOOGL")
        df = df.rename(columns={
            'Date': 'date',
            'Adj. Open': 'adj_open',
            'Adj. High': 'adj_high',
            'Adj. Low': 'adj_low',
            'Adj. Close': 'adj_close',
            'Adj. Volume': 'adj_volume'
        })
        return df

    def local_google_actions(self):
        df = pd.read_csv("./WIKI-PRICES.csv", index_col=1)
        df.index.names = ['Date']
        return df

    def google_actions(self, local=True):
        return self.local_google_actions() if local else self.remote_google_actions()