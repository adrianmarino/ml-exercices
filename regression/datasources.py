import pandas as pd
import quandl as Quandl
from tools import singleton

API_KEY="LSysv2r1X92zcGoeTqPw"

@singleton
class DataSources:
    def google_actions(self):
        df = Quandl.get("WIKI/GOOGL", api_key=API_KEY)
        df = df.rename(columns={
            'Date': 'date',
            'Adj. Open': 'adj_open',
            'Adj. High': 'adj_high',
            'Adj. Low': 'adj_low',
            'Adj. Close': 'adj_close',
            'Adj. Volume': 'adj_volume'
        })
        return df
