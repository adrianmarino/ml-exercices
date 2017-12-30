import quandl as Quandl
from tools import singleton

API_KEY="LSysv2r1X92zcGoeTqPw"

@singleton
class DataSources:
    def google_actions(self): return Quandl.get("WIKI/GOOGL", api_key=API_KEY)