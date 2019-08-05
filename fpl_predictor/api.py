import pandas as pd
import requests


class ApiData:
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"

    def __init__(self):
        self.data = self.get()
        self.elements = pd.DataFrame(self.data["elements"])

    @staticmethod
    def get():
        """Get the latest data from the fantasy API.
        """
        r = requests.get(ApiData.url)
        return r.json()

    @property
    def current_season(self):
        year = self.current_year
        return f"{year}-{(year - 2000) + 1}"

    @property
    def current_year(self):
        return int(self.data["events"][0]["deadline_time"][:4])

    @property
    def player_ids(self):
        """Dictionary of player IDs to full name from the current API data.
        """
        df = self.elements.copy()
        df["name"] = df["first_name"].str.cat(df["second_name"], sep=" ")
        return dict(zip(df["id"], df["name"]))
