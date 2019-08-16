
import numpy as np
import pandas as pd
import requests
import warnings


class ApiData:
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"

    def __init__(self):
        self.data = dict()
        self.elements = pd.DataFrame()
        self.get()

    def get(self):
        """Get the latest data from the fantasy API."""
        r = requests.get(ApiData.url)
        self.data = r.json()
        self.elements = pd.DataFrame(self.data["elements"])

    @property
    def current_season(self):
        year = self.current_year
        return f"{year}-{(year - 2000) + 1}"

    @property
    def current_year(self):
        return int(self.data["events"][0]["deadline_time"][:4])

    @property
    def player_ids(self):
        """Dictionary of player IDs to full name from the current API data."""
        if not len(self.elements):
            warnings.warn("No API data.")
            return dict()
        df = self.elements.copy()
        df["name"] = df["first_name"].str.cat(df["second_name"], sep=" ")
        return dict(zip(df["id"], df["name"]))

    @property
    def current_values(self):
        return dict(zip(self.elements["code"], self.elements["now_cost"]))

    def players_available(self, percent_chance=100):
        """Return list of player codes available, either NaN or with chance
        greater or equal than `percent_chance`."""
        df = self.elements[["code", "chance_of_playing_next_round"]].copy()
        df["available"] = np.where(df["chance_of_playing_next_round"] < percent_chance, "False", np.nan)
        df["available"] = df["available"].replace("False", False)
        df["available"] = df["available"].replace("nan", True)
        return dict(zip(df["code"], df["available"]))

