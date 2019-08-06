import os
import pandas as pd

from nav import DIR_DATA


class NotEnoughMoney(Exception):
    pass


class PositionFilled(Exception):
    pass


class AlreadySelected(Exception):
    pass


class Team:
    limits = {"FWD": 3, "MID": 5, "DEF": 5, "GK": 2}
    cap = 1000

    def __init__(self):
        self.__team = pd.DataFrame(columns=["uuid", "position", "value", "score", "roi_score"])
        fp = os.path.join(DIR_DATA, "players_master.csv")
        df = pd.read_csv(fp, encoding="utf-8")
        df["name"] = df["first_name"].str.cat(df["second_name"], sep=" ")
        self.players_master = df
        self.player_names = dict(zip(df["uuid"], df["name"]))
        self.player_positions = dict(zip(df["uuid"], df["position"]))

    @property
    def total_value(self):
        return self.__team["value"].sum()

    @property
    def selected_players(self):
        return sorted(self.__team["uuid"])

    def add_player(self, uuid: str, value: float, score: float, roi_score: float):
        position = self.player_positions[uuid]
        new_player = {"uuid": uuid, "value": value, "position": position,
                      "score": score, "roi_score": roi_score}
        if len(self.team.loc[self.team["position"] == position]) == self.limits[position]:
            raise PositionFilled()
        if self.total_value + value > self.cap:
            raise NotEnoughMoney()
        if uuid in self.selected_players:
            raise AlreadySelected()
        self.__team = self.__team.append(new_player, ignore_index=True)

    @property
    def team(self):
        df = self.__team.reset_index(drop=True)
        df["name"] = df["uuid"].map(self.player_names)
        return df

    @property
    def filled(self):
        if len(self.__team) == 15:
            return True
        else:
            return False

