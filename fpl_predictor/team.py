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
        self.team = pd.DataFrame(columns=["uuid", "name", "position", "value",
                                          "score", "roi_score"])
        fp = os.path.join(DIR_DATA, "players_master.csv")
        df = pd.read_csv(fp, encoding="utf-8")
        df["name"] = df["first_name"].str.cat(df["second_name"], sep=" ")
        self.players_master = df
        self.player_names = dict(zip(df["uuid"], df["name"]))
        self.player_positions = dict(zip(df["uuid"], df["position"]))

    def clear_team(self):
        self.team = pd.DataFrame(columns=["uuid", "name", "position", "value",
                                          "score", "roi_score"])

    @property
    def total_score(self):
        return self.team["score"].sum()

    @property
    def total_points_over_period(self):
        return self.team["score"].sum()

    @property
    def total_value(self):
        return self.team["value"].sum()

    @property
    def available_budget(self):
        return 1000 - self.total_value

    @property
    def selected_players(self):
        return sorted(self.team["uuid"])

    def add_player(self, uuid: str, value: float, score: float, roi_score: float):
        position = self.player_positions[uuid]
        new_player = {"uuid": uuid, "value": value, "position": position,
                      "score": score, "roi_score": roi_score,
                      "name": self.player_names[uuid]}
        if len(self.team.loc[self.team["position"] == position]) == self.limits[position]:
            raise PositionFilled()
        if self.total_value + value > self.cap:
            raise NotEnoughMoney()
        if uuid in self.selected_players:
            raise AlreadySelected()
        self.team = self.team.append(new_player, ignore_index=True)

    def remove_player(self, uuid):
        if uuid not in list(self.team["uuid"]):
            raise ValueError(f"uuid not in team: {uuid}")
        self.team = self.team.loc[self.team["uuid"] != uuid]

    @property
    def filled(self):
        return True if len(self.team) == 15 else False

    @property
    def filled_by_position(self):
        filled = dict(self.team["position"].value_counts())
        template = {"FWD": 0, "MID": 0, "DEF": 0, "GK": 0}
        return {**template, **filled}

    @property
    def empty_by_position(self):
        filled = self.filled_by_position
        return {k: v - filled[k] for k, v in self.limits.items()}

    @property
    def need_positions(self):
        return [k for k, v in self.empty_by_position.items() if v]

