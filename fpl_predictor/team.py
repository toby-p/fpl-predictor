
import pandas as pd


class NotEnoughMoney(Exception):
    pass


class PositionFilled(Exception):
    pass


class AlreadySelected(Exception):
    pass


class MaxedOutTeam(Exception):
    pass


class Team:
    limits = {"FWD": 3, "MID": 5, "DEF": 5, "GK": 2}
    cap = 1000
    columns = ["code", "name", "position", "value", "raw_score", "roi_score", "team"]

    def __init__(self):
        self.sqaud = pd.DataFrame(columns=self.columns)

    def clear_team(self):
        self.sqaud = pd.DataFrame(columns=self.columns)

    @property
    def total_raw_score(self):
        return self.sqaud["raw_score"].sum()

    @property
    def total_roi_score(self):
        return self.sqaud["roi_score"].sum()

    @property
    def total_points_over_period(self):
        return self.sqaud["raw_score"].sum()

    @property
    def total_value(self):
        return self.sqaud["value"].sum()

    @property
    def available_budget(self):
        return 1000 - self.total_value

    @property
    def selected_players(self):
        return dict(zip(self.sqaud["code"], self.sqaud["name"]))

    def add_player(self, code: str, name: str, position: str, value: float,
                   raw_score: float, roi_score: float, team: str, **kwargs):
        new_player = {"code": code, "name": name, "position": position, "team": team,
                      "value": value, "raw_score": raw_score, "roi_score": roi_score}
        if len(self.sqaud.loc[self.sqaud["position"] == position]) == self.limits[position]:
            raise PositionFilled()
        if self.total_value + value > self.cap:
            raise NotEnoughMoney()
        if code in self.selected_players:
            raise AlreadySelected()
        if team in self.maxed_out_teams:
            raise MaxedOutTeam()
        self.sqaud = self.sqaud.append(new_player, ignore_index=True)

    def remove_player(self, code):
        if code not in list(self.sqaud["code"]):
            raise ValueError(f"code not in team: {code}")
        self.sqaud = self.sqaud.loc[self.sqaud["code"] != code]

    @property
    def filled(self):
        return True if len(self.sqaud) == 15 else False

    @property
    def filled_by_position(self):
        filled = dict(self.sqaud["position"].value_counts())
        template = {"FWD": 0, "MID": 0, "DEF": 0, "GK": 0}
        return {**template, **filled}

    @property
    def empty_by_position(self):
        filled = self.filled_by_position
        return {k: v - filled[k] for k, v in self.limits.items()}

    @property
    def need_positions(self):
        return [k for k, v in self.empty_by_position.items() if v]

    @property
    def maxed_out_teams(self):
        """List of teams from which 3 players have already been selected."""
        df = self.sqaud.copy()
        if len(df) < 3:
            return list()
        team_count = df["team"].value_counts().to_dict()
        return [k for k, v in team_count.items() if v >= 3]
