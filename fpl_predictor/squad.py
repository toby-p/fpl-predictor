
import os
import pandas as pd

from nav import DIR_SQUADS


class NotEnoughMoney(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"{self.msg}"


class PositionFilled(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"{self.msg}"


class AlreadySelected(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"{self.msg}"


class MaxedOutTeam(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"{self.msg}"


class Squad:
    limits = {"FWD": 3, "MID": 5, "DEF": 5, "GK": 2}
    cap = 1000
    columns = ["code", "name", "position", "value", "score", "score_per_value", "team"]

    def __init__(self):
        """Class for tracking a fantasy squad."""
        self.selected = pd.DataFrame(columns=self.columns)

    def remove_all_players(self):
        self.selected = pd.DataFrame(columns=self.columns)

    @property
    def total_score(self):
        return self.selected["score"].sum()

    @property
    def total_score_per_val(self):
        return self.selected["score_per_value"].sum()

    @property
    def total_value(self):
        return self.selected["value"].sum()

    @property
    def available_budget(self):
        return 1000 - self.total_value

    @property
    def _selected_list(self):
        return sorted(self.selected["code"])

    def add_player(self, code: str, name: str, position: str, value: float,
                   score: float, score_per_value: float, team: str, **kwargs):
        new_player = {"code": code, "name": name, "position": position, "team": team,
                      "value": value, "score": score, "score_per_value": score_per_value}
        if len(self.selected.loc[self.selected["position"] == position]) == self.limits[position]:
            raise PositionFilled(new_player)
        if self.total_value + value > self.cap:
            raise NotEnoughMoney(new_player)
        if code in self._selected_list:
            raise AlreadySelected(new_player)
        if team in self.maxed_out_teams:
            raise MaxedOutTeam(new_player)
        self.selected = self.selected.append(new_player, ignore_index=True).reset_index(drop=True)

    def remove_player(self, code):
        if code not in list(self.selected["code"]):
            raise ValueError(f"code not in team: {code}")
        self.selected = self.selected.loc[self.selected["code"] != code].reset_index(drop=True)
        print(f"Removed player {code} from squad, {len(self.selected)} remain.")

    @property
    def squad_full(self):
        return True if len(self.selected) == 15 else False

    @property
    def full_by_position(self):
        full = dict(self.selected["position"].value_counts())
        template = {"FWD": 0, "MID": 0, "DEF": 0, "GK": 0}
        return {**template, **full}

    @property
    def empty_by_position(self):
        full = self.full_by_position
        return {k: v - full[k] for k, v in self.limits.items()}

    @property
    def need_positions(self):
        return [k for k, v in self.empty_by_position.items() if v]

    @property
    def maxed_out_teams(self):
        """List of teams from which 3 players have already been selected."""
        df = self.selected.copy()
        if len(df) < 3:
            return list()
        team_count = df["team"].value_counts().to_dict()
        return [k for k, v in team_count.items() if v >= 3]

    def save_squad(self, filename):
        fp = os.path.join(DIR_SQUADS, f"{filename}.csv")
        self.selected.to_csv(fp, encoding="utf-8", index=False)
        print(f"Squad of {len(self.selected)} players saved at: {fp}")

    def load_squad(self, filename):
        fp = os.path.join(DIR_SQUADS, f"{filename}.csv")
        assert os.path.isfile(fp), f"Squad file not found: {filename}"
        df = pd.read_csv(fp, encoding="utf-8")
        self.selected = df
        print(f"Squad of {len(self.selected)} players loaded from {fp}")
