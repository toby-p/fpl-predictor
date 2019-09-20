
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
    def selected_list(self):
        return sorted(self.selected["code"])

    @property
    def substitutes(self):
        df = self.selected.loc[~self.selected["code"].isin(list(self.first_team["code"]))]
        return df.reset_index(drop=True)

    def add_player(self, code: str, name: str, position: str, value: float,
                   score: float, score_per_value: float, team: str, **kwargs):
        new_player = {"code": int(code), "name": name, "position": position, "team": team,
                      "value": value, "score": score, "score_per_value": score_per_value}
        if len(self.selected.loc[self.selected["position"] == position]) == self.limits[position]:
            raise PositionFilled(new_player)
        if self.total_value + value > self.cap:
            raise NotEnoughMoney(new_player)
        if code in self.selected_list:
            raise AlreadySelected(new_player)
        if team in self.maxed_out_teams:
            raise MaxedOutTeam(new_player)
        self.selected = self.selected.append(new_player, ignore_index=True)
        self.selected.sort_values(by=["score", "score_per_value", "value"],
                                  ascending=[False, False, True], inplace=True)
        self.selected.reset_index(drop=True, inplace=True)

    def remove_player(self, code):
        if code not in list(self.selected["code"]):
            raise ValueError(f"code not in team: {code}")
        name = self.selected.loc[self.selected["code"] == code, "name"].values[0]
        self.selected = self.selected.loc[self.selected["code"] != code].reset_index(drop=True)
        print(f"Removed player from squad ({len(self.selected)} remain): {name}")

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

    @property
    def first_team(self):
        if not self.squad_full:
            print("Squad not full, can't pick first team.")
            return
        df = self.selected
        df.sort_values(by=["score", "score_per_value", "value"], ascending=[False, False, True], inplace=True)

        first_team = pd.DataFrame()

        # First pick a keeper:
        keeper = df.loc[df["position"] == "GK", :].iloc[0].to_dict()
        first_team = first_team.append(keeper, ignore_index=True)
        df = df.loc[df["position"] != "GK"]

        # Then pick the rest of the team:
        max_picks = {"DEF": 5, "MID": 5, "FWD": 3}
        picked = {"DEF": 0, "MID": 0, "FWD": 0}
        for row in df.iterrows():
            if len(first_team) == 11:
                break
            player = row[1].to_dict()
            if picked[player["position"]] < max_picks[player["position"]]:
                first_team = first_team.append(player, ignore_index=True)
                picked[player["position"]] += 1
            # Can't have 5 midfielders AND 5 defenders:
            if picked["DEF"] == 5:
                max_picks["MID"] = 4
            if picked["MID"] == 5:
                max_picks["DEF"] = 4

        first_team["code"] = first_team["code"].astype(int)
        return first_team

    @property
    def total_first_team_score(self):
        if not self.squad_full:
            print("Squad not full.")
            return
        return self.first_team["score"].sum()

    @property
    def total_first_team_score_per_val(self):
        if not self.squad_full:
            print("Squad not full.")
            return
        return self.first_team["score_per_value"].sum()

    @property
    def total_first_team_value(self):
        if not self.squad_full:
            print("Squad not full.")
            return
        return self.first_team["value"].sum()

    @property
    def captain(self):
        if not self.squad_full:
            print("Squad not full, can't pick captain.")
            return
        else:
            df = self.selected
            df.sort_values(by=["score", "score_per_value", "value"], ascending=[False, False, True], inplace=True)
            captain = df.iloc[0]
        return {captain["code"]: captain["name"]}

    @property
    def vice_captain(self):
        if not self.squad_full:
            print("Squad not full, can't pick vice-captain.")
            return
        else:
            df = self.selected
            df.sort_values(by=["score", "score_per_value", "value"], ascending=[False, False, True], inplace=True)
            vc = df.iloc[1]
        return {vc["code"]: vc["name"]}
