
import numpy as np
import pandas as pd

from fpl_predictor.squad_builder import SquadBuilder
from fpl_predictor.squad import MaxedOutTeam


def build(self: SquadBuilder, year: int, week: int, live=False, **kwargs):
    """Build a team from scratch using the cheapest possible players. Probably
    not the optimum strategy."""
    self.squad.remove_all_players()

    # Get all available players:
    df = self._player_pool(year, week, live=live)

    # Sort by value ascending, score descending:
    df.sort_values(by=["value", "score"], ascending=[True, False], inplace=True)

    # Just keep adding players!
    while not self.squad.squad_full:
        df = df.loc[(df["position"].isin(self.squad.need_positions)) &
                    (~df["team"].isin(self.squad.maxed_out_teams))]
        row = df.reset_index().iloc[0].to_dict()
        self.squad.add_player(**row)
        df = df.iloc[1:]

    return self.squad.selected


def evaluate_transfers(self, year, week, live=False, percent_chance=100,
                       unavailable_first=True):
    """Evaluate transfers on the current team for the given year-week."""
    df = pd.DataFrame()
    spare_budget = self.squad.available_budget
    for row in self.squad.selected.iterrows():
        player = row[1].to_dict()
        pool = self._player_pool(year, week, live=live, position=player["position"],
                                 min_score=player["score"], max_val=spare_budget + player["value"])
        if len(pool):
            old_player = {f"out_{k}": v for k, v in player.items()}
            transfer = self._select_player(pool)
            transfer = {**old_player, **transfer}
            df = df.append(transfer, ignore_index=True)
    df["score_gain"] = df["score"] - df["out_score"]
    df["score_per_val_gain"] = df["score_per_value"] - df["out_score_per_value"]
    df["val_gain"] = df["out_value"] - df["value"]
    df.sort_values(by=["score_gain", "score_per_val_gain"], ascending=False, inplace=True)

    # Prioritize current first_team/squad who are unavailable.
    squad_unavailable = self.squad_unavailable(year, week, live=live, percent_chance=percent_chance)
    first_team_unavailable = self.first_team_unavailable(year, week, live=live, percent_chance=percent_chance)
    df["squad_unavailable"] = np.where(df["out_code"].isin(squad_unavailable), True, False)
    df["first_team_unavailable"] = np.where(df["out_code"].isin(first_team_unavailable), True, False)
    if unavailable_first:
        df.sort_values(by=["first_team_unavailable", "squad_unavailable"], ascending=False, inplace=True)

    return df.reset_index(drop=True)


def make_tranfers(self: SquadBuilder, n_transfers: int,
                  transfer_list: pd.DataFrame):
    """Attempt to make the transfers returned by the `evaluate_transfers`
    function to the currently selected squad.

    Args:
        self (SquadBuilder): instance of the SquadBuilder class.
        n_transfers (int): total number of transfers to make to squad. Note it
            may not be possible to make the total number of transfers.
        transfer_list (pd.DataFrame): output from the `evaluate_transfers`
            function.
    """
    n_made = 0
    for t in range(n_transfers):
        if not len(transfer_list):
            break
        for row in transfer_list.iterrows():
            try:
                p = row[1]
                test_squad = self._test_new_squad
                test_squad.selected = self.squad.selected
                remove_code = p["out_code"]
                add_code = p["code"]
                test_squad.remove_player(remove_code)
                test_squad.add_player(code=p["code"], name=p["name"], position=p["position"],  value=p["value"],
                                      score=p["score"], score_per_value=p["score_per_value"], team=p["team"])
                transfer_list = transfer_list.loc[transfer_list["code"] != add_code]
                transfer_list = transfer_list.loc[transfer_list["out_code"] != remove_code]
                self.squad.selected = test_squad.selected
                n_made += 1
                break
            except MaxedOutTeam:
                continue

    return n_made
