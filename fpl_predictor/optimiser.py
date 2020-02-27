"""Algorithm for optimising the squad selections to maximise the score on the
chosen scoring metric."""

import itertools
import pandas as pd

from fpl_predictor.functions import verbose_print
from fpl_predictor.squad import AlreadySelected, MaxedOutTeam


def get_perms(raw, r=2):
    """Make all permutations of `r` length combinations of players in the
    DataFrame. Must contain columns: `code`, `score`, `position`, `value`."""
    assert all([c in raw.columns for c in ["code", "score", "position", "value"]]), f"Missing required columns."
    if "score_per_value" not in raw.columns:
        raw["score_per_value"] = raw["score"] / raw["value"]
    codes = list(raw["code"])
    perms = list(itertools.combinations(codes, r=r))
    df = pd.DataFrame(data=perms, columns=[f"p{i}" for i in range(r)])
    scores = dict(zip(raw["code"], raw["score"]))
    score_per_vals = dict(zip(raw["code"], raw["score_per_value"]))
    positions = dict(zip(raw["code"], raw["position"]))
    values = dict(zip(raw["code"], raw["value"]))
    for i in range(r):
        df[f"p{i}_val"] = df[f"p{i}"].map(values)
        df[f"p{i}_score"] = df[f"p{i}"].map(scores)
        df[f"p{i}_score_per_val"] = df[f"p{i}"].map(score_per_vals)
        df[f"p{i}_position"] = df[f"p{i}"].map(positions)
    score_cols = [f"p{i}_score" for i in range(r)]
    val_cols = [f"p{i}_val" for i in range(r)]
    score_per_val_cols = [f"p{i}_score_per_val" for i in range(r)]
    df["score_total"] = df[score_cols].sum(axis=1)
    df["value_total"] = df[val_cols].sum(axis=1)
    df["score_per_value_total"] = df[score_per_val_cols].sum(axis=1)
    position_cols = [f"p{i}_position" for i in range(r)]
    df["positions"] = tuple(zip(*[df[c] for c in position_cols]))
    df["positions"] = [tuple(sorted(p)) for p in df["positions"]]
    player_cols = [c for c in df.columns if "p" in c and len(c) == 2]
    df["players"] = tuple(zip(*[df[c] for c in player_cols]))
    df["players"] = [tuple(sorted(p)) for p in df["players"]]
    return df


def subset_perms(df, positions: tuple, min_score: float, max_val: float,
                 score_per_value: bool = False):
    """Get a subset of a `perms` DataFrame meeting the criteria."""
    score_col = "score_per_value" if score_per_value else "score"
    df = df.loc[(df["positions"] == positions) & (df[f"{score_col}_total"] > min_score) & (df["value_total"] < max_val)]
    return df.sort_values(by=[f"{score_col}_total", "value_total"], ascending=[False, True])


def squad_optimisations(self, year, week, live=False, r=2,
                        score_per_value: bool = False):
    """Get a DataFrame of all possible optimisations of the current squad,
    by removing r-length combinations of players. Note that not all
    optimisations returned may be possible, as the 3-players-per-team limit
    may be met by already selected players."""
    df = pd.DataFrame()
    score_col = "score_per_value" if score_per_value else "score"

    # Get all r-length permutations of the current squad:
    squad_perms = get_perms(self.Squad.selected, r=r)
    squad_perms.sort_values(by=[f"{score_col}_total", "value_total"], ascending=[True, False], inplace=True)
    squad_perms.reset_index(drop=True, inplace=True)

    # Get all possible r-length permutations of players not in the squad:
    pool = self.get_player_pool(year, week, live=live).reset_index()
    pool_perms = get_perms(pool, r=r)

    # Amount of spare budget to add to total available to spend:
    spare_budget = self.Squad.available_budget

    for row_i in squad_perms.iterrows():
        remove = row_i[1].to_dict()
        players_to_remove = remove["players"]
        budget = spare_budget + remove["value_total"]
        better_perms = subset_perms(df=pool_perms, positions=remove["positions"],
                                    min_score=remove[f"{score_col}_total"],
                                    max_val=budget, score_per_value=score_per_value)
        better_perms = better_perms.loc[better_perms["players"] != players_to_remove]
        if len(better_perms):
            s = better_perms.iloc[0].to_dict()
            s["players_to_remove"] = players_to_remove
            s["to_remove_value_total"] = remove["value_total"]
            s[f"to_remove_score_total"] = remove["score_total"]
            s[f"to_remove_score_per_value_total"] = remove["score_per_value_total"]
            df = df.append(s, ignore_index=True)
    if len(df):
        df["value_diff"] = df["value_total"] - df["to_remove_value_total"]
        df["score_diff"] = df["score_total"] - df[f"to_remove_score_total"]
        df["score_per_value_diff"] = df["score_per_value_total"] - df["to_remove_score_per_value_total"]
        df.sort_values(by=[f"{score_col}_diff", "value_diff"], ascending=[False, True], inplace=True)
    return df.reset_index(drop=True)


def optimiser(self, year: int, week: int, iterations: int = 100,
              live: bool = False, r: int = 2,
              score_per_value: bool = False):
    """Optimise the currently selected squad by selecting r-length combinations
    of players in the squad, then comparing them against all r-length
    combinations of non-selected (available) players and seeing if
    substituting the combinations raises the overall team score."""
    score_col = "score_per_value" if score_per_value else "score"
    increase = 0
    original_score = self.Squad.total_score_per_val if score_per_value else self.Squad.total_score

    for i in range(iterations):
        optimisations = squad_optimisations(self, year, week, live=live, r=r, score_per_value=score_per_value)
        if not len(optimisations):
            new_score = self.Squad.total_score_per_val if score_per_value else self.Squad.total_score
            verbose_print("Can't optimise team anymore:"
                          f"\n> Original total {score_col}: {original_score:,.3f}"
                          f"\n> Raised {score_col} by total: {increase:,.3f}"
                          f"\n> New total {score_col}: {new_score:,.3f}")
            return
        for row in optimisations.iterrows():
            opt = row[1].to_dict()
            to_remove = opt["players_to_remove"]

            # Create a copy of the current team to try replacements:
            test = self._test_new_squad
            current_squad = self.Squad.selected.copy()
            test.selected = current_squad.loc[~current_squad["code"].isin(to_remove)]

            # Try to add each new player:
            try:
                for p in range(r):
                    add_player = dict(code=int(opt[f"p{p}"]))
                    add_player["name"] = self.PlayerInformation.player_names(live=live)[add_player["code"]]
                    add_player["position"] = opt[f"p{p}_position"]
                    add_player["value"] = opt[f"p{p}_val"]
                    add_player["score"] = opt[f"p{p}_score"]
                    add_player["score_per_value"] = opt[f"p{p}_score_per_val"]
                    add_player["team"] = self.PlayerInformation.player_teams(year, live=live)[add_player["code"]]
                    test.add_player(**add_player)
                if score_per_value:
                    added = (test.total_score_per_val - self.Squad.total_score_per_val)
                else:
                    added = (test.total_score - self.Squad.total_score)
                verbose_print(f"Increased {score_col} by: {added}")
                increase += added
                self.Squad.selected = test.selected
                break
            except (AlreadySelected, MaxedOutTeam):
                continue
