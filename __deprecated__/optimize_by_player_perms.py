
import itertools
import pandas as pd

from fpl_predictor.squad_builder import SquadBuilder
from fpl_predictor.squad import PositionFilled, AlreadySelected, MaxedOutTeam


class Optimize(SquadBuilder):

    def __init__(self, *args, r=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r
        all_codes = sorted(set(self.player_pool().index))

        self.all_perms = self.make_player_perms(*all_codes, r=r)
        self._optimisations_tried = list()

    def make_player_perms(self, *codes, r=3):
        perms = list(itertools.combinations(codes, r=r))
        df = pd.DataFrame(data=perms, columns=[f"p{i}" for i in range(r)])
        scores = dict(zip(self.scores.index, self.scores["raw_score"]))
        positions = self._scoring_data.positions[self.year]
        for i in range(r):
            df[f"p{i}_val"] = df[f"p{i}"].map(self._player_values)
            df[f"p{i}_score"] = df[f"p{i}"].map(scores)
            df[f"p{i}_position"] = df[f"p{i}"].map(positions)
        score_cols = [f"p{i}_score" for i in range(r)]
        val_cols = [f"p{i}_val" for i in range(r)]
        df["score_total"] = df[score_cols].sum(axis=1)
        df["value_total"] = df[val_cols].sum(axis=1)
        position_cols = [f"p{i}_position" for i in range(r)]
        df["position"] = tuple(zip(*[df[c] for c in position_cols]))
        df["position"] = [tuple(sorted(p)) for p in df["position"]]
        player_cols = [c for c in df.columns if "p" in c and len(c) == 2]
        df["ballers"] = tuple(zip(*[df[c] for c in player_cols]))
        df["ballers"] = [tuple(sorted(p)) for p in df["ballers"]]
        return df

    def subset_all_perms(self, position_combo: tuple, min_score_total: float,
                         max_total_value: float):
        """Get a subset of the `all_perms` attribute meeting the criteria."""
        df = self.all_perms.copy()
        df = df.loc[(df["position"] == position_combo) &
                    (df["score_total"] > min_score_total) &
                    (df["value_total"] < max_total_value)]
        return df.sort_values(by=["score_total", "value_total"], ascending=[False, True])

    @property
    def test_new_team(self):
        return TeamBuilder(scoring_data=self._scoring_data, year=self.year,
                           week=self.week, n=self.n,
                           min_minute_percent=self.min_minute_percent,
                           cross_seasons=self.cross_seasons,
                           agg_func=self.agg_func,
                           n_by_raw_score=self.n_by_raw_score,
                           pick_team_by=self.pick_team_by,
                           live_data=self.live_data)

    def optimize(self):
        codes = sorted(self.team.sqaud["code"])
        team_perms = self.make_player_perms(*codes, r=self.r)
        team_perms.sort_values(by=["score_total", "value_total"], ascending=[True, False], inplace=True)
        team_perms.reset_index(drop=True, inplace=True)
        team_perms = team_perms.loc[~team_perms["ballers"].isin(self._optimisations_tried)]
        squad = self.team.sqaud.copy()
        av_budget = self.team.available_budget
        old_score = self.team.total_score
        for count, row_i in enumerate(team_perms.iterrows()):
            remove_players = row_i[1].to_dict()
            ballers_to_remove = remove_players["ballers"]
            budget = av_budget + remove_players["value_total"]
            better_perms = self.subset_all_perms(remove_players["position"],
                                                 remove_players["score_total"],
                                                 budget)
            better_perms = better_perms.loc[better_perms["ballers"] != ballers_to_remove]
            if len(better_perms):
                for row_j in better_perms.iterrows():
                    test = self.test_new_team
                    test.team.sqaud = squad.loc[~squad["code"].isin(ballers_to_remove)]
                    try:
                        p2 = row_j[1].to_dict()
                        ballers_to_add = p2["ballers"]
                        for b in ballers_to_add:
                            test.add_player(b)

                        new_score = test.team.total_score
                        if new_score > old_score:
                            self.team.sqaud = test.team.sqaud
                            print(f"Increased score by {new_score - old_score:,.3f}")
                            return new_score - old_score
                    except (PositionFilled, AlreadySelected, MaxedOutTeam) as e:
                        continue

            self._optimisations_tried.append(ballers_to_remove)
        print("Unable to increase score.")
        return None

    def get_current_team_optimisations(self):
        """Get a DataFrame of all possible optimisations for the current team.
        Note that not all optimisations returned are possible, as the 3-ballers-
        per-team limit may be met by already selected players."""
        df = pd.DataFrame()
        codes = sorted(self.team.sqaud["code"])
        team_perms = self.make_player_perms(*codes, r=self.r)
        team_perms.sort_values(by=["score_total", "value_total"], ascending=[True, False], inplace=True)
        team_perms.reset_index(drop=True, inplace=True)
        av_budget = self.team.available_budget

        for row_i in team_perms.iterrows():
            remove_players = row_i[1].to_dict()
            ballers_to_remove = remove_players["ballers"]
            budget = av_budget + remove_players["value_total"]
            better_perms = self.subset_all_perms(remove_players["position"],
                                                 remove_players["score_total"],
                                                 budget)
            better_perms = better_perms.loc[better_perms["ballers"] != ballers_to_remove]
            if len(better_perms):
                s = better_perms.iloc[0].to_dict()
                s["ballers_to_remove"] = ballers_to_remove
                s["to_remove_value_total"] = remove_players["value_total"]
                s["to_remove_score_total"] = remove_players["score_total"]
                df = df.append(s, ignore_index=True)
        df = df[["position", "ballers", "score_total", "value_total",
                 "ballers_to_remove", "to_remove_score_total", "to_remove_value_total"]]
        df["value_diff"] = df["value_total"] - df["to_remove_value_total"]
        df["score_diff"] = df["score_total"] - df["to_remove_score_total"]
        df.sort_values(by=["score_diff", "value_diff"], ascending=[False, True], inplace=True)
        return df.reset_index(drop=True)

    def run(self):
        while True:
            o = self.optimize()
            if not o:
                break

