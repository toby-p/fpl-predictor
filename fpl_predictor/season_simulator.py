
import pandas as pd

from fpl_predictor.dataloader import DataLoader
from fpl_predictor.team import AlreadySelected


class SeasonSimulator:

    def __init__(self, year, strategy):
        self.year = year
        self.strategy = strategy
        self.dataloader = DataLoader(make_master=False, build_player_db=False)
        self.transfers = pd.DataFrame()
        self.team_points = pd.DataFrame()

    def simulate(self, verbose=True):
        self.transfers = pd.DataFrame(columns=["GW", "out", "in"])
        self.team_points = pd.DataFrame(columns=["GW", "total_points"])
        current_team = self.strategy.pick_first_gameweek_team(self.year)
        current_team.build_team()

        # Score the team:
        uuids = sorted(current_team.team.team["uuid"])
        score = self.score_team(self.year, 2, *uuids)
        self.team_points = self.team_points.append({"GW": 1, "total_points": score}, ignore_index=True)

        players = current_team.team.team
        n_transfers = 1
        for gw in range(2, 38, 1):
            if verbose:
                print(f"Evaluating team for gw {gw} ... ", end="")
            new_team = self.strategy.pick_gameweek_team(self.year, gw)
            new_team.team.team = players
            transfers = new_team.evaluate_transfers()
            made_transfers = 0
            if len(transfers):
                for row in transfers.iterrows():
                    if n_transfers > 0:
                        remove = row[1]["old"]
                        new_team.team.remove_player(remove)
                        add = row[1]["new"]
                        new_team.add_player(add)
                        self.transfers = self.transfers.append({"GW": gw, "out": remove, "in": add}, ignore_index=True)
                        n_transfers -= 1
                        made_transfers += 1
            if verbose:
                print(f"made {made_transfers} transfer(s).")

            # Score the team:
            uuids = sorted(new_team.team.team["uuid"])
            score = self.score_team(self.year, gw+1, *uuids)
            self.team_points = self.team_points.append({"GW": gw, "total_points": score}, ignore_index=True)

            players = new_team.team.team
            n_transfers += 1
            n_transfers = 2 if n_transfers > 2 else n_transfers

    def score_team(self, year, week, *uuids):
        """Get the total score for the given players in the year-week gameweek.
        By default scores the team at the `first_team` attribute.
        """
        df = self.dataloader.gw(year, week)
        df = df.loc[df["uuid"].isin(uuids)]
        return df["total_points"].sum()
