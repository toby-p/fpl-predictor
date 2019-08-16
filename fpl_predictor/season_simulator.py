
import pandas as pd

from __deprecated__.dataloader import DataLoader
from __deprecated__.teambuilder import TeamBuilder


class SeasonSimulator:

    def __init__(self, year, strategy):
        self.year = year
        self.strategy = strategy
        self.dataloader = DataLoader(make_master=False, build_player_db=False)
        self.transfers = pd.DataFrame()
        self.team_points = pd.DataFrame()
        self.current_team = TeamBuilder(year, 1)
        self.new_team = TeamBuilder(year, 1)

    def simulate(self, verbose=True):
        self.transfers = pd.DataFrame(columns=["GW", "out", "in"])
        self.team_points = pd.DataFrame(columns=["GW", "total_points"])
        self.current_team = self.strategy.pick_first_gameweek_team(self.year)
        self.current_team.build_team()

        # Score the team:
        uuids = sorted(self.current_team.first_team["uuid"])
        score = self.score_team(self.year, 2, *uuids)
        self.team_points = self.team_points.append({"GW": 1, "total_points": score}, ignore_index=True)

        players = self.current_team.team.team
        n_transfers = 1
        for gw in range(2, 38, 1):
            if verbose:
                print(f"Evaluating team for gw {gw} ... ", end="")
            self.new_team = self.strategy.pick_gameweek_team(self.year, gw)
            self.new_team.team.team = players
            transfers = self.new_team.evaluate_transfers()
            made_transfers = 0
            if len(transfers):
                for row in transfers.iterrows():
                    if n_transfers > 0:
                        remove = row[1]["old"]
                        self.new_team.team.remove_player(remove)
                        add = row[1]["new"]
                        self.new_team.add_player(add)
                        self.transfers = self.transfers.append({"GW": gw, "out": remove, "in": add}, ignore_index=True)
                        n_transfers -= 1
                        made_transfers += 1
            if verbose:
                print(f"made {made_transfers} transfer(s).")

            # Score the team:
            self.new_team.pick_first_team()
            uuids = sorted(self.new_team.first_team["uuid"])
            score = self.score_team(self.year, gw+1, *uuids)
            self.team_points = self.team_points.append({"GW": gw, "total_points": score}, ignore_index=True)

            players = self.new_team.team.team
            n_transfers += 1
            n_transfers = 2 if n_transfers > 2 else n_transfers

    def score_team(self, year, week, *uuids):
        """Get the total score for the given players in the year-week gameweek.
        """
        df = self.dataloader.gw(year, week)
        df = df.loc[df["uuid"].isin(uuids)]
        return df["total_points"].sum()
