
import pandas as pd

from fpl_predictor.player_information import PlayerInformation
from fpl_predictor.squad_builder import SquadBuilder


class SeasonSimulator:

    def __init__(self, year, week, builder):
        self.squad_builder = SquadBuilder()
        self._player_information = PlayerInformation()
        self.week = week
        self.year = year
        self.builder = builder
        self.transfers = pd.DataFrame()
        self.team_points = pd.DataFrame()

    def simulate(self, verbose=True):
        self._current_squad = self.builder(self.squad_builder, year=self.year, week=self.week)
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

    def score_team(self):
        """Get the total score for the current squad, year, week attributes."""
        codes = sorted(self.squad_builder.squad.selected["code"])
        points = self._player_information.points_scored(year=self.year, week=self.week)
        points = {k: v for k, v in points.items() if k in codes}
        minutes = self._player_information.minutes_played(self.year, self.week)
        minutes = {k: v for k, v in minutes.items() if k in codes}

        # Double captain/vice-captain's points if they played:
        captain = list(self.squad_builder.squad.captain.keys())[0]
        vice_captain = list(self.squad_builder.squad.vice_captain.keys())[0]
        if minutes[captain] > 0:
            points[captain] = points[captain] * 2
        elif minutes[vice_captain] > 0:
            points[vice_captain] = points[vice_captain] * 2

        # Sub in substitutes who played over first-teamers who didn't:
        subs = self.squad_builder.squad.substitutes
        subs["minutes"] = subs["code"].map(minutes)
        subs = subs.loc[subs["minutes"] > 0]
        first_team = self.squad_builder.squad.first_team
        first_team["minutes"] = first_team["code"].map(minutes)
        first_team = first_team.loc[first_team["minutes"] == 0]
        # TODO - Finish this logic.

        return first_team
