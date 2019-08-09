

class Simulator:
    def __init__(self, year, *args, **kwargs):
        self.year = year

    def score_team(self, year, week, team=None):
        """Get the total score for the given players in the year-week gameweek.
        By default scores the team at the `first_team` attribute.
        """
        if not team:
            team = self.first_team
        uuids = team["uuid"]
        df = self.dataloader.gw(year, week)
        df = df.loc[df["uuid"].isin(uuids)]
        return df["total_points"].sum()
