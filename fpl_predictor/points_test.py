
from fpl_predictor.squad_builder import SquadBuilder


class PointsTest:

    def __init__(self, sqb: SquadBuilder, year: int, week: int, metric: str,
                 n: int):

        sqb.set_n(n)
        sqb.set_scoring_metric(metric)

        # Calculate scores on a per-value basis:
        self.pv_scores = dict()
        initial_squad_pv = sqb.top_fill(year, week, score_per_value=True)
        opt_squad_pv = sqb.optimize(year, week, score_per_value=True)
        self.pv_scores["first_team_score"] = sqb.team_points(year, week, benchboost=False)
        self.pv_scores["benchboost_score"] = sqb.team_points(year, week, benchboost=True)

        # Calculate scores on a raw basis:
        self.raw_scores = dict()
        initial_squad_raw = sqb.top_fill(year, week, score_per_value=False)
        opt_squad_pv = sqb.optimize(year, week, score_per_value=False)
        self.raw_scores["first_team_score"] = sqb.team_points(year, week, benchboost=False)
        self.raw_scores["benchboost_score"] = sqb.team_points(year, week, benchboost=True)

    @property
    def total_scores(self):
        return dict(
            pv_first_team_score=sum(self.pv_scores["first_team_score"].values()),
            pv_benchboost_score=sum(self.pv_scores["benchboost_score"].values()),
            raw_first_team_score=sum(self.raw_scores["first_team_score"].values()),
            raw_benchboost_score=sum(self.raw_scores["benchboost_score"].values()),
        )