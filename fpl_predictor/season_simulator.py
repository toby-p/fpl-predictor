import datetime
import json
import os
import pandas as pd

from fpl_predictor.squad_builder import SquadBuilder
from nav import DIR_DATA


class SeasonSimulator:

    def __init__(self):
        pass

    def wildcards_all_year(self, year, squad_builder_kw, build_func, build_kw,
                           optimiser_func, optimiser_kw):
        sq = SquadBuilder(**squad_builder_kw)
        points, squads = dict(), dict()

        for week in range(1, 39, 1):
            sq.build_squad(build_function=build_func, year=year, week=week, **build_kw)
            optimiser_func(sq, year=year, week=week, **optimiser_kw)
            points[week] = sq.team_points(year, week)
            squads[week] = sq.squad.selected

        self.points = points
        self.squads = squads

        parameters = {"squadbuilder_kw": {k: str(v) for k, v in squad_builder_kw.items()},
                      "build_func": str(build_func),
                      "build_kw": {k: str(v) for k, v in build_kw.items()},
                      "optimiser_func": str(optimiser_func),
                      "optimiser_kw": {k: str(v) for k, v in optimiser_kw.items()}}
        self.save_results(year=year, parameters=parameters)

    @property
    def now(self):
        return datetime.datetime.now().strftime("%Y_%m_%d %H;%M;%S")

    @property
    def total_points(self):
        return sum({sum(v.values()) for k, v in self.points.items()})

    def save_results(self, year, parameters):
        dir_name = f"{self.total_points} - {year} - {self.now}"
        dir_path = os.path.join(DIR_DATA, "results", "wildcard_all_season", dir_name)
        os.mkdir(dir_path)

        fp = os.path.join(dir_path, "squads.xlsx")
        writer = pd.ExcelWriter(fp)
        for k, v in self.squads.items():
            v.to_excel(excel_writer=writer, sheet_name=f"{k}", index=False)
        writer.save()

        with open(os.path.join(dir_path, "week_points.json"), "w") as fp:
            json.dump(self.points, fp)

        with open(os.path.join(dir_path, "parameters.json"), "w") as fp:
            json.dump(parameters, fp)
