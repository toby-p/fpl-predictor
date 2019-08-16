
from __deprecated__.teambuilder import TeamBuilder


class Strategy:

    def __init__(self, first_gameweek_kw=None, gameweek_kw=None):
        if not first_gameweek_kw:
            first_gameweek_kw = dict()
        if not gameweek_kw:
            gameweek_kw = dict()
        self.first_gameweek_kw = first_gameweek_kw
        self.gameweek_kw = gameweek_kw

    def pick_first_gameweek_team(self, year):
        team = TeamBuilder(year, 1, **self.first_gameweek_kw)
        return team

    def pick_gameweek_team(self, year, week):
        team = TeamBuilder(year, week, **self.gameweek_kw)
        return team
