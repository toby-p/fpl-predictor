class NotEnoughMoney(Exception):
    pass


class PositionFilled(Exception):
    pass


class AlreadySelected(Exception):
    pass


class Team:
    limits = {"forward": 3, "midfielder": 5,
              "defender": 5, "goalkeeper": 2}
    cap = 1000

    def __init__(self):
        self.__team = {"forward": list(), "midfielder": list(),
                       "defender": list(), "goalkeeper": list()}

    @property
    def total_value(self):
        val = [i["value"] for s in list(self.team.values()) for i in s]
        return sum(val)

    @property
    def selected_players(self):
        players = [i["uuid"] for s in list(self.team.values()) for i in s]
        return players

    def add_player(self, uuid: str, position: str, value: float):
        assert position in self.limits.keys(), f"Invalid position: {position}"
        new_player = {"uuid": uuid, "value": value}
        if len(self.team[position]) == self.limits[position]:
            raise PositionFilled()
        if self.total_value + value > self.cap:
            raise NotEnoughMoney()
        if uuid in self.selected_players:
            raise AlreadySelected()
        self.__team[position].append(new_player)

    @property
    def team(self):
        return self.__team

    @property
    def filled(self):
        if all([len(self.__team[pos]) == self.limits[pos] for pos in self.limits.keys()]):
            return True
        else:
            return False
