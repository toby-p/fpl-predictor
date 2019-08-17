
def build_team(self, year, week, n_premium=3, live=False):
    """Build a team, first by selecting `n_premium` players with the highest
    total scores available, then by selecting the rest on `score_per_value`.

    Args:
        year (int): game year.
        week (int): game week. Team is selected FOR this week, using data up
            to the previous week to calculate scores.
        n_premium (int): number of players to select by highest score.
        live (bool): if True, get live player availability, value etc. from
            the API instead of historic data.
    """
    self.__squad.remove_all_players()

    # Choose `n_premium` players first:
    for p in range(n_premium):
        need_positions = self.__squad.need_positions
        total_budget = self.__squad.available_budget
        pool = self._player_pool(year, week, position=need_positions, live=live, max_val=total_budget)
        if not len(pool) and n_premium > 0:  # Can't fill squad, reduce the target number of n_premium by 1.
            self.build_team(year, week, n_premium - 1, live=live)
            return
        player = self._select_player(pool, score_per_value=False)
        self.__squad.add_player(**player)

    # Fill rest of squad:
    for p in range(15 - len(self.__squad.selected)):
        need_positions = self.__squad.need_positions
        total_budget = self.__squad.available_budget
        pool = self._player_pool(year, week, position=need_positions, live=live, max_val=total_budget)
        if not len(pool) and n_premium > 0:  # Can't fill squad, reduce the target number of n_premium by 1.
            print(n_premium)
            self.build_team(year, week, n_premium - 1, live=live)
            return
        player = self._select_player(pool, score_per_value=True)
        self.__squad.add_player(**player)

    print(f"Build team with {n_premium} players by raw score, {15 - n_premium} by roi, "
          f"using {self.n} games data. Total score = {self.__squad.total_score}. "
          f"Remaining budget = {self.__squad.available_budget}")
    self.pick_first_team()
    return self.squad

