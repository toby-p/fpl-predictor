def optimise_1(self, n_top_score=5, x_bottom_roi=5):
    """Select a player from the top n number of players ranked by score, and
    replace them with a cheaper player with a higher roi score. Then drop a
    player with one of the bottom x roi scores and replace them with a
    player with a higher total score. If the total score of the team
    increases keep the changes, else revert to the previous team.
    """
    if not self.team.filled:
        print("Team must be filled before being optimized.")
        return

    self._old_team, old_score = self.team.team.copy(), self.team.total_score
    team = self.team.team.copy()
    n_new_player, x_new_player = None, None
    n_remove_player, x_remove_player = None, None

    # Randomly choose a player with a top n score:
    team.sort_values(by="score", ascending=False, inplace=True)
    team.reset_index(drop=True, inplace=True)
    top_n = list(team.loc[:n_top_score, "uuid"])
    random.shuffle(top_n)
    for n in top_n:
        position = self.team.player_positions[n]
        roi = team.loc[team["uuid"] == n, "roi_score"].values[0]
        max_price = 1000 - (self.team.total_value - team.loc[team["uuid"] == n, "value"].values[0])
        print(max_price)
        pool = self.player_pool(position=position, drop_missed_prev_week=True,
                                drop_unavailable=True, min_score=None, min_roi=roi,
                                max_price=max_price)
        if len(pool):
            n_new_player = self.select_player(pool)
            n_remove_player = n

    # Randomly choose a player with a bottom x roi:
    team.sort_values(by="roi_score", ascending=True, inplace=True)
    team.reset_index(drop=True, inplace=True)
    bottom_x = list(team.loc[:x_bottom_roi, "uuid"])
    random.shuffle(bottom_x)
    for x in bottom_x:
        position = self.team.player_positions[x]
        score = self.team.team.loc[team["uuid"] == x, "score"].values[0]
        max_price = 1000 - (self.team.total_value - team.loc[team["uuid"] == x, "value"].values[0])
        print(max_price)
        pool = self.player_pool(position=position, drop_missed_prev_week=True,
                                drop_unavailable=True, min_score=score, min_roi=None,
                                max_price=max_price)
        if len(pool):
            x_new_player = self.select_player(pool)
            x_remove_player = x

    if n_new_player and x_new_player:
        self.team.remove_player(n_remove_player)
        print(f"Removed player {self.team.player_names[n_remove_player]} ", end="")
        self.team.add_player(**n_new_player)
        print(f"added player {self.team.player_names[n_new_player['uuid']]}")
        self.team.remove_player(x_remove_player)
        print(f"Removed player {self.team.player_names[x_remove_player]} ", end="")
        self.team.add_player(**x_new_player)
        print(f"added player {self.team.player_names[x_new_player['uuid']]}")

    else:
        print("Couldn't find players.")
        self.revert()
        return

    if self.team.total_score > old_score:
        print(f"Increased score by {self.team.total_score - old_score:,}")
    else:
        print("No score improvement.")
        self.revert()


def optimize(self, iterations=10):
    """Needs some work."""
    for i in range(iterations):
        df = self.team.team.sort_values(by=["roi_score"], ascending=True)
        for row in df.iterrows():
            player = row[1]
            budget = self.team.available_budget + player["value"]
            print(budget)
            position = player["position"]
            score = player["score"]
            roi_score = player["roi_score"]
            pool = self.player_pool(position=position, drop_missed_prev_week=True,
                                    drop_unavailable=True, min_score=None,
                                    min_roi=roi_score, max_price=budget)

            if len(pool):
                remove_uuid = player["uuid"]
                r_name = self.team.player_names[remove_uuid]
                self.team.remove_player(remove_uuid)
                new = self.select_player(pool, select_by_roi=True)
                self.team.add_player(**new)
                a_name = self.team.player_names[new["uuid"]]
                print(f"Found improvement - removed {r_name} for {a_name}.")

