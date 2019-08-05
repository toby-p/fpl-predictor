import os
import pandas as pd
import re
import uuid

from .api import ApiData
from .functions import check
from nav import DIR_DATA


class DataLoader:

    def __init__(self, make_master=True, build_player_db=True):
        self.api_data = ApiData()

        master_path = os.path.join(DIR_DATA, "master.csv")
        if not os.path.exists(master_path) or make_master:
            self.make_master()

        players_path = os.path.join(DIR_DATA, "player_ids.csv")
        if not os.path.exists(players_path) or build_player_db:
            self.make_player_ids()
        self.player_ids = pd.read_csv(players_path, encoding="utf-8")

        players_uuids = os.path.join(DIR_DATA, "players_uuids.csv")
        if not os.path.exists(players_uuids) or build_player_db:
            self.make_player_uuids()

        players_master = os.path.join(DIR_DATA, "players_master.csv")
        if not os.path.exists(players_master) or build_player_db:
            self.make_player_database()
        self.players_master = pd.read_csv(players_master, encoding="utf-8")

        # Make the master DataFrame of all player gameweek data:
        df = pd.read_csv(os.path.join(DIR_DATA, "master.csv"), encoding="utf-8")
        df["year"] = [int(s[:4]) for s in df["Season"]]
        df["year, id"] = tuple(zip(df["year"], df["element"]))
        uuids = self.player_uuids.copy()
        uuids["year, id"] = tuple(zip(uuids["year"], uuids["id"]))
        uuid_map = dict(zip(uuids["year, id"], uuids["cross_season_id"]))
        df["player_uuid"] = df["year, id"].map(uuid_map)
        self.DF = df

    def subset(self, year: int, week: int, n: int, cross_seasons=True):
        """Get a subset of raw gameweek data. The data will count back from the
        year-week combination (INCLUDING that year-week) to the n weeks before.
        """
        # Create the indexes of year-week combinations meeting the criteria:
        years = [i for l in [[y] * 38 for y in DataLoader.years()] for i in l]
        weeks = [i for l in [list(range(1, 39, 1)) for _ in DataLoader.years()] for i in l]
        ix = pd.DataFrame(data={"years": years, "weeks": weeks})
        if not cross_seasons:
            ix = ix.loc[ix["years"] == year].reset_index(drop=True)
        try:
            max_ix = ix.loc[(ix["years"] == year) & (ix["weeks"] == week)].index.values[0] + 1
        except IndexError:
            raise IndexError(f"No data for year {year}, week {week}")
        min_ix = max_ix - n
        min_ix = 0 if min_ix < 0 else min_ix
        ix = ix.loc[range(min_ix, max_ix, 1), :]
        ix = tuple(zip(ix["years"], ix["weeks"]))

        # Filter the DF to the year-weeks index:
        df = self.DF.copy()
        df["year"] = [int(s[:4]) for s in df["Season"]]
        df["year, week"] = tuple(zip(df["year"], df["GW"]))
        df = df.loc[df["year, week"].isin(ix)]
        return df.reset_index(drop=True)

    def player_values(self, year, week):
        df = self.subset(year=year, week=week, n=1)
        return dict(zip(df["player_uuid"], df["value"]))

    def get_players(self, *uuids, year=None, names_only=False):
        """Get player details from UUIDs.
        """
        if not year:
            year = self.api_data.current_year
        df = self.players_master.copy()
        df = df.loc[(df["uuid"].isin(uuids)) & (df["year"] == year)]
        df.reset_index(drop=True, inplace=True)
        if names_only:
            df["name"] = df["first_name"].str.cat(df["second_name"], sep=" ")
            df = dict(zip(df["uuid"], df["name"]))
        return df

    @property
    def player_uuids(self):
        fp = os.path.join(DIR_DATA, "players_uuids.csv")
        df = pd.read_csv(fp, encoding="utf-8")
        df["year, id"] = tuple(zip(df["year"], df["id"]))
        return df

    def make_player_database(self):
        """Build master database of all player details with their unique UUID.
        """
        print("Making master player database CSV file ... ", end="")
        fp = os.path.join(DIR_DATA, "players_master.csv")

        years = DataLoader.years()

        # Get first year's data:
        first_year = years[0]
        df = DataLoader.players(year=first_year)
        df["year"] = first_year
        df["Season"] = DataLoader.year_to_season(first_year)

        # Merge with all the other years' data by player name:
        for y in years[1:]:
            new = DataLoader.players(year=y)
            new["year"] = y
            new["Season"] = DataLoader.year_to_season(y)
            df = df.append(new, sort=False)

        df["year, id"] = tuple(zip(df["year"], df["id"]))
        df.reset_index(drop=True, inplace=True)
        uuids = self.player_uuids.copy()
        uuids["year, id"] = tuple(zip(uuids["year"], uuids["id"]))
        uuid_map = dict(zip(uuids["year, id"], uuids["cross_season_id"]))
        df["uuid"] = df["year, id"].map(uuid_map)

        # Add in position:
        df["position"] = df["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})

        df.to_csv(fp, encoding="utf-8", index=False)
        check()

    def make_player_uuids(self):
        print("Making player UUIDs CSV file ... ", end="")
        fp = os.path.join(DIR_DATA, "players_uuids.csv")
        cols = [c for c in self.player_ids.columns if c not in ["name", "cross_season_id"]]
        df = pd.DataFrame()
        for col in cols:
            new = self.player_ids.dropna(subset=[col]).copy()
            new["year"] = int(col[:4])
            new.rename(columns={col: "id"}, inplace=True)
            df = df.append(new[["year", "cross_season_id", "id"]])
        df["id"] = df["id"].astype(int)
        df.to_csv(fp, encoding="utf-8", index=False)
        check()


    @staticmethod
    def make_player_ids():
        print("Making player IDs CSV file ... ", end="")
        df = DataLoader.build_player_uuids()
        fp = os.path.join(DIR_DATA, "player_ids.csv")
        df.to_csv(fp, encoding="utf-8", index=False)
        check()

    @staticmethod
    def season_dir(year: int):
        """Absolute file path of the directory of season data for the given
        start year.
        """
        return os.path.join(DIR_DATA, f"{year}-{(year - 2000) + 1}")

    @staticmethod
    def seasons():
        """List of seasons currently available in the data directory.
        """
        return [x for x in os.listdir(DIR_DATA) if "." not in x]
    
    @staticmethod
    def years():
        """List of years currently available in the data directory.
        """
        return sorted([int(x[:4]) for x in DataLoader.seasons()])

    @staticmethod
    def year_to_season(year: int):
        return f"{year}-{(year - 2000) + 1}"

    @staticmethod
    def player_ids_dict_from_csvs(year: int):
        """Dictionary of player IDs to full name.
        """
        path = os.path.join(DataLoader.season_dir(year), "player_idlist.csv")
        df = pd.read_csv(path)
        df["full_name"] = df["first_name"].str.cat(df["second_name"], sep=" ")
        return dict(zip(df["id"], df["full_name"]))

    @staticmethod
    def search_players(*terms: str, year: int):
        """Case insensitive search of `player_ids_dict` for player names.
        """
        return {k: v for k, v in DataLoader.player_ids_dict_from_csvs(year).items()
                if all([str.lower(t) in str.lower(v) for t in terms])}

    @staticmethod
    def player_dir(year: int, player_id: int):
        """Absolute file path of the directory of player data for the given season start year.
        """
        path = os.path.join(DataLoader.season_dir(year), "players")
        player_dirs = os.listdir(path)
        ids = [int(re.findall(r"\d+", i)[0]) for i in player_dirs]
        id_to_dirs = dict(zip(ids, player_dirs))
        player_dir = os.path.join(path, id_to_dirs[player_id])
        return player_dir

    @staticmethod
    def load_player_gw_data(year: int, player_id: int):
        """Load year's gameweek data for a single player.
        """
        path = DataLoader.player_dir(year, player_id)
        fp = os.path.join(path, "gw.csv")
        return pd.read_csv(fp)

    @staticmethod
    def player_history(year: int, player_id: int):
        path = DataLoader.player_dir(year, player_id)
        fp = os.path.join(path, "history.csv")
        return pd.read_csv(fp)

    @staticmethod
    def make_master_gw_df(year: int):
        """Make master DF of all player GW data for the given year.
        """
        df = pd.DataFrame()
        for i in range(1, 39, 1):
            try:
                new = DataLoader.gw(year, i)
                new["GW"] = i
                df = df.append(new, sort=False)
            except FileNotFoundError:
                continue
        df["Season"] = f"{year}-{(year - 2000) + 1}"
        return df.reset_index(drop=True)

    @staticmethod
    def make_master():
        """Make master DF of all player GW data for all years.
        """
        print("Making master gameweek CSV file ... ", end="")

        years = [int(y[:4]) for y in DataLoader.seasons()]
        df = pd.DataFrame()
        for year in years:
            df = df.append(DataLoader.make_master_gw_df(year), sort=False)

        for col in ["GW", "element"]:
            df[col] = df[col].astype(int)
        df.reset_index(drop=True, inplace=True)
        fp = os.path.join(DIR_DATA, "master.csv")
        df.to_csv(fp, encoding="utf-8", index=False)
        check()

    @staticmethod
    def gw(year: int, week: int):
        path = os.path.join(DataLoader.season_dir(year), "gws")
        if week in range(1, 39, 1):
            filename = f"gw{week}.csv"
            df = pd.read_csv(os.path.join(path, filename), encoding='latin1')
        elif week == "merged":
            filename = "merged_gw.csv"
            fp = os.path.join(path, filename)
            if not os.path.exists(fp):
                DataLoader.make_merged_gw_csv(year)
            df = pd.read_csv(os.path.join(path, filename), encoding='latin1')
        return df

    @staticmethod
    def player_gw(year: int, week: int, player_id: int):
        """Get player gameweek statistics for an individual player.

        Args:
            year (int): starting year of the season.
            week (int/str): either a specific gameweek from 1 to 38,
                or `merged` in which case a DataFrame of the full
                season data is returned.
        """
        df = DataLoader.gw(year, week)
        df = df.loc[df["element"] == player_id]
        if len(df) == 1:
            return dict(df.loc[df.index[0]])
        elif len(df) > 1:
            return df.set_index(["GW"])
        else:
            raise ValueError("Unable to load player gameweek data.")

    @staticmethod
    def build_player_uuids():
        """Build a database of all players with their IDs across different
        seasons, and a single unique UUID.
        """
        years = DataLoader.years()

        # Get first year's data:
        first_year = years[0]
        players = DataLoader.player_ids_dict_from_csvs(year=first_year)
        year_col = DataLoader.year_to_season(first_year)
        df = pd.DataFrame(data={year_col: list(players.keys()), "name": list(players.values())})

        # Merge with all the other years' data by player name:
        for y in years[1:]:
            players = DataLoader.player_ids_dict_from_csvs(year=y)
            year_col = DataLoader.year_to_season(y)
            new = pd.DataFrame(data={year_col: list(players.keys()), "name": list(players.values())})
            df = pd.merge(df, new, left_on="name", right_on="name", how="outer")

        # Get the latest data via API too to make sure all players captured:
        api_data = ApiData()
        new = api_data.player_ids
        new = pd.DataFrame(data={api_data.current_season: list(new.keys()), "name": list(new.values())})
        df = pd.merge(df, new, left_on="name", right_on="name", how="outer")

        df["cross_season_id"] = [uuid.uuid4() for _ in range(len(df))]
        return df.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def players(year):
        fp = os.path.join(DataLoader.season_dir(year), "players_raw.csv")
        return pd.read_csv(fp, encoding="utf-8")

    def score_mean(self, agg_cols=None, subset=None, **kwargs):
        if not subset:
            subset = self.subset(**kwargs)
        if not agg_cols:
            agg_cols=["total_points", "bonus", "bps", "ict_index",
                      "influence", "creativity", "threat"]
        df = subset.sort_values(by=["year, week", "name"])
        df = df.groupby(["player_uuid"])[agg_cols].mean()
        df.sort_values(by=agg_cols, inplace=True, ascending=False)
        return df

    def score_total(self, agg_cols=None, subset=None, **kwargs):
        if not subset:
            subset = self.subset(**kwargs)
        if not agg_cols:
            agg_cols=["total_points", "bonus", "bps", "ict_index",
                      "influence", "creativity", "threat"]
        df = subset.sort_values(by=["year, week", "name"])
        df = df.groupby(["player_uuid"])[agg_cols].sum()
        df.sort_values(by=agg_cols, inplace=True, ascending=False)
        return df

    @property
    def current_values(self):
        df = self.api_data.elements.copy()
        uuids = self.year_player_uuids(self.api_data.current_year)
        df["uuid"] = df["id"].map(uuids)
        df.sort_values(by=["now_cost"], ascending=False, inplace=True)
        return dict(zip(df["uuid"], df["now_cost"]))

    def year_player_uuids(self, year: int):
        df = self.player_uuids
        df = df.loc[df["year"] == year]
        return dict(zip(df["id"], df["cross_season_id"]))

