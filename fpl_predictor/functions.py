
import datetime
import json
import numpy as np
import pandas as pd
import pytz


VERBOSE = False


def verbose_print(msg: str):
    if VERBOSE:
        print(msg)


def save_json(d: dict, fp: str):
    """Overwrite the file at the specified filepath with a new JSON file.

    Args:
        d (dict): dictionary to save.
        fp (str): location of JSON file.
    """
    fp = f"{fp}.json" if str.lower(fp)[-5:] != ".json" else fp

    with open(fp, "w") as f:
        json.dump(d, f)


def open_json(fp: str) -> dict:
    """Open a JSON file at the specified filepath and return it as a dict."""
    fp = f"{fp}.json" if str.lower(fp)[-5:] != ".json" else fp

    with open(fp) as f:
        return json.load(f)


def now_as_string(date_format="%Y_%m_%d %H;%M;%S", timezone="US/Eastern"):
    """Get a string representation of the current time in the desired
    format. Reference for string formatting dates available here:
        http://strftime.org/
    A list of available timezones can be checked at: `pytz.all_timezones`."""
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    tz_now = utc_now.astimezone(pytz.timezone(timezone)).strftime(date_format)
    return tz_now


def next_week(year: int, week: int):
    assert week in list(range(1, 39, 1)), "Invalid week - must be int between 1 and 38."
    assert isinstance(year, int) and year > 2000, "Invalid year."
    next_w = 1 if week == 38 else week + 1
    next_y = year + 1 if week == 38 else year
    return next_y, next_w


def previous_week(year: int, week: int):
    assert week in list(range(1, 39, 1)), "Invalid week - must be int between 1 and 38."
    assert isinstance(year, int) and year > 2000, "Invalid year."
    prev_w = 38 if week == 1 else week - 1
    prev_y = year - 1 if week == 1 else year
    return prev_y, prev_w


def check():
    print("\u2713")


def year_week_minus_n(year: int, week: int, n: int):
    """Get the gameweek `n` weeks prior to the given `year` and `week`
    combination INCLUDING the year-week. So if `n` is 1, just the year-week
    itself is returned."""
    assert 0 < week < 39
    assert n > 0
    n -= 1
    new_week = week - (n % 38)
    if new_week <= 0:
        new_week = sum([38, new_week])
    if n >= week:
        new_year = year - (1 + ((n - week) // 38))
    else:
        new_year = year
    return new_year, new_week


def index_players(df, column="ict_index"):
    """Create an indexed DataFrame of player values for the given column of data
     in the raw `df`. The returned DataFrame has player codes as columns, and
     year and gameweek as a nested index.

    Args:
        df (pd.DataFrame): raw data, (usually the `master` csv).
        column (str): column in `master` to get data from.
    """
    df = pd.pivot_table(df, index=["year", "GW"], columns=["code"], values=column)
    return df


def aggregate_index(index, year: int, week: int, n: int,
                    agg_func=np.mean, cross_seasons=True):
    """Aggregate the results from the `index_players` function across the `n`
    weeks up to and including the gameweek given by `year` and `week`.

    Args:
        index (pd.DataFrame): result from the `index_players` function.
        year (int): gameweek year to count back from (inclusive).
        week (int): gameweek week to count back from (inclusive).
        n (int): number of gameweeks to aggregate.
        agg_func (function): function to use to aggregate data.
        cross_seasons (bool): if not True, only gameweeks in the same year
            as `year` are returned (so `n` may be reduced).
    """
    df = index.copy()
    min_year, min_week = year_week_minus_n(year, week, n)
    if not cross_seasons:
        min_year = year
        min_week = max([1, week - n + 1])
    df = df.iloc[(((df.index.get_level_values("year") >= min_year) &
                   (df.index.get_level_values("GW") >= min_week)) |
                  (df.index.get_level_values("year") > min_year)) &
                 (((df.index.get_level_values("year") == year) &
                   (df.index.get_level_values("GW") <= week)) |
                  (df.index.get_level_values("year") < year))
                 ]
    return agg_func(df).sort_values(ascending=False)
