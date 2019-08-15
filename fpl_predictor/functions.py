

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
