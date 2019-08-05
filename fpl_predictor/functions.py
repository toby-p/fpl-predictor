

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
