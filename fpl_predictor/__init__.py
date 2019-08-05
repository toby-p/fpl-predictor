import os
import pandas as pd

from .functions import check
from .dataloader import DataLoader
from .api import ApiData
from nav import DIR_DATA


class FplPredictor:

    def __init__(self, **kwargs):
        self.dataloader = DataLoader(**kwargs)

