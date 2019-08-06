from .functions import check
from .dataloader import DataLoader
from .api import ApiData


class FplPredictor:

    def __init__(self, **kwargs):
        self.dataloader = DataLoader(**kwargs)

