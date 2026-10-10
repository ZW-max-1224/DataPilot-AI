from dataclasses import dataclass

import pandas as pd

from app.models.dataset_profile import DatasetProfile


@dataclass
class DatasetContext:
    """The active in-memory dataset shared by Phase 3 tools and the local UI."""

    dataset_id: str
    filename: str
    dataframe: pd.DataFrame
    profile: DatasetProfile
