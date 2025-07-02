from typing import Literal, Union, Dict
import pandas as pd

DatasetType = Dict[Literal['train', 'test', 'target', 'target_test'], Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
