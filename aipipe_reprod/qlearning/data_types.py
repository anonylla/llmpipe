from typing import Literal, Union, Dict, List, Tuple, Optional
import pandas as pd

# typedefs
DatasetType = Dict[Literal['train', 'test', 'target', 'target_test'], Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
