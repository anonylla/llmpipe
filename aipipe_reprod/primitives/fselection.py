from loguru import logger
from .primitive import Primitive
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from copy import deepcopy
from itertools import compress
import numpy as np

class VarianceThresholdPrim(Primitive):
    def __init__(self, random_state=0):
        super(VarianceThresholdPrim, self).__init__(name='VarianceThreshold')
        self.id = 1
        self.gid = 24
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Feature selector that removes all low-variance features."
        self.selector = VarianceThreshold()
        self.accept_type = 'c_t'
        self.need_y = True
        
    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        try:
            num_cols = train_x.select_dtypes('number').columns
            self.selector.fit(train_x[num_cols])

            cols = list(train_x.columns.astype(str))
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))   # TODO columns 不是 str
            train_data_x = pd.DataFrame(self.selector.transform(train_x[num_cols]), columns=final_cols)

            cols = list(test_x.columns.astype(str))
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            test_data_x = pd.DataFrame(self.selector.transform(test_x[num_cols]), columns=final_cols)
            return train_data_x, test_data_x, train_y
        except Exception as e:
            logger.error(e)
            return train_x, test_x, train_y
