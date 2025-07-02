import pandas as pd

class Primitive:
    def __init__(self, name: str=None) -> None:
        self.id = 0
        self.gid = 25 
        self.name = name
        self.description = str(name)
        self.hyperparams = []
        self.type = "blank"

    def get_name(self):
        return self.__class__.__name__
    
    @classmethod
    def get_name(cls):
        return cls.__name__

    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError('The method has not be implemented!')

    def transform(self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series):
        return train_x, test_x, train_y
    
    def can_accept(self, data: pd.DataFrame):
        return True

    def can_accept_a(self, data: pd.DataFrame): 
        '''当传入的数据非空，且有数字列时，则接受该处理'''
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        num_cols = data._get_numeric_data().columns
        if not len(num_cols) == 0:
            return True
        return False

    def can_accept_b(self, data: pd.DataFrame):
        '''当传入的数据非空，则接受该处理'''
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        return True

    def can_accept_c(self, data: pd.DataFrame, task=None, larpack=False):
        '''当传入的数据只有数字列且没有空值，则接受该处理'''
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        with pd.option_context('mode.use_inf_as_na', True):
            if data.isna().any().any():
                return False
        if not len(cat_cols) == 0:
            return False
        return True

    def can_accept_c1(self, data: pd.DataFrame, task=None, larpack=False):
        '''当传入的数据只有数字列，则接受该处理'''
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        if not len(cat_cols) == 0:
            return False
        return True

    def can_accept_c2(self, data: pd.DataFrame, task=None, larpack=False):
        '''当传入的数据只有字符串列，则接受该处理'''
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns

        if not len(num_cols) == 0:
            return False
        return True

    def can_accept_d(self, data: pd.DataFrame, task): 
        '''只有数字列，且没有空值'''
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if not len(cat_cols) == 0:
            return False

        with pd.option_context('mode.use_inf_as_na', True):
            if data.isna().any().any():
                return False
        return True

    def is_needed(self, data: pd.DataFrame):
        return True

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__class__.__name__
