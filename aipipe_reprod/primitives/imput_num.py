from .primitive import Primitive
from .encoders import split_cat_and_num_cols
from sklearn.impute import SimpleImputer
import pandas as pd

class ImputerMeanPrim(Primitive):
    def __init__(self, random_state=0):
        super(ImputerMeanPrim, self).__init__(name='ImputerMean')
        self.id = 1
        self.gid = 1
        self.hyperparams = []
        self.type = 'ImputerNum'
        self.description = "Imputation transformer for completing missing values by mean."
        self.imp = SimpleImputer()
        self.accept_type = 'c'
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        if data.isna().any().any():
            return True
        return False

    def transform(self, train_x, test_x, train_y):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        if not self.is_needed(train_x) and not self.is_needed(test_x):
            return train_x, test_x, train_y
        cat_trainX, num_trainX = split_cat_and_num_cols(train_x)
        # cat_testX, num_testX = split_cat_and_num_cols(test_x)    
        cat_testX = test_x[cat_trainX.columns]
        num_testX = test_x[num_trainX.columns]
        self.imp.fit(num_trainX)
        cols = list(num_trainX.columns)
        num_trainX = self.imp.fit_transform(num_trainX)
        num_trainX = pd.DataFrame(num_trainX).reset_index(drop=True).infer_objects()
        cols = ['num_'+str(i) for i in num_trainX.columns]
        num_trainX.columns = cols
        train_data_x = pd.concat([cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)],axis=1)

        cols = list(num_testX.columns)
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        cols = ['num_'+str(i) for i in num_testX.columns]
        num_testX.columns = cols
        test_data_x = pd.concat([cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)],axis=1)
        return train_data_x, test_data_x, train_y
        
class ImputerMedianPrim(Primitive):
    def __init__(self, random_state=0):
        super(ImputerMedianPrim, self).__init__(name='ImputerMedian')
        self.id = 2
        self.gid = 2
        self.hyperparams = []
        self.type = 'ImputerNum'
        self.description = "Imputation transformer for completing missing values by median."
        self.imp = SimpleImputer(strategy='median')
        self.accept_type = 'c'
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        if data.isna().any().any():
            return True
        return False

    def transform(self, train_x, test_x, train_y):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        if not self.is_needed(train_x) and not self.is_needed(test_x):
            return train_x, test_x, train_y
        cat_trainX, num_trainX = split_cat_and_num_cols(train_x)
        cat_testX = test_x[cat_trainX.columns]
        num_testX = test_x[num_trainX.columns]
        # cat_testX, num_testX = split_cat_and_num_cols(test_x)    
        self.imp.fit(num_trainX)
        cols = list(num_trainX.columns)
        num_trainX = self.imp.fit_transform(num_trainX)
        num_trainX = pd.DataFrame(num_trainX).reset_index(drop=True).infer_objects()
        cols = ['num_'+str(i) for i in num_trainX.columns]
        num_trainX.columns = cols
        train_data_x = pd.concat([cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)],axis=1)
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        cols = ['num_'+str(i) for i in num_testX.columns]
        num_testX.columns = cols
        test_data_x = pd.concat([cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)],axis=1)
        return train_data_x, test_data_x, train_y


class ImputerNumPrim(Primitive):
    def __init__(self, random_state=0):
        super(ImputerNumPrim, self).__init__(name='ImputerNumMode')
        self.id = 4
        self.gid = 4
        self.hyperparams = []
        self.type = 'ImputerNum'
        self.description = "Imputation transformer for completing missing values by mode."
        self.imp = SimpleImputer(strategy='most_frequent')
        self.accept_type = 'c'
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        if data.isna().any().any():
            return True
        return False

    def transform(self, train_x, test_x, train_y):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        if not self.is_needed(train_x) and not self.is_needed(test_x):
            return train_x, test_x, train_y
        cat_trainX, num_trainX = split_cat_and_num_cols(train_x)
        cat_testX = test_x[cat_trainX.columns]
        num_testX = test_x[num_trainX.columns]
        # cat_testX, num_testX = split_cat_and_num_cols(test_x)    
        self.imp.fit(num_trainX)

        cols = list(num_trainX.columns)
        num_trainX = self.imp.fit_transform(num_trainX)
        num_trainX = pd.DataFrame(num_trainX).reset_index(drop=True).infer_objects()
        cols = ['num_'+str(i) for i in num_trainX.columns]
        num_trainX.columns = cols
        train_data_x = pd.concat([cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)],axis=1)

        cols = list(num_testX.columns)
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        cols = ['num_'+str(i) for i in num_testX.columns]
        num_testX.columns = cols
        test_data_x = pd.concat([cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)],axis=1)
        
        return train_data_x, test_data_x, train_y
    