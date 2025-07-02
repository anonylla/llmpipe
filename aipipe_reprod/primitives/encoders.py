from sklearn.preprocessing import OneHotEncoder, LabelEncoder as LEncoder

'''   重写LabelEncoder   '''
# 重写LabelEncoder，将没有在编码规则里的填充Unknown
class LabelEncoder(LEncoder):

    def fit(self, y):
        return super(LabelEncoder, self).fit(list(y) + ['Unknown'])

    def fit_transform(self, y):
        return super(LabelEncoder, self).fit_transform(list(y) + ['Unknown'])

    def transform(self, y):
        new_y = ['Unknown' if x not in set(self.classes_) else x for x in y]
        return super(LabelEncoder, self).transform(new_y)


import pandas as pd

from .primitive import Primitive

def split_cat_and_num_cols(data: pd.DataFrame):
    data = data.infer_objects()
    # 确保所有列名都是字符串类型
    data.columns = data.columns.astype(str)
    num_x = data.select_dtypes('number')
    num_cols = num_x.columns
    cat_cols = list(set(data.columns) - set(num_cols))
    cat_x = data[cat_cols]
    return cat_x, num_x


class NumericDataPrim(Primitive):
    def __init__(self, random_state=0):
        super(NumericDataPrim, self).__init__(name='NumericData')
        self.id = 1
        self.gid = 6
        self.hyperparams = []
        self.type = 'Encoder'
        self.description = "Extracts only numeric data columns from input."
        self.accept_type = 'a'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_a(data)

    def is_needed(self, data):
        cols = data.columns
        num_cols = data._get_numeric_data().columns
        if not len(cols) == len(num_cols):
            return True
        return False

    def transform(self, train_x, test_x, train_y):
        num_cols = train_x._get_numeric_data().columns
        train_x = train_x[num_cols]
        num_cols = test_x._get_numeric_data().columns
        test_x = test_x[num_cols]
        return train_x, test_x, train_y

class OneHotEncoderPrim(Primitive):
    # can handle missing values. turns nans to extra category
    def __init__(self, random_state=0):
        super(OneHotEncoderPrim, self).__init__(name='OneHotEncoder')
        self.id = 2
        self.gid = 7
        self.hyperparams = []
        self.type = 'data preprocess'
        self.description = "Encode categorical integer features as a one-hot numeric array. "
        self.preprocess = OneHotEncoder()
        self.accept_type = 'c2'
        self.need_y = False

    def can_accept(self, data):
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if len(cat_cols) > 15:
            return False
        return True

    def is_needed(self, data):
        # data = handle_data(data)
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if len(cat_cols) == 0:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        # 确保列名都是字符串类型
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)

        cat_col_num = train_x.select_dtypes('object').shape[1]
        if cat_col_num == 0:  # 如果没有分类特征，直接返回原始数据
            return train_x, test_x, train_y
        
        cat_trainX, num_trainX = split_cat_and_num_cols(train_x)
        cat_testX = test_x[cat_trainX.columns]
        num_testX = test_x[num_trainX.columns]
        # cat_testX, num_testX = split_cat_and_num_cols(test_x)
        cat_cols = cat_trainX.columns
        
        # 使用实际的OneHotEncoder而不是get_dummies
        self.preprocess = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.preprocess.fit(cat_trainX.fillna('MISSING'))
        
        # 转换数据
        train_encoded = pd.DataFrame(
            self.preprocess.transform(cat_trainX.fillna('MISSING')),
            columns=self.preprocess.get_feature_names_out(cat_cols)
        )
        test_encoded = pd.DataFrame(
            self.preprocess.transform(cat_testX.fillna('MISSING')),
            columns=self.preprocess.get_feature_names_out(cat_cols)
        )
        
        # 合并数值列和编码后的分类列
        train_data_x = pd.concat([num_trainX.reset_index(drop=True), train_encoded], axis=1)
        test_data_x = pd.concat([num_testX.reset_index(drop=True), test_encoded], axis=1)
        
        return train_data_x, test_data_x, train_y

class LabelEncoderPrim(Primitive):
    def __init__(self, random_state=0):
        super(LabelEncoderPrim, self).__init__(name='LabelEncoder')
        self.id = 3
        self.gid = 8
        self.hyperparams = []
        self.type = 'data preprocess'
        self.description = "Encode labels with value between 0 and n_classes-1."
        self.preprocess = {}
        self.accept_type = 'b'
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if len(cat_cols) == 0:
            return False
        return True
    
    def transform(self, train_x, test_x, train_y):
        # 确保列名都是字符串类型
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)

        cat_col_num = train_x.select_dtypes('object').shape[1]
        if cat_col_num == 0:  # 如果没有分类特征，直接返回原始数据
            return train_x, test_x, train_y
        
        cat_trainX, num_trainX = split_cat_and_num_cols(train_x)
        cat_testX = test_x[cat_trainX.columns]
        num_testX = test_x[num_trainX.columns]
        # cat_testX, num_testX = split_cat_and_num_cols(test_x)
        cols = cat_trainX.columns

        for col in cols:
            self.preprocess[col] = LabelEncoder()
            # 只在训练集上fit
            self.preprocess[col].fit(cat_trainX[col].astype(str).fillna('MISSING'))
            # 训练集和测试集都使用transform
            cat_trainX[col] = self.preprocess[col].transform(cat_trainX[col].astype(str).fillna('MISSING'))
            cat_testX[col] = self.preprocess[col].transform(cat_testX[col].astype(str).fillna('MISSING'))

        train_data_x = pd.concat([cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)], axis=1)
        test_data_x = pd.concat([cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1)
        return train_data_x, test_data_x, train_y


