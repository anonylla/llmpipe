import numpy as np
from sklearn.metrics import accuracy_score

from .dataset_type import DatasetType
from ..primitives.predictors import LogisticRegressionPrim

from ..primitives import (
    PrimFactory,
    ImputerCatPrimFactory,
    ImputerNumPrimFactory,
    EncoderPrimFactory,
    FprocessorPrimFactory,
    FenginePrimFactory,
    FselectionPrimFactory,
    PredictorPrimFactory,

    Primitive,
)


class QActionProvider:
    factories: list[PrimFactory] = [
        ImputerCatPrimFactory(),  # 1 => index 从 0  开始
        ImputerNumPrimFactory(),  # 3 => index 从 1  开始
        EncoderPrimFactory(),     # 2 => index 从 4  开始
        FprocessorPrimFactory(),  # 9 => index 从 6  开始
        FenginePrimFactory(),     # 8 => index 从 15 开始
        FselectionPrimFactory(),  # 1 => index 从 24 开始
    ]
    methods: list[type[Primitive]] = []
    
    for f in factories:
        for p in f.get_prims:
            methods.append(p)

    # 互斥的算子，当有一个算子被选中时，对应的其他算子就无需再被选中
    # 目前还是手写的，后面可能需要用某种方式，直接在算子属性中定义，相同类别的就不用再被选中了
    exclusive_methods = [
        [1, 2, 3],
        [4, 5],
        [6, 7, 8, 9],
        [15, 16, 23],
        [17, 18, 19, 20, 21, 22],
    ]

    n_action = len(methods) + 1   # 所有的算子 + 1 下游任务
    done_action = len(methods)
    action_ids = np.arange(n_action)

    imputer_ids = (list(range(len(factories[0].get_prims))) 
                   + list(range(len(factories[0].get_prims), len(factories[0].get_prims) + len(factories[1].get_prims))))

    encoder_ids = list(
        range(len(factories[0].get_prims) + len(factories[1].get_prims), 
              len(factories[0].get_prims) + len(factories[1].get_prims) + len(factories[2].get_prims)))

    @classmethod
    def get(cls, action: int):
        if action == cls.done_action:
            return LogisticRegressionPrim()
        return cls.methods[action]()

    @classmethod
    def remove_invalid_actions(cls, selected_action: int, valid_action_ids: list[int]):
        if selected_action == cls.done_action:
            return valid_action_ids
        for exclusive_group in cls.exclusive_methods:
            if selected_action in exclusive_group:
                valid_action_ids = list(set(valid_action_ids) - set(exclusive_group))
        return valid_action_ids
    
    @classmethod
    def calculate_reward(cls, dataset: DatasetType):
        """
        计算当前的reward

        Returns:
            tuple[float, float]: 第一个是当前的reward，第二个是当前的accuracy
        """
        if dataset is None:
            return 0., 0.
        
        if dataset['train'].empty or dataset['test'].empty:
            return 0., 0.
        
        if dataset['train'].shape[1] > 100:
            d = {'train': dataset['train'].iloc[:, :100],
                 'test': dataset['test'].iloc[:, :100],
                 'target': dataset['target'],
                 'target_test': dataset['target_test']}
        else:
            d = dataset

        predictor = LogisticRegressionPrim()
        y_pred = predictor.transform(d['train'], d['target'], d['test'])
        accuracy = accuracy_score(d['target_test'], y_pred)
        return accuracy, accuracy
        
    @classmethod
    def str_to_idx(cls, s: str):
        s = s.replace('<', '').replace('>', '')
        if s == 'blank':
            return QActionProvider.done_action
        for i, m in enumerate(cls.methods):
            if m().__class__.__name__.__contains__(s) or s.__contains__(m.__name__):
                return i
        return QActionProvider.done_action
    