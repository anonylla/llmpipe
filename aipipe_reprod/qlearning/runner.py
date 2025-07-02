import sys
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

import numpy as np
from loguru import logger

sys.path.append('/home/user/projects/autoprep')

# from aipipe_reprod.qlearning.qlearner_l2c import Qlearner
from aipipe_reprod.qlearning.qlearner_reward_only import Qlearner
# from aipipe_reprod.qlearning.ql_re_llm_dis import Qlearner
# from aipipe_reprod.qlearning.experience.ql_re_lm_exp import Qlearner
# from aipipe_reprod.qlearning.distribution.ql_re_llm_dis_double import Qlearner
from operators.loading import Reader

dataset_names = [
    ('abalone', 'label'), # 0
    ('ada_prior', 'label'), # 1
    ('avila', 'Class'),      # 2
    ('connect-4', 'class'),  # 3
    ('eeg', 'Class'),        # 4
    ('google', 'Rating>4.2'), # 5
    ('house_prices', 'SalePrice>150k'), # 6
    ('jungle_chess', 'class'),  # 7
    ('micro', 'class'), # 8
    ('mozilla4', 'state'),     #9
    ('obesity', 'NObeyesdad'), #10
    ('page-blocks', 'class'), #11
    ('pbcseq', 'binaryClass'), #12
    ('pol', 'binaryClass'),  # 13
    ('run_or_walk', 'activity'), #14
    ('shuttle', 'class'), #15
    ('uscensus', 'Income'), #16
    ('wall-robot-nav', 'Class'), #17
]

today = datetime.now().strftime("%Y%m%d")
now = datetime.now().strftime("%H%M%S")

logger.add(f'logs/l2c_enhance/{today}/explorer-{now}.log', level='DEBUG')

d_not_enc = Reader(sep=',') 

for i in range(18):
    filename, target = dataset_names[i]

    logger.info(filename)

    dataset = d_not_enc.train_test_split2(f'/SSD/00/user/diffprep_dataset/{filename}/data.csv', target, rand_state=0)
    ql = Qlearner(dataset, 'LogisticRegressionPrim', target, filename=filename, n_episodes=100,
                weights={'downstream':0.7, 'quality': 0.05, 'cost': 0.01}, epsilon_decay=0.99)
    
    ql_members = ql.__dict__
    for k, v in ql_members.items():
        if isinstance(v, (dict, np.ndarray, pd.DataFrame, pd.Series)):
            continue
        if k.__contains__('downstream_task_factory') or k.__contains__('methods'):
            continue
        logger.info(f'{k} : {v}')

    qtable = ql.train()
    strategies, best_strategy = ql.infer(qtable)
    print(f'Best strategy: {best_strategy[2]}, {best_strategy[1]}')
