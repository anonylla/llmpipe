import json
import sys
import numpy as np

import os
import pandas as pd
import warnings
from loguru import logger
from dotenv import load_dotenv
from datetime import datetime

sys.path.append('/home/user/projects/autoprep')

from aipipe_reprod.qlearning.llm.get_action_distri import get_dataset_description
from operators.loading import Reader
# from aipipe_reprod.qlearning.trigger_by_deri.qlearner_tri_deri import Qlearner
from aipipe_reprod.qlearning.trigger_by_deri.qlearner_tri_slo_newexp import Qlearner

warnings.filterwarnings("ignore")
load_dotenv()

today = datetime.now().strftime("%Y%m%d")
now = datetime.now().strftime("%H%M%S")

logger.add(f'logs/l2c_enhance/{today}/tri_exp-{now}.log', rotation='10 MB')
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

d_not_enc = Reader(sep=',') 

for i in [2]:
    filename, target = dataset_names[i]

    logger.info(filename)

    # best_pipes = {}
    # if os.path.exists('aipipe_reprod/qlearning/trigger_by_deri/best_pipes.json'):
    #     with open('aipipe_reprod/qlearning/trigger_by_deri/best_pipes.json', 'r') as f:
    #         best_pipes = json.load(f)

    dataset = d_not_enc.train_test_split2(f'/SSD/00/user/diffprep_dataset/{filename}/data.csv', target, rand_state=0)
    ql = Qlearner(dataset, 'LogisticRegressionPrim', target, filename=filename, n_episodes=100,
                epsilon_decay=0.99, init_epsilon=1.,
                rand_seed=42)
    if os.path.exists(f'save/vector_store/advisorpp'):
        ql.semantic_search.load_vector_store(f'save/vector_store/advisorpp')

    ql_members = ql.__dict__
    for k, v in ql_members.items():
        if isinstance(v, (dict, np.ndarray, pd.DataFrame, pd.Series)):
            continue
        if k.__contains__('factor') or k.__contains__('methods'):
            continue
        logger.info(f'{k} : {v}')

    # 手动加载 q table
    # ql.q = np.load('q_tables/structured/20250618/LogisticRegressionPrim_d0.7_q0.05_124141.npy')
    qtable = ql.train()

    strategies, best_strategy = ql.infer(qtable)
    print(f'Best strategy: {best_strategy[2]}, {best_strategy[0]}')
    # logger.info(ql.best_pipes)
    # with open(f'aipipe_reprod/qlearning/trigger_by_deri/best_pipes_{today}_{now}.json', 'w') as f:
    #     json.dump(ql.best_pipes, f)
    ql.semantic_search.save_vector_store(f'save/vector_store/advisorpp')
    today = datetime.now().strftime("%Y%m%d")
    now = datetime.now().strftime("%H%M%S")
    ql.semantic_search.save_vector_store(f'save/vector_store/advisorpp/slice_{filename}_{today}_{now}')
