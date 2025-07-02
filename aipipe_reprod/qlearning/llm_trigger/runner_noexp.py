import json
import sys
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

import numpy as np
from loguru import logger
import os

sys.path.append('/home/user/projects/autoprep')

# from aipipe_reprod.qlearning.qlearner_l2c import Qlearner
# from aipipe_reprod.qlearning.qlearner_reward_only import Qlearner
# from aipipe_reprod.qlearning.ql_re_llm_dis import Qlearner
# from aipipe_reprod.qlearning.experience.ql_re_lm_exp import Qlearner
from aipipe_reprod.qlearning.llm_trigger.ql_re_llm_dis_tri_noexp import Qlearner
from operators.loading import Reader
from dotenv import load_dotenv

load_dotenv()

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

logger.add(f'logs/l2c_enhance/{today}/trigger_exp-{now}.log', rotation='10 MB')

d_not_enc = Reader(sep=',') 

for i in range(0, 18):
    filename, target = dataset_names[i]

    logger.info(filename)
    best_pipes = {}
    if os.path.exists('aipipe_reprod/qlearning/llm_trigger/best_pipes.json'):
        with open('aipipe_reprod/qlearning/llm_trigger/best_pipes.json', 'r') as fin:
            best_pipes = json.load(fin)

    dataset = d_not_enc.train_test_split2(f'/SSD/00/user/diffprep_dataset/{filename}/data.csv', target)
    ql = Qlearner(dataset, 'LogisticRegressionPrim', target, filename=filename, n_episodes=100, rand_seed=1984,
                  weights={'downstream':0.7, 'quality': 0.05, 'cost': 0.01}, epsilon_decay=0.99, best_pipes=best_pipes)
    # if os.path.exists(f'save/vector_store/advisor/index.faiss'):
    #     ql.semantic_search.load_vector_store('save/vector_store/advisor')
    
    ql_members = ql.__dict__
    for k, v in ql_members.items():
        if isinstance(v, (dict, np.ndarray, pd.DataFrame, pd.Series)):
            continue
        if k.__contains__('downstream_task_factory') or k.__contains__('methods'):
            continue
        logger.info(f'{k} : {v}')

    qtable = ql.train(verbose_all_episodes=False)
    strategies, best_strategy = ql.infer(qtable)
    # today = datetime.now().strftime("%Y%m%d")
    # now = datetime.now().strftime("%H%M%S")
    # ql.semantic_search.save_vector_store(f'save/vector_store/advisor_{today}_{now}')
    print(f'Best strategy: {best_strategy[2]}, {best_strategy[0]}')
