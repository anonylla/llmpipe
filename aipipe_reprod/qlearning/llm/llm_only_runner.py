import sys
import time
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

import numpy as np
from loguru import logger

sys.path.append('/home/user/projects/autoprep')

from operators.loading import Reader
from aipipe_reprod.qlearning.qlearner_reward_only import Qlearner
from aipipe_reprod.qlearning.llm.llm_provider import LlmProvider
from langchain_core.prompts import ChatPromptTemplate
from aipipe_reprod.qlearning.llm.get_action_distri import get_dataset_description, get_actions_description
from aipipe_reprod.new_ql.q_action_provider import QActionProvider

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

model_name = 'qwen/qwen3-32b:free'
# model_name = 'meta-llama/llama-3.3-70b-instruct:free'

llm = LlmProvider(model_source='openrouter', model_name=model_name)
today = datetime.now().strftime("%Y%m%d")
now = datetime.now().strftime("%H%M%S")

logger.add(f'logs/l2c_enhance/{today}/llm_only-{now}.log', level='DEBUG')

d_not_enc = Reader(sep=',') 

for i in range(18):
    filename, target = dataset_names[i]

    logger.info(filename)

    dataset = d_not_enc.train_test_split2(f'/SSD/00/user/diffprep_dataset/{filename}/data.csv', target)
    ql = Qlearner(dataset, 'LogisticRegressionPrim', target, filename=filename, n_episodes=100,
                weights={'downstream':0.7, 'quality': 0.05, 'cost': 0.01}, epsilon_decay=0.99)
    t = time.time()
    prompt = ChatPromptTemplate.from_template(
        '''
        You are an expert Data Preparation Pipeline Strategist integrated into LLaPipe, a Q-Learning framework for automated data preparation 
        pipeline construction. Your Q-Learning agent is currently STUCK in a performance plateau, indicating a local optimum. Your goal is 
        to propose a **complete, strategic sequence of data preparation operators (a new pipeline segment)** that, when applied from the 
        current data state, has a high potential to escape this local optimum and lead to significantly better downstream model performance.

        Current Situation & Context:

        1.  Problem: The Q-Learning agent has hit a "performance plateau." Recent explorations of single operator changes from the current 
        pipeline tip have not yielded significant improvement in the evaluation metric (accuracy).

        2.  Current Data State (st): {data_state}

        3.  Available Data Preparation Operators (O):
            *   List of available operator names: {available_operators}

        Based on the provided context, your task is to:

        1.  Propose a Strategic Pipeline Segment: Design a new sequence of operators (e.g., 2 to 6 operators long, each element is the operator name)
        2.  Provide Rationale (Briefly): Explain the reasoning behind your proposed pipeline segment. Why do you believe this sequence has a high chance of breaking the current local optimum? Then give your confidence (within 0 to 1) of this pipeline.

        Output your response in the following JSON format:

        ```json
        {{
        "suggested_pipeline": [operator_X, operator_Y, operator_Z],
        "rationale": "This pipeline first [action of operator_X and why it's chosen for the current state/problem], then [action of operator_Y, building on X or addressing another aspect], and finally [action of operator_Z].",
        "confidence": 0.8
        }}
        ```
        '''
    ).format(data_state=get_dataset_description(filename, dataset['train']),
             available_operators=get_actions_description(QActionProvider.action_ids))
    res = llm.invoke(prompt, verbose=True)
    pipe = res['suggested_pipeline']
    _, acc, _, _ = ql._run_pipe(pipe)
    t = time.time() - t
    with open('save/using_aipipe/llm.txt', 'a') as f:
        print((model_name, filename, t, acc), file=f)
