import time
import numpy as np
import re
import random
import pandas as pd
from loguru import logger
import os
import copy

from typing import Literal, Union, Dict, List, Tuple, Optional, Any

from data_quality.metric_basic import check_data_quality
from sklearn.metrics import accuracy_score

import dataclasses
import datetime
import math

from ..primitives import (
    FenginePrimFactory, 
    EncoderPrimFactory, 
    FprocessorPrimFactory, 
    FselectionPrimFactory, 
    ImputerCatPrimFactory, 
    ImputerNumPrimFactory, 
    PredictorPrimFactory,
    PrimFactory,
    Primitive,
)


from .data_types import DatasetType

random.seed(42)
np.random.seed(42)


@dataclasses.dataclass
class Metrics:
    downstream: float
    quality: float
    cost: float

    @staticmethod
    def default():
        return Metrics(-1, -1, -1)
    
    def is_empty(self):
        return self.downstream == -1 and self.quality == -1 and self.cost == -1
    
    def __str__(self):
        return f'Metrics(downstream={self.downstream:.4f}, quality={self.quality:.4f}, cost={self.cost:.4f})'
    
    def copy(self):
        return Metrics(self.downstream, self.quality, self.cost)

class Qlearner:
    def __init__(self,
                 dataset: DatasetType,
                 goal: str,
                 target_name: str,
                 rand_seed=42,
                 n_episodes=100,
                 weights: dict={'downstream':0.7, 'quality': 0.1, 'cost': 0.001},
                 length_theshold = 8,
                 filename='',
                 alpha=1,
                 gamma=0.9,
                 init_epsilon=1.,
                 epsilon_decay=0.99,
                 min_epsilon=0.1,
                 **kwargs):
        self.dataset = dataset
        self.init_dataset = {
            'train': dataset['train'].copy(),
            'test': dataset['test'].copy(),
            'target': dataset['target'].copy(),
            'target_test': dataset['target_test'].copy(),
        }

        ### 加快中间步骤的 reward 计算用
        train_len = self.dataset['train'].shape[0]
        test_len = self.dataset['test'].shape[0]

        max_train_samples = min(400, int(train_len / 4))
        max_test_samples = min(100, int(test_len / 4))

        self.d4step_reward = {
            'train': dataset['train'].sample(max_train_samples, random_state=rand_seed),
            'test': dataset['test'].sample(max_test_samples, random_state=rand_seed)}
        self.d4step_reward['target'] = dataset['target'].iloc[self.d4step_reward['train'].index]
        self.d4step_reward['target_test'] = dataset['target_test'].iloc[self.d4step_reward['test'].index]

        for k, v in self.d4step_reward.items():
            v.reset_index(drop=True, inplace=True)

        self._d4step_reward_init = {
            'train': self.d4step_reward['train'].copy(),
            'test': self.d4step_reward['test'].copy(),
            'target': self.d4step_reward['target'].copy(),
            'target_test': self.d4step_reward['target_test'].copy()
        }
        ###

        self.goal = goal
        self.target_name = target_name

        self.rand_seed = rand_seed
        self.weights = {
            'downstream': weights.get('downstream', 0.7),
            'quality': weights.get('quality', 0.1),
            'cost': weights.get('cost', 0.001)
        }
        self.n_episodes = n_episodes
        self.no_prep_metric = Metrics.default()
        self.last_prep_metric = Metrics.default()
        self.kwargs = kwargs
        self.length_theshold = length_theshold
        self.filename = filename

        self.q: np.ndarray = None
        self.n_states = 0
        self.n_actions = 0

        self.check_missing: bool = self.dataset['train'].isnull().sum().sum() > 0

        logger.info(f'Check missing: {self.check_missing}')

        self.factories: list[PrimFactory] = [
            ImputerCatPrimFactory(),
            ImputerNumPrimFactory(),
            EncoderPrimFactory(),            
            FprocessorPrimFactory(),
            FenginePrimFactory(),
            FselectionPrimFactory(),
        ]
        '''实际上是 class of primitive, call 一下某个 method 之后得到的是 primitive 的实例'''
        self.methods: list[type[Primitive]] = []
        self.downstream_task_factory = PredictorPrimFactory()
        
        for f in self.factories:
            for p in f.get_prims:
                self.methods.append(p)

        self.len_methods = len(self.methods)

        self.n_states = len(self.methods) + 1
        self.n_actions = len(self.methods) + 1

        self.__res_dict = {}

        self.beta = alpha
        self.gamma = gamma
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.init_r_table()
        self.pipeline = []

    @property
    def downstream_task(self):
        return self.downstream_task_factory.find_prim_from_name(self.goal)()
    
    def _run_pipe(self, current_pipe_idx: list[int]):
        pipe_idx_no_clf = [i for i in current_pipe_idx if i < self.len_methods]
        if self.__res_dict.get(tuple(pipe_idx_no_clf), None) is not None:
            return self.__res_dict[tuple(pipe_idx_no_clf)]

        __pipe_start_time = time.time()
        d = copy.deepcopy(self.dataset)
        for i in pipe_idx_no_clf:
            prim = self.methods[i]()
            d['train'], d['test'], d['target'] = prim.transform(d['train'], d['test'], d['target'])
        
        logger.info(f'X train {d["train"].shape}, nan {d["train"].isnull().sum().sum()}, '
                    f'X test {d["test"].shape}, nan {d["test"].isnull().sum().sum()}')
                         
        pipe_run_time = time.time() - __pipe_start_time

        y_pred = self.downstream_task.transform(d['train'], d['target'], d['test'])

        total_time = time.time() - __pipe_start_time

        metric = accuracy_score(d['target_test'], y_pred)

        self.__res_dict[tuple(pipe_idx_no_clf)] = (d, float(metric), pipe_run_time, total_time)

        return d, float(metric), pipe_run_time, total_time

    def update_q(self, q: np.ndarray, r: float, state: int, next_state: int, action: int, beta: float, gamma: float):
        rsa = r
        qsa = q[state, action]
        new_q = qsa + beta * (rsa + gamma * max(q[next_state, :]) - qsa)
        q[state, action] = new_q
        rn = q[state][q[state] > 0] / np.sum(q[state][q[state] > 0])
        q[state][q[state] > 0] = rn
        return r
    
    def no_prep(self):
        d, res, t, tt = self._run_pipe([])

        if not os.path.exists('save/using_aipipe'):
            os.makedirs('save/using_aipipe')
        with open(f'save/using_aipipe/{self.downstream_task.get_name()}_results.txt', 'a') as fout:
            print((
                'no prep',
                self.target_name,
                float(res),
                {'quality': check_data_quality(self.dataset['train'])['totalScore']},
                f'total time {tt:.4f}',
                self.kwargs,
            ), file=fout)
        
        return float(res)

    def manual_pipe(self, current_pipe_idx: list[int]):
        d, res, t, tt = self._run_pipe(current_pipe_idx)
        os.makedirs('save/using_aipipe', exist_ok=True)

        with open(f'save/using_aipipe/{self.downstream_task.get_name()}_results.txt', 'a') as fout:
            print((
                'manual',
                current_pipe_idx,
                self.target_name(),
                float(res),
                check_data_quality(d['train'])['totalScore'],
                f'total time {tt:.4f}'
            ), file=fout)
        
        return float(res)
    
    def initialize_q_table(self):
        self.q = np.zeros((len(self.methods) + 1, len(self.methods) + 1))
        return self.q, self.n_states, self.n_actions
    
    def reset_dataset(self):
        '''中间被处理过的数据重置为初始状态'''
        self.dataset = {
            'train': self.init_dataset['train'].copy(),
            'test': self.init_dataset['test'].copy(),
            'target': self.init_dataset['target'].copy(),
            'target_test': self.init_dataset['target_test'].copy(),
        }
        self.pipeline = []
        self.d4step_reward = {
            'train': self._d4step_reward_init['train'].copy(),
            'test': self._d4step_reward_init['test'].copy(),
            'target': self._d4step_reward_init['target'].copy(),
            'target_test': self._d4step_reward_init['target_test'].copy(),
        }
        return self.dataset
    
    def partial_predict(self):
        '''只对少量数据进行下游任务的准确率预测'''
        y_pred = self.downstream_task.transform(self.d4step_reward['train'], self.d4step_reward['target'], self.d4step_reward['test'])
        acc = accuracy_score(self.d4step_reward['target_test'], y_pred)
        return acc
    
    def calculate_metrics_in_exploring(self):
        '''在 explore 阶段计算所有的 metrics'''
        # quality_metric = check_data_quality(self.d4step_reward['train'])['totalScore'] / 100
        # y_pred = self.downstream_task.transform(self.dataset['train'], self.dataset['target'], self.dataset['test'])
        # accuracy_metric = accuracy_score(self.dataset['target_test'], y_pred)
        accuracy_metric = self.partial_predict()
        cost = 5 - len(self.pipeline)
        return Metrics(accuracy_metric, 101, cost)
    
    def init_r_table(self):
        '''初始化静态的 reward table'''
        self.r_table = np.zeros((self.n_states, self.n_actions))
        self.r_table -= np.eye(self.n_states, self.n_actions)    # 不能连续运行两个相同的算子 -1
        self.r_table[-1, :] = -1                                 # 不能从下游任务跳回数据预处理
        self.cate_spans = []
        _cnt = 0

        # 为每个算子的最后一列定义额外的 reward
        additional_reward_dict = {
            'Imputer': 1,
            'Encoder': 1,
            'Fprocessor': 5,
            'Fengine': 5,
            'Fselection': 5}
        
        for f in self.factories:
            self.cate_spans.append(list(range(_cnt, _cnt + len(f.get_prims))))
            _cnt += len(f.get_prims)

            # 为最后一列附加 reward
            for k, v in additional_reward_dict.items():
                if f.__class__.__name__.startswith(k):
                    self.r_table[self.cate_spans[-1], -1] = v
            
        
        # # 同一类别的算子不连续执行
        # for span in self.cate_spans:
        #     self.r_table[span[0]:span[-1] + 1, span[0]:span[-1] + 1] = -1
    
    def update_constraints(self, action: int):
        '''@deprecated r_table 作为全局宏观的 reward 基础，不再作为约束而更新'''
        span = None
        for s in self.cate_spans:
            if action in s:
                span = s
        if span is None:  # 选到的是下游任务，不在 span 里面
            return
        self.r_table[:, span[0]:span[-1] + 1] = -1

    def choose_action_based_on_constraints(self, 
                                           state: int, 
                                           q: np.ndarray, 
                                           rand_state: np.random.RandomState, 
                                           acc_pipe_idx: list[int],
                                           is_train=True):
        by_qtable = False
        if is_train:
            if rand_state.rand() < self.epsilon:
                # valid_actions = np.where(q[state, :] >= 0)[0]  # 从合法的算子中抽取
                valid_actions = np.where(self.r_table[state] >= 0)[0]
                action: int = rand_state.choice(valid_actions)
                return action, by_qtable
        
        if any(q[state]) > 0:
            if len(acc_pipe_idx) > self.length_theshold:
                action = self.n_actions - 1
            else:
                action = np.argmax(q[state])
                by_qtable = True
                if self.r_table[state, action] < 0: # or (not is_train and action in acc_pipe_idx):  # 0614为避免预测阶段出现已经做过的操作，生成过长。原：self.r_table[state, action] < 0
                    action = self.n_actions - 1
        else:
            valid_actions = np.where(q[state, :] >= 0)[0]
            if len(valid_actions) > 0:
                action = rand_state.choice(valid_actions)
                if self.r_table[state, action] < 0:
                    action = self.n_actions - 1
            else:
                action = self.n_actions - 1
        return int(action), by_qtable
    
    def train(self, verbose_all_episodes=False):
        beta = self.beta
        gamma = self.gamma
        self.epsilon = self.init_epsilon

        random_state = np.random.RandomState(self.rand_seed)
        self.q, n_states, n_actions = self.initialize_q_table()
        update_cnt = 0

        __train_start_time = time.time()

        if self.no_prep_metric.is_empty():
            self.no_prep_metric = self.calculate_metrics_in_exploring()

        for episode in range(self.n_episodes):
            logger.info(f'Episode {episode + 1}/{self.n_episodes}')

            self.init_r_table()
            self.reset_dataset()

            states = list(range(n_states))
            current_state = int(random_state.choice(states))

            self.pipeline.append(current_state)
            # self.last_prep_metric = self.no_prep_metric.copy()
            _, self.last_prep_metric, done, err = self.step(current_state)
            if err:
                continue

            logger.info(self.last_prep_metric)

            while current_state < n_states - 1:
                # 如果随机数小于 0.1，则随机选择一个
                action, by_qtable = self.choose_action_based_on_constraints(current_state, self.q, random_state, self.pipeline)
                next_state, metrics, done, err = self.step(action)
                if err:
                    break
                
                self.pipeline.append(action)
                next_state = action

                logger.info(f'Choose by {"qtable" if by_qtable else "random"}, {action=}, {metrics}')

                # reward 构成：
                # 1. 3 个 component，其中下游任务部分用“提升率”
                # 2. 3 个 component，其中下游任务部分用“准确率”
                # 3. 直接用“提升率” ——reward范围大约为 [-0.1, 0.1]，实验证明在当前 gamma=0.9 状态下会过于注重及时收益
                # 4. 直接用“准确率” ——依据上面提升率的范围，依然会导致过于注重及时收益
                # downstream_improvement = metrics.downstream # (metrics.downstream - self.last_prep_metric.downstream) / (self.last_prep_metric.downstream + 1e-6)
                # reward = ((self.weights['downstream'] * downstream_improvement
                #           + self.weights['quality'] * metrics.quality 
                #           + self.weights['cost'] * metrics.cost) * 0.2
                #           + self.r_table[current_state, action])
                reward = ((metrics.downstream - self.last_prep_metric.downstream) / (self.last_prep_metric.downstream + 1e-6) 
                          + self.r_table[current_state, action])
                
                self.update_q(self.q, reward, current_state, next_state, action, beta, gamma)
                update_cnt += 1
                logger.info(f'shape={self.d4step_reward['train'].shape} {action=} {reward=:.4f}')
                
                current_state = next_state
                self.last_prep_metric = metrics.copy()

                if done:
                    logger.warning(f'Pipeline done {self.pipeline}')
                    break

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

            if verbose_all_episodes:
                best = self.infer_partial(self.q)
                print(f'episode {episode}', best[0].downstream, best[2])

        train_time = time.time() - __train_start_time
        logger.info(f'Train time: {train_time:.4f}')
        logger.info(f'Update cnt: {update_cnt}, Q table:\n{self.q}')

        today = datetime.datetime.now().strftime('%Y%m%d')
        current_time = datetime.datetime.now().strftime('%H%M%S')
        if not os.path.exists(f'q_tables/structured/{today}'):
            os.makedirs(f'q_tables/structured/{today}')

        save_file_name = os.path.join(
            f'q_tables/structured/{today}',
            f'{self.downstream_task.get_name()}_d{self.weights["downstream"]}_q{self.weights["quality"]}_{current_time}.npy'
        )
        self.save_q_table(save_file_name)
        logger.info(f'Q table saved to {save_file_name}')

        return self.q
    
    def step(self, action: int):
        try:
            if action == self.n_actions - 1:
                done = True
                metrics = self.calculate_metrics_in_exploring()
                return action, metrics, done, None

            self.d4step_reward['train'], self.d4step_reward['test'], self.d4step_reward['target'] = self.methods[action]().transform(
                self.d4step_reward['train'], self.d4step_reward['test'], self.d4step_reward['target']
            )
            done = False
            metrics = self.calculate_metrics_in_exploring()
            return action, metrics, done, None
        except Exception as e:
            logger.error(f'Error in step: {e}')
            return action, -1, True, e

    def save_q_table(self, path: str):
        np.save(path, self.q)

    def _pipe_idx_to_str(self, pipe_idx: list[int]):
        pipe = [self.methods[i] for i in pipe_idx if i < self.len_methods]
        return ('[' 
                + ', '.join(map(lambda x: x.get_name(), pipe)) + ', ' 
                + self.downstream_task.get_name() 
                + ']')

    def infer(self, q):
        if isinstance(q, str):
            q = np.load(q)
        self.q = np.array(q)

        rand_state = np.random.RandomState(42)

        n_states = self.len_methods + 1
        strategies: List[Tuple[Metrics, list[int], str]] = []

        __total_t = time.time()

        for i in range(len(q) - 1):
            action_list = []
            current_state = i
            self.init_r_table()
            self.reset_dataset()

            n_steps = 0
            action_list.append(i)
            
            while current_state != n_states - 1 and n_steps <= self.length_theshold:
                n_steps += 1
                next_state, _ = self.choose_action_based_on_constraints(current_state, self.q, rand_state, action_list, False)
                
                current_state = next_state
                action_list.append(current_state)
            
            logger.info(f'Pipeline {i+1}: {action_list}')

            d, metric, t, total_time = self._run_pipe(action_list)

            logger.info(f'Strategy {i+1}, {[self.methods[j].get_name() for j in action_list if j < self.len_methods]}, metric {metric}')

            # quality = check_data_quality(d['train'])['totalScore']
            quality = 101
            m = Metrics(metric, quality, t)
            pipe_str = self._pipe_idx_to_str(action_list)

            strategies.append((m, action_list, pipe_str))

        if self.downstream_task.type == 'Classifier':
            best_strategy = max(strategies, key=lambda x: x[0].downstream)
        else:
            best_strategy = min(strategies, key=lambda x: x[0].downstream)
        logger.info(f'Best strategy: {best_strategy[2]}, {best_strategy[0]}')

        __total_t = time.time() - __total_t

        if not os.path.exists('save/using_aipipe'):
            os.makedirs('save/using_aipipe')
        with open(f'save/using_aipipe/{self.downstream_task.get_name()}_results.txt', 'a') as fout:
            print((
                'reward only',
                self.filename,
                self.target_name,
                float(best_strategy[0].downstream),
                f'episodes={self.n_episodes}',
                best_strategy[2],
                best_strategy[1],
                f'quality {best_strategy[0].quality:.4f}',
                f'pipeline cost {best_strategy[0].cost:.4f}s',
                f'init_epsilon={self.init_epsilon}',
                f'epsilon_decay={self.epsilon_decay}',
                f'infer time {__total_t}',
                self.kwargs,
            ), file=fout)

        return strategies, best_strategy

    def infer_partial(self, q):
        if isinstance(q, str):
            q = np.load(q)
        self.q = np.array(q)

        rand_state = np.random.RandomState(42)

        n_states = self.len_methods + 1
        strategies: List[Tuple[Metrics, list[int], str]] = []

        for i in range(len(q) - 1):
            action_list = []
            current_state = i
            self.init_r_table()
            self.reset_dataset()

            n_steps = 0
            action_list.append(i)
            
            while current_state != n_states - 1 and n_steps <= self.length_theshold:
                n_steps += 1
                next_state, _ = self.choose_action_based_on_constraints(current_state, self.q, rand_state, action_list, False)
                
                current_state = next_state
                action_list.append(current_state)
            
            logger.info(f'Pipeline {i+1}: {action_list}')

            d, metric, t, total_time = self._run_pipe(action_list)
                
            # metric = self.calculate_metrics_in_exploring()
            m = Metrics(metric, 101, t)

            logger.info(f'Strategy {i+1}, {[self.methods[j].get_name() for j in action_list if j < self.len_methods]}, metric {metric}')

            # quality = check_data_quality(d['train'])['totalScore']
            # quality = 101
            pipe_str = self._pipe_idx_to_str(action_list)

            strategies.append((m, action_list, pipe_str))

        if self.downstream_task.type == 'Classifier':
            best_strategy = max(strategies, key=lambda x: x[0].downstream)
        else:
            best_strategy = min(strategies, key=lambda x: x[0].downstream)
        logger.info(f'Best strategy: {best_strategy[2]}, {best_strategy[0]}')
        return best_strategy

