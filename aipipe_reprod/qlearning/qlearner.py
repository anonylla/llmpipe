import time
import numpy as np
import re
import random
import pandas as pd
import logging
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
        return f'Metrics(downstream={self.downstream}, quality={self.quality}, cost={self.cost})'

class Qlearner:
    def __init__(self,
                 dataset: DatasetType,
                 goal: str,
                 target_name: str,
                 rand_seed=42,
                 n_episodes=400,
                 weights: dict={'downstream':0.7, 'quality': 0.03, 'cost': 0.001},
                 length_theshold = 8,
                 filename='',
                 **kwargs):
        self.dataset = dataset
        self.goal = goal
        self.target_name = target_name

        self.rand_seed = rand_seed
        self.weights = {
            'downstream': weights.get('downstream', 0.7),
            'quality': weights.get('quality', 0.03),
            'cost': weights.get('cost', 0.001)
        }
        self.n_episodes = n_episodes
        self.no_prep_metric = Metrics.default()
        self.last_prep_metric = Metrics.default()
        self.logger = logging.getLogger(__class__.__name__)
        self.kwargs = kwargs
        self.length_theshold = length_theshold
        self.filename=filename

        self.q: np.ndarray = None
        self.n_states = 0
        self.n_actions = 0

        self.check_missing: bool = self.dataset['train'].isnull().sum().sum() > 0

        self.logger.info(f'Check missing: {self.check_missing}')

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
        self.downstream_task = self.downstream_task_factory.find_prim_from_name(self.goal)()
        
        for f in self.factories:
            for p in f.get_prims:
                self.methods.append(p)

        self.len_methods = len(self.methods)

        self.n_states = len(self.methods) + 1
        self.n_actions = len(self.methods) + 1

        self.__res_dict = {}

        self.beta = 1.
        self.gamma = 0.8
        self.epsilon = 0.8

    
    def _run_pipe(self, current_pipe_idx: list[int]):
        pipe_idx_no_clf = [i for i in current_pipe_idx if i < self.len_methods]
        if self.__res_dict.get(tuple(pipe_idx_no_clf), None) is not None:
            return self.__res_dict[tuple(pipe_idx_no_clf)]

        __pipe_start_time = time.time()
        d = copy.deepcopy(self.dataset)
        for i in pipe_idx_no_clf:
            prim = self.methods[i]()
            d['train'], d['test'], d['target'] = prim.transform(d['train'], d['test'], d['target'])
        
        self.logger.info(f'X train {d["train"].shape}, nan {d["train"].isnull().sum().sum()}, '
                         f'X test {d["test"].shape}, nan {d["test"].isnull().sum().sum()}')
                         
        pipe_run_time = time.time() - __pipe_start_time

        y_pred = self.downstream_task.transform(d['train'], d['target'], d['test'])

        total_time = time.time() - __pipe_start_time

        metric = accuracy_score(d['target_test'], y_pred)

        self.__res_dict[tuple(pipe_idx_no_clf)] = (d, float(metric), pipe_run_time, total_time)

        return d, float(metric), pipe_run_time, total_time

    def _reward(self, d: DatasetType, res: Dict[str, Union[float, int]], pipe_run_time: float) -> float:
        if math.isinf(res):
            return -1
        
        if self.no_prep_metric.is_empty():
            _, no_res, _, _ = self._run_pipe([])
            self.no_prep_metric = Metrics(
                no_res,
                check_data_quality(d['train'])['totalScore'],
                0)
            self.logger.info(f'No prep metric: {self.no_prep_metric}')

        if self.last_prep_metric.is_empty():
            self.last_prep_metric = self.no_prep_metric

        quality_metric = check_data_quality(d['train'])['totalScore']
        cost = pipe_run_time
        
        current_metric = Metrics(
            res,
            quality_metric,
            cost)

        if self.downstream_task.type == 'Classifier':
            dr = current_metric.downstream - max(self.last_prep_metric.downstream, self.no_prep_metric.downstream)
            downstream_improvement = dr / self.last_prep_metric.downstream
        else:
            dr = min(self.last_prep_metric.downstream, self.no_prep_metric.downstream) - current_metric.downstream
            if self.last_prep_metric.downstream == 0:
                downstream_improvement = 0
            else:
                downstream_improvement = dr / self.last_prep_metric.downstream

        reward = (
            downstream_improvement
            + self.weights['quality'] * (current_metric.quality - self.last_prep_metric.quality) / self.last_prep_metric.quality
            - self.weights['cost'] * (current_metric.cost - self.last_prep_metric.cost + 10) / (self.last_prep_metric.cost + 10)
        )

        self.logger.info(f'Current metric: {current_metric}')

        self.last_prep_metric = current_metric
        return reward
        
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
        if not os.path.exists('save/using_aipipe'):
            os.makedirs('save/using_aipipe')

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
    
    def init_constraints(self):
        self.constraints = np.zeros((self.n_states, self.n_actions))
        # self.constraints -= np.eye(self.n_states, self.n_actions)    # 不能连续运行两个相同的算子 -1
        self.constraints[-1, :] = -1                                 # 不能从下游任务跳回数据预处理
        self.cate_spans = []
        _cnt = 0
        for f in self.factories:
            self.cate_spans.append(list(range(_cnt, _cnt + len(f.get_prims))))
            _cnt += len(f.get_prims)
        
        # 同一类别的算子不连续执行
        for span in self.cate_spans:
            self.constraints[span[0]:span[-1] + 1, span[0]:span[-1] + 1] = -1
    
    def update_constraints(self, action: int):
        span = None
        for s in self.cate_spans:
            if action in s:
                span = s
        if span is None:  # 选到的是下游任务，不在 span 里面
            return
        self.constraints[:, span[0]:span[-1] + 1] = -1

    def choose_action_based_on_constraints(self, 
                                           state: int, 
                                           q: np.ndarray, 
                                           rand_state: np.random.RandomState, 
                                           acc_pipe_idx: list[int],
                                           is_train=True):
        if is_train:
            if rand_state.rand() < self.epsilon:
                # valid_actions = np.where(q[state, :] >= 0)[0]  # 从合法的算子中抽取
                valid_actions = np.where(self.constraints[state] >= 0)[0]
                action: int = rand_state.choice(valid_actions)
                return action
        
        if any(q[state]) > 0:
            if len(acc_pipe_idx) > self.length_theshold:
                action = self.n_actions - 1
            else:
                action = np.argmax(q[state])
                if self.constraints[state, action] < 0:
                    action = self.n_actions - 1
        else:
            valid_actions = np.where(q[state, :] >= 0)[0]
            if len(valid_actions) > 0:
                action = rand_state.choice(valid_actions)
                if self.constraints[state, action] < 0:
                    action = self.n_actions - 1
            else:
                action = self.n_actions - 1
        return int(action)
    
    def train(self):
        beta = self.beta
        gamma = self.gamma
        epsilon = self.epsilon

        random_state = np.random.RandomState(self.rand_seed)
        self.q, n_states, n_actions = self.initialize_q_table()

        __train_start_time = time.time()

        for episode in range(self.n_episodes):
            self.logger.info(f'Episode {episode + 1}/{self.n_episodes}')

            self.init_constraints()

            states = list(range(n_states))
            current_state = int(random_state.choice(states))


            accumulated_pipe_idx = [current_state]
            self.last_prep_metric = Metrics.default()

            last_pipe_idx = tuple(accumulated_pipe_idx)

            self.update_constraints(current_state)
            while current_state < n_states - 1:
                # 如果随机数小于 0.1，则随机选择一个
                action = self.choose_action_based_on_constraints(current_state, self.q, random_state, accumulated_pipe_idx)
                # if random_state.rand() < epsilon:
                #     action = random_state.choice(states)
                # else:
                #     # 如果 q table 的这一行有大于 0 的元素，那么大部分情况下都直接选择数值最高的那个元素，但如果 pipeline 长度超出限制，则直接跳转到下游任务
                #     if any(self.q[current_state]) > 0:
                #         if len(accumulated_pipe_idx) > self.length_theshold:
                #             action = n_actions - 1
                #         else:
                #             action = np.argmax(self.q[current_state])
                    
                #     # 如果 q table 的这一行没有一个大于 0 的元素，则从值为 0 的里面随机选一个
                #     else:
                #         valid_actions = np.where(self.q[current_state] >= 0)[0]
                #         if len(valid_actions) > 0:
                #             action = random_state.choice(valid_actions)
                #         else:
                #             action = n_actions - 1
                self.update_constraints(action)

                next_state = action
                accumulated_pipe_idx.append(action)

                if tuple(list(last_pipe_idx) + [int(self.len_methods)]) == tuple(accumulated_pipe_idx):
                    break

                last_pipe_idx = tuple(accumulated_pipe_idx)
                
                self.logger.info(f'Current pipeline: {accumulated_pipe_idx}, '
                                 f'{self._pipe_idx_to_str(accumulated_pipe_idx)}')
                
                d, metric, t, total_time = self._run_pipe(accumulated_pipe_idx)

                reward = self._reward(d, metric, t)
                self.update_q(self.q, reward, current_state, next_state, action, beta, gamma)
                self.logger.info(f'{reward=:.4f}, pipe run time={t}, total time={total_time}')
                
                current_state = next_state

                if len(accumulated_pipe_idx) > self.length_theshold:
                    self.logger.warning(f'Stopped a pipeline since the pipeline length is larger than threshold.')
                    break

            if (episode + 1) % (self.n_episodes // 10) == 0:
                self.epsilon /= 2

        train_time = time.time() - __train_start_time
        self.logger.info(f'Train time: {train_time:.4f}')
        self.logger.info(f'Q table:\n{self.q}')

        today = datetime.datetime.now().strftime('%Y%m%d')
        current_time = datetime.datetime.now().strftime('%H%M%S')
        if not os.path.exists(f'q_tables/structured/{today}'):
            os.makedirs(f'q_tables/structured/{today}')

        save_file_name = os.path.join(
            f'q_tables/structured/{today}',
            f'{self.downstream_task.get_name()}_d{self.weights["downstream"]}_q{self.weights["quality"]}_{current_time}.npy'
        )
        self.save_q_table(save_file_name)
        self.logger.info(f'Q table saved to {save_file_name}')

        return self.q

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

        for i in range(len(q) - 1):
            action_list = []
            current_state = i
            self.init_constraints()

            self.update_constraints(current_state)

            n_steps = 0
            action_list.append(i)
            
            while current_state != n_states - 1 and n_steps <= self.length_theshold:
                n_steps += 1
                next_state = self.choose_action_based_on_constraints(current_state, self.q, rand_state, action_list, False)
                
                self.update_constraints(next_state)

                current_state = next_state
                action_list.append(current_state)

            d, metric, t, total_time = self._run_pipe(action_list)

            self.logger.info(f'Strategy {i+1}, {[self.methods[j].get_name() for j in action_list if j < self.len_methods]}, metric {metric}')

            quality = check_data_quality(d['train'])['totalScore']
            m = Metrics(metric, quality, t)
            pipe_str = self._pipe_idx_to_str(action_list)

            strategies.append((m, action_list, pipe_str))

        if self.downstream_task.type == 'Classifier':
            best_strategy = max(strategies, key=lambda x: x[0].downstream)
        else:
            best_strategy = min(strategies, key=lambda x: x[0].downstream)
        self.logger.info(f'Best strategy: {best_strategy[2]}, {best_strategy[0]}')

        if not os.path.exists('save/using_aipipe'):
            os.makedirs('save/using_aipipe')
        with open(f'save/using_aipipe/{self.downstream_task.get_name()}_results.txt', 'a') as fout:
            print((
                'infer with q table',
                self.filename,
                best_strategy[2],
                self.target_name,
                float(best_strategy[0].downstream),
                f'quality {best_strategy[0].quality:.4f}',
                f'pipeline cost {best_strategy[0].cost:.4f}s',
                f'weight_d {self.weights["downstream"]:.4f}',
                f'weight_q {self.weights["quality"]:.4f}',
                f'weight_c {self.weights["cost"]:.4f}',
                self.kwargs,
            ), file=fout)

        return strategies, best_strategy

