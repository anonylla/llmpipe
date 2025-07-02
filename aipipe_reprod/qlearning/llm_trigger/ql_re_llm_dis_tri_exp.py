from collections import deque
import time
import numpy as np
import re
import random
import pandas as pd
from loguru import logger
import os
import copy

from typing import Literal, Union, Dict, List, Tuple, Optional, Any
from _collections_abc import dict_values

from data_quality.metric_basic import check_data_quality
from sklearn.metrics import accuracy_score

import dataclasses
import datetime
import math

from langchain_core.documents import Document

from ...primitives import (
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


from ..data_types import DatasetType
from ..llm.llm_provider import LlmProvider
from ..llm.get_action_distri import get_actions_description, get_dataset_description, get_last_episode_distribution
from ...new_ql.q_action_provider import QActionProvider
from ..llm.embedding import EmbeddingsManager

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
                 llm_distri_frequ=5,
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
                 history_buffer_size=10,
                 **kwargs):
        self.dataset = dataset
        self.init_dataset = {
            'train': dataset['train'].copy(),
            'test': dataset['test'].copy(),
            'target': dataset['target'].copy(),
            'target_test': dataset['target_test'].copy(),
        }
        self.llm_distri_frequ = llm_distri_frequ
        logger.info(os.environ.get('LLM_USAGE'))
        logger.info(os.environ.get('LLM_MODEL_NAME'))
        self.llm = LlmProvider(model_source=os.environ.get('LLM_USAGE'), model_name=os.environ.get('LLM_MODEL_NAME'))

        ### 加快中间步骤的 reward 计算用
        train_len = self.dataset['train'].shape[0]
        test_len = self.dataset['test'].shape[0]

        max_train_samples = min(800, int(train_len))
        max_test_samples = min(200, int(test_len))

        self.d4step_reward = {
            'train': dataset['train'].sample(max_train_samples, random_state=rand_seed),
            'test': dataset['test'].sample(max_test_samples, random_state=rand_seed)}
        self.d4step_reward['target'] = dataset['target'].iloc[self.d4step_reward['train'].index]
        self.d4step_reward['target_test'] = dataset['target_test'].iloc[self.d4step_reward['test'].index]
        # self.d4step_reward = {
        #     'train': dataset['train'].copy(),
        #     'test': dataset['test'].copy(),
        #     'target': dataset['target'].copy(),
        #     'target_test': dataset['target_test'].copy()}

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
        self.llm_prob_distribution = None

        self.semantic_search = EmbeddingsManager(model_source=os.getenv('EMBEDDING_USAGE'), model_name=os.getenv('EMBEDDING_MODEL'))

        self.history_buffer = deque(maxlen=history_buffer_size)

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
        #                                       X_train     y_train      X_test
        y_pred = self.downstream_task.transform(d['train'], d['target'], d['test'])

        total_time = time.time() - __pipe_start_time
        #                       y_test (GT)       y_pred
        metric = accuracy_score(d['target_test'], y_pred)

        self.__res_dict[tuple(pipe_idx_no_clf)] = (d, float(metric), pipe_run_time, total_time)

        return d, float(metric), pipe_run_time, total_time

    def update_q(self, q: np.ndarray, r: float, state: int, next_state: int, action: int, beta: float, gamma: float, verbose=True):
        rsa = r
        qsa = q[state, action]
        new_q = qsa + beta * (rsa + gamma * max(q[next_state, :]) - qsa)
        if verbose:
            logger.info(f'qsa {qsa:.4f} --> {new_q:.4f}')
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

    def _choose_by_rand(self, state: int, rand_state: np.random.RandomState, llm_prob_distribution: dict):
        if llm_prob_distribution is None:
            valid_actions = np.where(self.r_table[state] >= 0)[0]
            action = rand_state.choice(valid_actions)
        else:
            prob = list(llm_prob_distribution.values())
            valid_mask = self.r_table[state] >= 0
            p = [prob[i] if valid_mask[i] else 0 for i in range(len(prob))]
            p = self._format_prob(p)
            action = rand_state.choice(list(range(len(prob))), p=p)
        return int(action)

    def choose_action_based_on_constraints(self, 
                                           state: int, 
                                           q: np.ndarray, 
                                           rand_state: np.random.RandomState, 
                                           acc_pipe_idx: list[int],
                                           llm_prob_distribution: dict=None,
                                           is_train=True):
        by_qtable = False
        
        if len(acc_pipe_idx) > self.length_theshold:
            action = self.n_actions - 1
            return action, by_qtable
        
        if is_train:
            if rand_state.rand() < self.epsilon:
                action = self._choose_by_rand(state, rand_state, llm_prob_distribution)
                return int(action), by_qtable
            
        if any(q[state]) > 0:
            action = np.argmax(q[state])
            by_qtable = True
            # 后面可能还要考虑一下这个跳出的约束要不要加（因为这个约束等价于选自己，选自己会导致无限循环，因此可以认为直接让它跳出）
            if self.r_table[state, action] < 0 or action in acc_pipe_idx:
                action = self.n_actions - 1
        else:
            action = self._choose_by_rand(state, rand_state, llm_prob_distribution)
        return int(action), by_qtable
    
    def _format_prob(self, p):
        if p is None:
            return None
        if isinstance(p, (list, np.ndarray, dict_values)):
            if sum(p) <= 0:
                return None
            p = [p[i] / sum(p) for i in range(len(p))]
            return p
        elif isinstance(p, dict):
            prob = list(p.values())
            if sum(prob) <= 0:
                return None
            prob = [prob[i] / sum(prob) for i in range(len(prob))]
            return prob
    
    def train(self, verbose_all_episodes=False):
        beta = self.beta
        gamma = self.gamma
        self.epsilon = self.init_epsilon

        random_state = np.random.RandomState(self.rand_seed)
        self.q, n_states, n_actions = self.initialize_q_table()

        update_cnt = 0

        __train_start_time = time.time()

        self.accept_llm_times = 0

        if self.no_prep_metric.is_empty():
            self.no_prep_metric = self.calculate_metrics_in_exploring()

        step_rewards = []
        for episode in range(self.n_episodes):
            logger.info(f'Episode {episode + 1}/{self.n_episodes}')

            self.init_r_table()
            self.reset_dataset()

            states = list(range(n_states - 1))
            current_state = int(random_state.choice(states))
            logger.info(f'Initial state {current_state}')

            self.pipeline.append(current_state)
            _, self.last_prep_metric, done, err = self.step(current_state)
            if err:
                continue
            logger.info(self.last_prep_metric)

            accuracy_list = [self.last_prep_metric.downstream]

            step_rewards = [(current_state, 
                             (self.last_prep_metric.downstream - self.no_prep_metric.downstream) / (self.no_prep_metric.downstream + 1e-6), 
                             self.last_prep_metric.downstream)]

            should_trigger_llm = False

            temp_q = np.array(self.q)

            # semantic_search
            data_features_full = []
            metadatas = []

            __cached_data = None
            while current_state < n_states - 1:
                action, by_qtable = self.choose_action_based_on_constraints(state=current_state, 
                                                                            q=self.q, 
                                                                            rand_state=random_state, 
                                                                            acc_pipe_idx=self.pipeline, 
                                                                            llm_prob_distribution=self.llm_prob_distribution)
                __data_feat = get_dataset_description(self.filename, self.d4step_reward['train'], limited_k=10)
                next_state, metrics, done, err = self.step(action)
                if err:
                    break
                
                self.pipeline.append(action)
                accuracy_list.append(metrics.downstream)
                
                next_state = action

                logger.info(f'{action=}, acc={metrics.downstream}')

                should_trigger_llm = self.llm_trigger(accuracy_list, self.history_buffer)

                if should_trigger_llm:
                    __cached_data = self.d4step_reward['train'].copy()
                    break

                reward = ((metrics.downstream - self.last_prep_metric.downstream) / (self.last_prep_metric.downstream + 1e-6) 
                          + self.r_table[current_state, action])
                step_rewards.append((action, reward - self.r_table[current_state, action], metrics.downstream))

                data_features_full.append(__data_feat)
                metadatas.append({'act': action, 'acc': metrics.downstream, 'reward': step_rewards[-1][1], 'type': 'act'})
                
                self.update_q(temp_q, reward, current_state, next_state, action, beta, gamma, verbose=True)
                logger.info(f'Choose by {"q table" if by_qtable else "random"}, shape={self.d4step_reward['train'].shape} {action=} {reward=:.4f}, epsilon={self.epsilon:.4f}')
                update_cnt += 1
                
                current_state = next_state
                self.last_prep_metric = metrics.copy()

                if done:
                    logger.warning(f'Pipeline done {self.pipeline}')
                    break
            
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

            if should_trigger_llm:
                logger.info('Using llm to generate pipeline')
                data_features_full = []
                metadatas = []
                temp_q = np.array(self.q)
                current_df = self.reset_dataset()
                query_results = self.semantic_search.search_by_text_dedup(
                    get_dataset_description(self.filename, __cached_data, limited_k=10),
                    filter={'reward': {'$gt': 0}},
                )

                pipeline: list[int] = self.generate_full_pipe(__cached_data, 
                                                               step_rewards, 
                                                               query_results)
                if pipeline is None:
                    logger.warning('LLM did not provided a valid pipeline.')
                    continue

                if QActionProvider.done_action != pipeline[-1]:
                    if QActionProvider.done_action in pipeline:
                        pipeline.remove(QActionProvider.done_action)
                    pipeline.append(QActionProvider.done_action)

                accuracy_list = []
                current_state = pipeline[0]
                step_rewards = []
                self.last_prep_metric = self.no_prep_metric.copy()
                for _i, step in enumerate(pipeline):
                    __data_feat = get_dataset_description(self.filename, current_df['train'], limited_k=10)
                    next_state, metrics, done, err = self.step(step)
                    if err: continue
                    self.pipeline.append(step)
                    reward = (metrics.downstream - self.last_prep_metric.downstream) / (self.last_prep_metric.downstream + 1e-6)
                    # semantics add
                    data_features_full.append(__data_feat)
                    metadatas.append({'act': step, 'acc': metrics.downstream, 'reward': reward, 'type': 'act'})

                    accuracy_list.append(metrics.downstream)
                    step_rewards.append((step, reward, metrics.downstream))
                    if _i >= 1:  # 依据 2000 的结果： reward > 0 and _i >= 1
                        self.update_q(temp_q, reward + self.r_table[current_state, step], current_state, next_state, step, 
                                      beta, gamma, verbose=True)

                    self.last_prep_metric = metrics.copy()
                    current_state = next_state

                self.accept_llm_times += 1
                logger.info(f'Using llm pipeline {pipeline}, {self.last_prep_metric}')

            # 最后一步相比做了第一步的提升
            self.history_buffer.append(accuracy_list[-1] - accuracy_list[0])
            self.q = np.array(temp_q)

            # 将 pipeline 保存到 semantic search 中
            if len(self.pipeline) > 0:
                data_features_full.append(get_dataset_description(self.filename, self.init_dataset['train'], limited_k=10))
                metadatas.append({'pipe': self.pipeline[:], 'acc': accuracy_list[-1], 
                                'reward': accuracy_list[-1] - self.no_prep_metric.downstream, 'type': 'pipe'})
                
                self.semantic_search.add_texts(
                    data_features_full,
                    metadatas=metadatas,
                )

            if verbose_all_episodes:
                best = self.infer_partial(self.q, episode, int(should_trigger_llm))
                logger.info(f'episode {episode}', f'use llm {int(should_trigger_llm)}', best[0].downstream, best[2])

        train_time = time.time() - __train_start_time
        logger.info(f'Train time: {train_time:.4f}')
        logger.info(f'Update cnt: {update_cnt}, Q table:\n{self.q}')

        today = datetime.datetime.now().strftime('%Y%m%d')
        current_time = datetime.datetime.now().strftime('%H%M%S')
        if not os.path.exists(f'q_tables/structured/{today}'):
            os.makedirs(f'q_tables/structured/{today}')

        save_file_name = os.path.join(
            f'q_tables/structured/{today}',
            f'{self.downstream_task.get_name()}_{current_time}_{self.filename}.npy'
        )
        self.save_q_table(save_file_name)
        logger.info(f'Q table saved to {save_file_name}')

        return self.q
    
    def generate_full_pipe(self, df: pd.DataFrame, 
                        step_rewards: list[tuple[int, float, float]], 
                        semantic_best_pipes: list[Document]=[]):
        dataset_desc = get_dataset_description(self.filename, df)
        actions_text = get_actions_description(list(range(self.n_actions)))

        # 总结最优的 top 5 pipeline 的字符串
        best_pipe_str = ''
        if semantic_best_pipes is not None and len(semantic_best_pipes) > 0:
            for i, doc in enumerate(semantic_best_pipes):
                dataset_desc = doc.page_content
                metadata: dict[str, int | float | str] = doc.metadata
                if metadata['type'].startswith('act'):
                    best_pipe_str += f'\nExample {i+1} {dataset_desc}\n action: {QActionProvider.get(metadata["act"]).get_name()}, accuracy {metadata["acc"]:.4f}, improvement {metadata["reward"]:.4f}\n'
                else:
                    pipe_str = ', '.join([f'{op_id} {QActionProvider.get(op_id).get_name()}' for op_id in metadata['pipe']])
                    best_pipe_str += f'\nExample {i+1} {dataset_desc}\n pipeline: [{pipe_str}], accuracy {metadata["acc"]:.4f}, improvement {metadata["reward"]:.4f}\n'

        current_pipe_str = ''
        if step_rewards is not None and len(step_rewards) > 0:
            for i, (action_id, reward, accuracy) in enumerate(step_rewards):
                current_pipe_str += f'Step {i}: {QActionProvider.get(action_id).get_name()}, accuracy {accuracy:.4f}\n'
            
        prompt_template = self.llm.get_prompt_pipeline_full() # TODO
        prompt = prompt_template.format(
            data_state=dataset_desc,
            current_pipeline=current_pipe_str,
            available_operators=actions_text,
            experience=best_pipe_str,
        )
        logger.debug(f"LLM Prompt: {prompt}")
        try:
            res = self.llm.invoke(prompt, verbose=True)
            return res['suggested_pipeline']
        except Exception as e:
            logger.error(f"Error in LLM invocation: {e}")
            return None
    
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
            return action, Metrics.default(), True, e
        
    def llm_trigger(self, accuracy_list: list[float], history_accuracy: deque[float], reward_theta_threshold=0., history_theta_stagnation=0.005):
        if len(accuracy_list) <= 2:
            return False
        
        ## 如果当前做的这个步骤准确率下降非常多，则调用大模型
        if accuracy_list[-1] < accuracy_list[-2] * 0.6:
            logger.info(f'LLM triggerer condition 1: accuracy {accuracy_list[-2]:.3f} -> {accuracy_list[-1]:.3f}')
            return True
        
        reward_list = []
        for i in range(len(accuracy_list) - 1):
            reward_list.append(accuracy_list[i + 1] - accuracy_list[i])

        if len(reward_list) >= 2 and (reward_list[-1] <= -0.001 and reward_list[-2] <= -0.001):
            logger.info(f'LLM triggerer condition 2: over 2 times reward <= -0.001, calling LLM')
            return True
        
        if len(reward_list) >= 3 and np.mean(reward_list) < reward_theta_threshold:
            logger.info(f'LLM triggerer condition 3: average reward {np.mean(reward_list):.6f} < {reward_theta_threshold}, calling LLM')
            return True
        
        if len(history_accuracy) <= 5:
            return False
        
        if max(history_accuracy) < history_theta_stagnation:
            logger.info(f'LLM triggerer condition 4: last episode histories {max(history_accuracy)} < {history_theta_stagnation}, calling LLM')
            return True

        return False

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
                next_state, _ = self.choose_action_based_on_constraints(current_state, self.q, rand_state, action_list, is_train=False)
                
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
        with open(f'save/using_aipipe/re_llm_dis_tri_{self.downstream_task.get_name()}_results.txt', 'a') as fout:
            print((
                'llm distri trig',
                self.filename,
                self.target_name,
                float(best_strategy[0].downstream),
                f'episodes={self.n_episodes}',
                best_strategy[2],
                f'quality {best_strategy[0].quality:.4f}',
                f'pipeline cost {best_strategy[0].cost:.4f}s',
                f'init_epsilon {self.init_epsilon}',
                f'epsilon_decay {self.epsilon_decay}',
                f'infer time {__total_t}',
                self.kwargs,
            ), file=fout)

        return strategies, best_strategy
        
    def infer_partial(self, q, episode, use_llm):
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
                next_state, _ = self.choose_action_based_on_constraints(current_state, self.q, rand_state, action_list, is_train=False)
                
                current_state = next_state
                action_list.append(current_state)
            
            logger.info(f'Pipeline {i+1}: {action_list}')

            # d, metric, t, total_time = self._run_pipe(action_list)
                
            m = self.calculate_metrics_in_exploring()
            # m = Metrics(metric, 101, t)

            logger.info(f'Strategy {i+1}, {[self.methods[j].get_name() for j in action_list if j < self.len_methods]}, metric {m.downstream}')

            # quality = check_data_quality(d['train'])['totalScore']
            # quality = 101
            pipe_str = self._pipe_idx_to_str(action_list)

            strategies.append((m, action_list, pipe_str))

        if self.downstream_task.type == 'Classifier':
            best_strategy = max(strategies, key=lambda x: x[0].downstream)
        else:
            best_strategy = min(strategies, key=lambda x: x[0].downstream)
        logger.info(f'Best strategy: {best_strategy[2]}, {best_strategy[0]}')

        if not os.path.exists('save/using_aipipe'):
            os.makedirs('save/using_aipipe')
        with open(f'save/using_aipipe/re_llm_dis_tri_{self.downstream_task.get_name()}_results.txt', 'a') as fout:
            print((
                'observe downstream',
                episode,
                self.filename + ' ori',
                self.target_name,
                float(best_strategy[0].downstream),
                f'llm={os.getenv("LLM_USAGE")} {use_llm} with prompt say best',
                best_strategy[2],
                f'episodes={self.n_episodes}',
                f'pipeline cost {best_strategy[0].cost:.4f}s',
                f'epsilon_decay {self.epsilon_decay}',
                self.kwargs,
            ), file=fout)

        return best_strategy
