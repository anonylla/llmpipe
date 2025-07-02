from collections import Counter
import dataclasses
import dataclasses_json
import json
import tolerantjson as tjson
import os
import re
import textwrap
from deprecated import deprecated
from langchain_core.prompts import ChatPromptTemplate
from volcenginesdkarkruntime import Ark

import numpy as np
import random
import time
import traceback
import sys

from loguru import logger
from os import getenv

sys.path.append('/home/user/projects/autoprep')
from aipipe_reprod.new_ql.q_action_provider import QActionProvider
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

random.seed(42)
np.random.seed(42)

class ArkWrapper:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.client = Ark(api_key=self.api_key)
        self.model_name = model_name
    
    def invoke(self, prompt, **kwargs):
        llm_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
        )
        llm_response = llm_response.choices[0].message.content
        return llm_response

@dataclasses.dataclass
class PipeRecord:
    pipe: list[int]
    accuracy: float


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class LlmReturnFullPipe:
    no: int
    pipe: list[int]
    rationale: str


class LlmProvider:
    def __init__(self, model_source='Local', model_name="llama3.3"):
        self.model_source = model_source
        if model_source.lower() == 'local' or model_source.lower() == 'ollama':
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(
                model=model_name,
                temperature=0.1,
                num_predict=2048
            )
            logger.info(f'using {self.llm.model}')
        elif model_source.lower() == 'openrouter':
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                openai_api_key=getenv("OPENROUTER_API_KEY"),
                openai_api_base=getenv("OPENROUTER_API"),
                model_name=model_name,
                temperature=0.1,
            )
            logger.info(f'using {self.llm.model_name}')
        elif model_source.lower() == 'doubao':
            self.llm = ArkWrapper(api_key=os.environ["DOUBAO_API_KEY"], model_name=model_name)
            logger.info(f'using {self.llm.model_name}')
        else:
            raise ValueError(f'model_source must be `Local`, `OpenRouter` or `Doubao`')
        
        self.output_prompt = 0
        self.model_name = model_name
        
    def get_prompt_en(self):
        prompt_template = textwrap.dedent(
            '''
            We are building a data preparation pipeline.
            Current dataset characteristics: {dataset_description}
            Available Preprocessing Operations:
            {action_text}
            Current partially built pipeline (P_current): {current_pipe}
            For your reference, here are some historical pipeline construction examples on similar data or from similar starting points:
            {best_pipe}
            {rule}
            We recently tried operator {bad_operator}, which resulted in a poor reward.
            Given this context and the tried operators with poor reward, Please suggest the most appropriate single operator to add next to P_current.
            Please provide your answer as a JSON object with a key "recommended_op_id" and its integer value, and the confidence of choosing this action, ranging from 0 to 1.
            For example: {{"recommended_op_id": 5, "confidence": 0.8}}
            Only return the JSON object.
            '''
        )
        return ChatPromptTemplate.from_template(prompt_template)
    
    def get_prompt_pipeline_full(self):
        prompt_template = textwrap.dedent(
        '''
        You are an expert Data Preparation Pipeline Strategist integrated into LLaPipe, a Q-Learning framework for automated data preparation 
        pipeline construction. Your Q-Learning agent is currently STUCK in a performance plateau, indicating a local optimum. Your goal is 
        to propose a **complete, strategic sequence of data preparation operators (a new pipeline segment)** that, when applied from the 
        current data state, has a high potential to escape this local optimum and lead to significantly better downstream model performance.

        Current Situation & Context:

        1.  Problem: The Q-Learning agent has hit a "performance plateau." Recent explorations of single operator changes from the current 
        pipeline tip have not yielded significant improvement in the evaluation metric (accuracy).

        2.  Current Data State (st): {data_state}

        3.  Stagnant Pipeline (P_stuck) that led to the plateau:
            *   Operator sequence with the current performance: {current_pipeline}

        4.  Available Data Preparation Operators (O):
            *   List of available operator names: {available_operators}

        5.  Distilled Experiential Knowledge (E) - Insights from Past Successes/Failures:
            *   Provide relevant snippets from Mdistill. Focus on patterns or rules related to overcoming plateaus or handling similar data states.
            {experience}

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
        ''')
        return ChatPromptTemplate.from_template(prompt_template)
    
    def invoke(self, prompt: str, verbose=False):
        try:
            llm_response = self.llm.invoke(prompt)

            if hasattr(llm_response, 'content'):
                llm_response_content = llm_response.content
            else:
                llm_response_content = str(llm_response)
            if verbose:
                logger.info(f"LLM Response: {llm_response_content}")
            return self._parse_llm_action_recommendation(llm_response_content)
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            logger.error(traceback.format_exc())
            return None
        
    def chat(self, prompt: str, verbose=False):
        try:
            llm_response = self.llm.invoke(prompt)
            if hasattr(llm_response, 'content'):
                llm_response_content = llm_response.content
            else:
                llm_response_content = str(llm_response)
            if verbose:
                logger.info(f"LLM Response: {llm_response_content}")
            return llm_response_content
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            logger.error(traceback.format_exc())
            return ''

    def _parse_llm_action_recommendation(self, llm_response_content: str):
        """解析 LLM 返回的推荐动作 ID"""
        # logger.debug(f"LLM raw response for action recommendation: {llm_response_content}")
        try:
            # 尝试直接解析 JSON
            content_cleaned = llm_response_content.strip()
            if content_cleaned.startswith("```json"):
                content_cleaned = re.search(r"```json\n(.*?)\n```", content_cleaned, re.DOTALL).group(1)
            elif content_cleaned.startswith("```"):
                content_cleaned = re.search(r"```\n(.*?)\n```", content_cleaned, re.DOTALL).group(1)

            data = json.loads(content_cleaned)
            pipeline = data.get("suggested_pipeline")
            rationale = data.get("rationale")
            if isinstance(pipeline, list):
                # 判断 pipeline 里面的每个元素是不是数字，如果是字符串，则需要将字符串表达的 pipeline 转换为数字
                if all(str(op_id).isdigit() for op_id in pipeline):
                    pipeline = [int(op_id) for op_id in pipeline]
                else:
                    logger.warning(f"LLM recommended_op_id is not a list of numbers: {pipeline}")
                    pipeline = [QActionProvider.str_to_idx(str(op_id).strip()) for op_id in pipeline]
                return {"suggested_pipeline": pipeline, "rationale": rationale}
            else:
                logger.warning(f"LLM recommended_op_id is not a list: {pipeline}")
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode LLM JSON response: {llm_response_content}")
            # 尝试从文本中提取数字 (更不鲁棒的方法)
            match = re.search(r'\[\d+(,\s*\d+)*\]', llm_response_content)
            if match:
                pipeline = match.group(0)
                pipeline = [int(op_id.strip()) for op_id in pipeline[1:-1].split(',')]
                return {"suggested_pipeline": pipeline, "rationale": ''}
        except Exception as e:
            logger.error(f"Error parsing LLM action recommendation: {e}")
        return None

    def summarize_and_get_operator_distribution(self, description: str):
        prompt_template = textwrap.dedent(
        '''
        You are an AI assistant tasked with designing an intelligent exploration strategy for a Q-Learning agent. Your goal is to propose a probability distribution over available actions that will help the agent discover better data preprocessing pipelines more efficiently in the *next* episode, especially if the agent seems stuck or could benefit from trying new approaches. This is *not* about picking the single best next action, but about guiding exploration.

        The Q-Learning task is to find the best data preprocessing pipeline constructed with multiple operators, to improve the current dataset's downstream logistic regression accuracy.
        The available operators (actions) are:
        {operators}

        In the environment, the agent receives the downstream logistic regression accuracy improvement as the reward for each action, while receiving a penalty if the accuracy decreases. The agent's actions will terminate when the total number of actions reaches max_steps or choose the LogisticRegression action. The goal is to maximize the total improvement of the accuracy by executing the provided actions.

        The dataset and pipeline constructed in the previous episode was: {previous_episode_desc}

        Based on your analysis of the previous episode's dataset characteristics, the pipeline constructed, and its performance, identify potential areas for improvement or diversification in the pipeline construction process. For example, consider:
        - Were certain types of operators (e.g., scalers, transformers for skewed data, outlier handlers, dimensionality reduction) underutilized or overutilized given the dataset properties and stated needs?
        - Given the presence of outliers and skewed distributions, which operators might be particularly beneficial to explore more?

        Then, translate this analysis into a probability distribution for exploring the {action_num} available actions in the next episode. The probabilities should sum to 1.0.
        Prioritize actions that seem promising for exploration based on your analysis. Assign very low probabilities to actions that are clearly irrelevant to the dataset (e.g., categorical imputers for a numeric-only dataset without missing values) or were part of a demonstrably suboptimal strategy previously.

        Please output the distribution of the {action_num} action explorations for the next episode based on your analysis in decimal form.
        Give your answer *only* in the specified JSON-like format without analysis: {{0: p0, 1: p1, 2: p2, ..., {last_act_id}: p{last_act_id}}}
        ''')
        prompt = ChatPromptTemplate.from_template(prompt_template)
        operators = []

        # 将所有 action 整合为字符串
        for i, op in enumerate(QActionProvider.action_ids):
            operators.append(f'{i} {QActionProvider.get(op).get_name()}')

        operators = '\n'.join(operators)
        operators += '(LogisticRegression is the downstream task, choosing this action means to finish the pipeline, this action should not has the prob of 0)'

        prompt = prompt.format(
            operators=operators,
            previous_episode_desc=description,
            action_num=QActionProvider.n_action,
            last_act_id=QActionProvider.n_action - 1,
        )
        if self.output_prompt < 1:
            logger.info(prompt)
            self.output_prompt += 1

        llm_response = self.llm.invoke(prompt)
        if hasattr(llm_response, 'content'):
            llm_response = llm_response.content
        logger.debug(llm_response)

        l_idx = llm_response.find('{')
        r_idx = llm_response.find('}')

        pattern = r'\b\d*\.\d+\b'
        matches = re.findall(pattern, llm_response[l_idx:r_idx+1])
        probs = matches[-QActionProvider.n_action:]
        probs_llm = {i: float(item) for i, item in enumerate(probs)}
        return probs_llm

    @deprecated
    def reduce_candicates(self, dataset_description: str):
        prompt_template = textwrap.dedent(
            '''
            You are an AI assistant tasked with designing an intelligent exploration strategy for a Q-Learning agent. Your goal is to propose a list of operator candidates (a subset of the provided actions) that will help the agent discover better data preprocessing pipelines more efficiently in the *next* episode, especially if the agent seems stuck or could benefit from trying new approaches. This is *not* about picking the single best next action, but about guiding exploration.

            The Q-Learning task is to find the best data preprocessing pipeline constructed with multiple operators, to improve the current dataset's downstream logistic regression accuracy.
            The available operators (actions) are:
            {operators}

            In the environment, the agent receives the downstream logistic regression accuracy improvement as the reward for each action, while receiving a penalty if the accuracy decreases. The agent's actions will terminate when the total number of actions reaches max_steps or choose the LogisticRegression action. The goal is to maximize the total improvement of the accuracy by executing the provided actions.

            The dataset feature description is: {dataset_description}

            Based on your analysis of the previous episode's dataset characteristics, identify potential areas for improvement or diversification in the pipeline construction process. For example, consider:
            - Were certain types of operators (e.g., scalers, transformers for skewed data, outlier handlers, dimensionality reduction) underutilized or overutilized given the dataset properties and stated needs?
            - Given the presence of outliers and skewed distributions, which operators might be particularly beneficial to explore more?

            Then select the top operators that seem promising for exploration based on your analysis, and output the id of these operators in a list without analysis, for example: [0, 1, 2, 4, ..., 18, 23, 25]
            '''
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)

        operators = []
        # 将所有 action 整合为字符串
        for i, op in enumerate(QActionProvider.action_ids):
            operators.append(f'{i} {QActionProvider.get(op).get_name()}')
        operators = '\n'.join(operators)
        operators += '(LogisticRegression is the downstream task, choosing this action means to finish the pipeline)'

        prompt = prompt.format(
            operators=operators,
            dataset_description=dataset_description,
        )

        llm_response = self.llm.invoke(prompt)
        if hasattr(llm_response, 'content'):
            llm_response = llm_response.content
        logger.debug(llm_response)
        
        l_idx = llm_response.find('[')
        r_idx = llm_response.rfind(']')

        pattern = r'\b\d+\b'
        matches = re.findall(pattern, llm_response[l_idx:r_idx+1])
        candidates = [int(item) for item in matches]
        return candidates
    
    @deprecated
    def get_distributions_from_candidates(self, dataset_description: str, candidates: list):
        prompt_template = textwrap.dedent(
        '''
        You are an AI assistant tasked with designing an intelligent exploration strategy for a Q-Learning agent. Your goal is to propose a probability distribution over candidate actions that will help the agent discover better data preprocessing pipelines more efficiently in the *next* episode, especially if the agent seems stuck or could benefit from trying new approaches. This is *not* about picking the single best next action, but about guiding exploration.

        The Q-Learning task is to find the best data preprocessing pipeline constructed with multiple operators, to improve the current dataset's downstream logistic regression accuracy.
        The candidate operators (actions) are:
        {operators}

        In the environment, the agent receives the downstream logistic regression accuracy improvement as the reward for each action, while receiving a penalty if the accuracy decreases. The agent's actions will terminate when the total number of actions reaches max_steps or choose the LogisticRegression action. The goal is to maximize the total improvement of the accuracy by executing the provided actions.

        The dataset and pipeline constructed in the previous episode was: {previous_episode_desc}

        Based on your analysis of the previous episode's dataset characteristics, the pipeline constructed, and its performance, identify potential areas for improvement or diversification in the pipeline construction process. For example, consider:
        - Were certain types of operators (e.g., scalers, transformers for skewed data, outlier handlers, dimensionality reduction) underutilized or overutilized given the dataset properties and stated needs?
        - Given the presence of outliers and skewed distributions, which operators might be particularly beneficial to explore more?

        Then, translate this analysis into a probability distribution for exploring the candidate actions in the next episode. The probabilities should sum to 1.0.
        Prioritize actions that seem promising for exploration based on your analysis. Assign very low probabilities to actions that are clearly irrelevant to the dataset (e.g., categorical imputers for a numeric-only dataset without missing values) or were part of a demonstrably suboptimal strategy previously.

        Please output the distribution of the {action_num} action explorations for the next episode based on your analysis in decimal form.
        Give your answer *only* in the specified JSON-like format (operator id as key, prob as value) without analysis: {{0: p0, 1: p1, 4: p4, ...}}
        ''')
        prompt = ChatPromptTemplate.from_template(prompt_template)
        operators = []

        # 将所有 action 整合为字符串
        for i, op in enumerate(candidates):
            operators.append(f'{op} {QActionProvider.get(op).get_name()}')

        operators = '\n'.join(operators)
        operators += '(LogisticRegression is the downstream task, choosing this action means to finish the pipeline, this action should not has the prob of 0)'

        prompt = prompt.format(
            operators=operators,
            previous_episode_desc=dataset_description,
            action_num=len(candidates),
        )
        if self.output_prompt < 1:
            logger.info(prompt)
            self.output_prompt += 1

        llm_response = self.llm.invoke(prompt)
        if hasattr(llm_response, 'content'):
            llm_response = llm_response.content
        logger.debug(llm_response)

        l_idx = llm_response.find('{')
        r_idx = llm_response.rfind('}')

        probs_str = llm_response[l_idx:r_idx+1].strip()
        # 直接 loads 时，key 不能是数字类型，会报错
        # probs_llm = json.loads(probs_str)
        pattern = r'"?\b(\d+)"?:\s*(\d+\.\d+)\b'
        matches = re.findall(pattern, probs_str)
        probs_llm = {int(k): float(v) for k, v in matches}
        logger.info(probs_llm)
        return probs_llm
    
    @deprecated
    def get_score_for_all_actions(self, dataset_description: str, history: list[tuple[int, float]]):
        prompt_template = textwrap.dedent(
        '''
        You are an AI assistant tasked with designing an intelligent exploration strategy for a Q-Learning agent.
        Your goal is to propose a list of scores for all actions in the action space, to help the agent discover 
        better data preprocessing pipelines more efficiently in the *next* episode, especially if the agent seems 
        stuck or could benefit from trying new approaches. This is *not* about picking the single best next action, 
        but about guiding exploration.

        The Q-Learning task is to find the best data preprocessing pipeline constructed with multiple operators, 
        to improve the current dataset's downstream logistic regression accuracy.
        The available operators (actions) are:
        {operators}

        In the environment, the agent receives the downstream logistic regression accuracy improvement as the reward 
        for each action, while receiving a penalty if the accuracy decreases. The agent's actions will terminate when
        the total number of actions reaches max_steps or choose the LogisticRegression action. The goal is to maximize 
        the total improvement of the accuracy by executing the provided actions.

        {history}

        The dataset feature description is: {dataset_description}

        Based on your analysis of the previous episode's dataset characteristics, the pipeline constructed, and 
        its performance, identify potential areas for improvement or diversification in the pipeline construction 
        process. For example, consider:
        - Were certain types of operators (e.g., scalers, transformers for skewed data, outlier handlers, 
        dimensionality reduction) underutilized or overutilized given the dataset properties and stated needs?
        - Given the presence of outliers and skewed distributions, which operators might be particularly beneficial to explore more?

        Then, translate this analysis into a score for exploring the available actions in the next episode (score ranges from 0 to 100). 
        Prioritize actions that seem promising for exploration based on your analysis. Assign very low scores 
        to actions that are clearly irrelevant to the dataset (e.g., categorical imputers for a numeric-only dataset 
        without missing values) or were part of a demonstrably suboptimal strategy previously.

        Please output the distribution of the {action_num} action explorations for the next episode based on your analysis in decimal form.
        Give your answer *only* in the specified JSON-like format (operator id as key, score as value) without analysis: {{0: s0, 1: s1, 2: s2, ..., {last_act_id}: s{last_act_id}}}
        ''')
        prompt = ChatPromptTemplate.from_template(prompt_template)

        operators = []
        # 将所有 action 整合为字符串
        for i, op in enumerate(QActionProvider.action_ids):
            operators.append(f'{i} {QActionProvider.get(op).get_name()}')
        operators = '\n'.join(operators)
        operators += '(LogisticRegression is the downstream task, choosing this action means to finish the pipeline)'

        history_str = ''
        if history is not None and len(history) > 0:
            history_str = 'The previous steps are: \n'
            for op_id, accuracy in history:
                history_str += f'{op_id}: {QActionProvider.get(op_id).get_name()} accuracy: {accuracy}\n'

        prompt = prompt.format(
            operators=operators,
            dataset_description=dataset_description,
            action_num=QActionProvider.n_action,
            last_act_id=QActionProvider.n_action - 1,
            history=history_str,
        )

        try:
            llm_response = self.llm.invoke(prompt)

        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            logger.error(traceback.format_exc())
            return None

        if hasattr(llm_response, 'content'):
            llm_response = llm_response.content
        logger.debug(llm_response)

        l_idx = llm_response.find('{')
        r_idx = llm_response.rfind('}')

        scores_str = llm_response[l_idx:r_idx+1].strip()
        pattern = r'"?(\d+)"?:\s*(\d+)'
        matches = re.findall(pattern, scores_str)
        scores = {int(k): float(v) for k, v in matches}
        logger.info(scores)
        scores = {k: v / sum(scores.values()) for k, v in scores.items()}  # normalize to a probability distribution
        return scores

    def get_multiple_pipelines_from_last_episodes(self,
                                                  dataset_description: str,
                                                  last_episode_records: list[PipeRecord]):
        prompt_template = textwrap.dedent(
        '''
        You are an expert Data Preparation Pipeline Strategist integrated into LLaPipe, 
        a Q-Learning framework for automated data preparation pipeline construction. 
        Your Q-Learning agent is currently STUCK in a performance plateau, indicating 
        a local optimum. Your goal is to propose a **complete, strategic sequence of 
        data preparation operators (a new pipeline segment)** that, when applied from 
        the current data state, has a high potential to escape this local optimum and 
        lead to significantly better downstream model performance.

        Current Situation & Context:

        1.  Problem: The Q-Learning agent has hit a "performance plateau." Recent 
        explorations of single operator changes from the current pipeline tip have 
        not yielded significant improvement in the evaluation metric (accuracy).

        2.  Current Data State (st): {data_state}

        3.  Stagnant Pipeline (P_stuck) that led to the plateau:
            *   Operator sequence with the current performance: {current_pipeline}

        4.  Available Data Preparation Operators (O):
            *   List of available operator names: {available_operators}

        Based on the provided context, your task is to:

        1.  Propose some new Strategic Pipeline Segment (no less than one pipeline): 
            Design several new sequence of operators (e.g., 2 to 6 operators long, 
            each element is the operator id)
        2.  Provide Rationale (Briefly): Explain the reasoning behind your proposed 
            pipeline segment. Why do you believe this sequence has a high chance of 
            breaking the current local optimum?

        Output your response in the following JSON format:

        ```json
        [
        {{
        "no": 1,
        "pipe": [operator_X_id, operator_Y_id, operator_Z_id],
        "rationale": "This pipeline first [action of operator_X and why it's chosen for the current state/problem], then [action of operator_Y, building on X or addressing another aspect], and finally [action of operator_Z]."
        }},
        {{
        "no": 2,
        "pipe": [operator_X_id, operator_W_id, operator_K_id],
        "rationale": "This pipeline first [action of operator_X and why it's chosen for the current state/problem], ..."
        }},
        ...
        ]
        ```
        '''
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        operators = []
        # 将所有 action 整合为字符串
        for i, op in enumerate(QActionProvider.action_ids):
            operators.append(f'{i} {QActionProvider.get(op).get_name()}')
        operators = '\n'.join(operators)
        operators += '(LogisticRegression is the downstream task, choosing this action means to finish the pipeline)'

        recent_pipelines = []
        for i, record in enumerate(last_episode_records):
            pipeline_str = ', '.join([QActionProvider.get(op_id).get_name() for op_id in record.pipe])
            recent_pipelines.append(f'pipeline {i + 1}: [{pipeline_str}], accuracy {record.accuracy:.4f}')

        recent_pipelines = '\n'.join(recent_pipelines)

        prompt = prompt.format(
            data_state=dataset_description,
            current_pipeline=recent_pipelines,
            available_operators=operators,
        )

        llm_response = self.llm.invoke(prompt)
        if hasattr(llm_response, 'content'):
            llm_response = llm_response.content
        logger.debug(llm_response)
        
        l_idx = llm_response.find('[')
        r_idx = llm_response.rfind(']')

        objs = tjson.tolerate(llm_response[l_idx:r_idx+1])
        all_pipelines: list[LlmReturnFullPipe] = [LlmReturnFullPipe.from_dict(obj) for obj in objs]
        return all_pipelines

    def count_patterns(self, best_pipes: list[list[int]]):
        counter = {2: Counter(), 3: Counter(), 4: Counter()}
        for pipe in best_pipes:
            opid_list = pipe
            for window_size in [2, 3, 4]:
                for i in range(len(opid_list) - window_size + 1):
                    pattern = tuple(opid_list[i:i+window_size])
                    counter[window_size][pattern] += 1
        return counter
    
    def get_rules_from_patterns(self, pattern_counters: dict[int, Counter], current_pipe: list[int]):
        rules = []
        for window_size, counter in pattern_counters.items():
            for pattern, count in counter.most_common(3):
                if len(current_pipe) == 0:
                    continue
                if pattern[0] != current_pipe[-1]:
                    continue
                rules.append((pattern, count))
        return rules
    
    def rules_to_prompt(self, rules: list[tuple[tuple[int], int]]):
        rules_str = ''
        if len(rules) > 0:
            rules_str = 'For more context, we have extracted some frequent patterns and rules that may introduce improvements in the pipeline.\n'
            for i, (pattern, cnt) in enumerate(rules):
                rules_str += f'Rule {i + 1}, used {cnt} times: ('
                rules_str += ', '.join([f'{opid} {QActionProvider.get(opid).get_name()}' for opid in pattern])
                rules_str = rules_str[:-2] + ')\n'
        return rules_str

    def get_multiple_pipelines_with_experience(self,
                                            dataset_description: str,
                                            last_episode_records: list[PipeRecord],
                                            experience: list[Document]=[],
                                            exp_cnt=3):
        prompt_template = textwrap.dedent(
        '''
        You are an expert Data Preparation Pipeline Strategist integrated into LLaPipe, 
        a Q-Learning framework for automated data preparation pipeline construction. 
        Your Q-Learning agent is currently STUCK in a performance plateau, indicating 
        a local optimum. Your goal is to propose a **complete, strategic sequence of 
        data preparation operators (a new pipeline segment)** that, when applied from 
        the current data state, has a high potential to escape this local optimum and 
        lead to significantly better downstream model performance.

        Current Situation & Context:

        1.  Problem: The Q-Learning agent has hit a "performance plateau." Recent 
        explorations of single operator changes from the current pipeline tip have 
        not yielded significant improvement in the evaluation metric (accuracy).

        2.  Current Data State (st): {data_state}

        3.  Stagnant Pipeline (P_stuck) that led to the plateau:
            *   Operator sequence with the current performance: {current_pipeline}

        4.  Available Data Preparation Operators (O):
            *   List of available operator names: {available_operators}

        {experience}

        Based on the provided context, your task is to:

        1.  Propose some new Strategic Pipeline Segment (1-3 pipelines, the best one should be placed in front): 
            Design several new sequence of operators (e.g., 2 to 6 operators long, 
            each element is the operator name from the available 26 operators)
        2.  Provide Rationale (Briefly): Explain the reasoning behind your proposed 
            pipeline segment. Why do you believe this sequence has a high chance of 
            breaking the current local optimum?

        Output your response in the following JSON format, do not include comments:

        ```json
        [
        {{
        "no": 1,
        "pipe": ["operator_X", "operator_Y", "operator_Z"], // only include operator names, no comments
        "rationale": "This pipeline first [action of operator_X and why it's chosen for the current state/problem], then [action of operator_Y, building on X or addressing another aspect], and finally [action of operator_Z]."
        }},
        {{
        "no": 2,
        "pipe": ["operator_X", "operator_W", "operator_K"],
        "rationale": "This pipeline first [action of operator_X and why it's chosen for the current state/problem], ..."
        }},
        ...
        ]
        ```
        '''
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        operators = []
        # 将所有 action 整合为字符串
        for i, op in enumerate(QActionProvider.action_ids):
            operators.append(f'{i} {QActionProvider.get(op).get_name()}')
        operators = '\n'.join(operators)
        operators += '(LogisticRegression is the downstream task, choosing this action means to finish the pipeline)'

        recent_pipelines = []
        for i, record in enumerate(last_episode_records[-3:]):
            pipeline_str = ', '.join([QActionProvider.get(op_id).get_name() for op_id in record.pipe])
            recent_pipelines.append(f'pipeline {i + 1}: [{pipeline_str}], accuracy {record.accuracy:.4f}')

        recent_pipelines = '\n'.join(recent_pipelines)

        experience_str = ''
        if experience is not None and len(experience) > 0:
            experience_str = '5.  Experience (E), some of the pipelines with high performance: \n'
            for i, doc in enumerate(experience[:exp_cnt]):
                dataset_desc = doc.page_content
                metadata: dict[str, int | float | str] = doc.metadata
                if metadata['type'].startswith('act'):
                    experience_str += f'\nExample {i+1} {dataset_desc}\n action: {QActionProvider.get(metadata["act"]).get_name()}, accuracy {metadata["acc"]:.4f}, improvement {metadata["reward"]:.4f}\n'
                else:
                    pipe_str = ', '.join([f'{op_id} {QActionProvider.get(op_id).get_name()}' for op_id in metadata['pipe']])
                    experience_str += f'\nExample {i+1} {dataset_desc}\n pipeline: [{pipe_str}], accuracy {metadata["acc"]:.4f}, improvement {metadata["reward"]:.4f}\n'

        prompt = prompt.format(
            data_state=dataset_description,
            current_pipeline=recent_pipelines,
            available_operators=operators,
            experience=experience_str,
        )

        if self.output_prompt < 1:
            logger.info(prompt)
            self.output_prompt += 1

        try:
            llm_response = self.llm.invoke(prompt)
            if hasattr(llm_response, 'content'):
                llm_response = llm_response.content
            logger.debug(llm_response)
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            return []
        
        l_idx = llm_response.find('[')
        r_idx = llm_response.rfind(']')

        try:
            objs = tjson.tolerate(llm_response[l_idx:r_idx+1])
            for obj in objs:
                pipe = obj['pipe'].copy()
                if all(str(op_id).isdigit() for op_id in pipe):
                    pipe = [int(op_id) for op_id in pipe]
                    pipe = list(filter(lambda op_id: op_id in QActionProvider.action_ids, pipe))
                else:
                    logger.warning(f'LLM recommend pipeline operators are string')
                    pipe = [QActionProvider.str_to_idx(str(op_id).strip()) for op_id in pipe]
                obj['pipe'] = pipe
            all_pipelines: list[LlmReturnFullPipe] = [LlmReturnFullPipe.from_dict(obj) for obj in objs]
        except Exception as e:
            logger.error(llm_response)
            logger.error(f'Error parsing LLM response: {e}')
            logger.error(traceback.format_exc())
            all_pipelines = []
        return all_pipelines
