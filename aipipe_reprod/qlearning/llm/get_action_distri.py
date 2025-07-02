import pandas as pd
import traceback

from loguru import logger

from .llm_provider import LlmProvider
from .data_desc_builder import create_dataset_description_en, extract_dataset_features
from ...new_ql.q_action_provider import QActionProvider


def get_dataset_description(df_name: str,
                            df: pd.DataFrame, limited_k=3):
    df_description = f'Raw Dataset: {df_name}\n'
    df_description += create_dataset_description_en(extract_dataset_features(df), limit_k=limited_k)
    return df_description

def get_actions_description(actions: list[int]):
    actions_text = "(op_id. Name):\n"
    for action_id in actions:
        action = QActionProvider.get(action_id)
        if action: # 确保动作存在
            actions_text += f"{action_id}. {action.get_name()}\n"
    
    if QActionProvider.done_action in actions:
        actions_text += '(LogisticRegression is the downstream task, choosing this action means to finish the pipeline)'
        
    return actions_text

def get_distribution_from_candidates(
        llm: LlmProvider,
        df_name: str,
        df: pd.DataFrame,
        pipe: list[int],
        total_reward: float,
        accuracy: float,
        candidates: list[int],
):
    df_description = get_dataset_description(df_name, df)
    df_description += f'\nThe pipeline is [ {",".join(map(lambda x: f"{x} {QActionProvider.get(x).get_name()}", pipe))} ], with total reward {total_reward:.4f} and task accuracy {accuracy:.4f}'
    
    prob_llm = None

    try:
        prob_llm = llm.get_distributions_from_candidates(df_description, candidates)
    except Exception as e:
        prob_llm = None
        traceback.print_exc()

    return prob_llm


def get_last_episode_distribution(
        llm: LlmProvider,
        df_name: str,
        df: pd.DataFrame,
        pipe: list[int],
        total_reward: float,
        accuracy: float,
):
    llm_retry_times = 3
    df_description = get_dataset_description(df_name, df)
    df_description += f'\nThe pipeline is [ {",".join(map(lambda x: f"{x} {QActionProvider.get(x).get_name()}", pipe))} ], with total reward {total_reward:.4f} and task accuracy {accuracy:.4f}'
    
    prob_llm = None

    for i in range(llm_retry_times):
        try:
            prob_llm = llm.summarize_and_get_operator_distribution(df_description)
            if len(prob_llm.keys()) == QActionProvider.n_action:
                break

            logger.warning(f'LLM failed to summarize and get operator distribution, with len {len(prob_llm)}, times {i}, retrying...')
        except Exception as e:
            prob_llm = None
            traceback.print_exc()
            logger.error(f'API call failed, {e}, retrying...')
    
    return prob_llm


def get_step_distribution(
        llm: LlmProvider,
        df_name: str,
        df: pd.DataFrame,
        pipe: list[int],
        accuracy_list: float,
):
    llm_retry_times = 3
    df_description = get_dataset_description(df_name, df)
    df_description += '\nIn the current episode, the steps are:'
    for action, accuracy in zip(pipe, accuracy_list):
        df_description += f'\n{action} {QActionProvider.get(action).get_name()}: {accuracy:.4f}'

    prob_llm = None

    for i in range(llm_retry_times):
        try:
            prob_llm = llm.summarize_and_get_operator_distribution(df_description)
            if len(prob_llm.keys()) == QActionProvider.n_action:
                break

            logger.warning(f'LLM failed to summarize and get operator distribution, with len {len(prob_llm)}, times {i}, retrying...')
        except Exception as e:
            prob_llm = None
            traceback.print_exc()
            logger.error(f'API call failed, {e}, retrying...')
    
    return prob_llm

