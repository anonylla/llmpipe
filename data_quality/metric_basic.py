import pandas as pd
import numpy as np
import re
from datetime import datetime
import argparse
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# import gensim.downloader as api
from typing import Tuple, Dict, Union

from utils.data import safe_div

def parse_args():
	arg_parser =argparse.ArgumentParser()
	arg_parser.add_argument("--data",type=str)
	args = arg_parser.parse_args()
	return args

def load_data(file_path):
    """
    加载文件数据
    """
    # 根据文件格式选择合适的加载方法，如 pd.read_csv()、pd.read_excel() 等
    # data = pd.read_csv(file_path)
    data = pd.read_json(file_path, typ="series")
    return data


def check_uniqueness(df):
    """
    检查是否符合独特性
    数据项根据其本身或另一个数据集或数据库中的对应项进行测量。
    与数据集中的事物记录数量相比，分析“现实世界”中评估的事物数量，现实世界的事物数量可以通过不同且可能更可靠的数据集或相关的外部比较器来确定。
    """


def check_completeness(df: pd.DataFrame) -> float:
    """
    检查是否符合完整性
    衡量是否存在空白（null 或空字符串）值或是否存在非空白值的度量。
    """
    # print("检查数据完整性中……")
    total_values = df.size  # 总数值数量
    # 计算空白值（null 或空字符串）的数量
    missing_values = df.isnull().sum().sum() + (df == '').sum().sum()
    # print("空白值数量：", missing_values)
    # 计算空白值占比
    if total_values > 0:
        missing_ratio = missing_values / total_values
    else:
        missing_ratio = 0.
    completeness_score = (1 - missing_ratio) * 100
    # print("完整性得分为：", completeness_score)
    return completeness_score

def check_datetime_column(column_data: pd.Series):
    # 尝试将列数据转换为日期时间类型
    try:
        pd.to_datetime(column_data)
        return True
    except ValueError:
        pass
    
    # 使用正则表达式匹配日期和时间格式的字符串
    patterns = [
        r'\d{4}-\d{2}-\d{2}',  # yyyy-mm-dd
        r'\d{2}-\d{2}-\d{2}',  # yy-mm-dd
        r'\d{2}/\d{2}/\d{4}',  # mm/dd/yyyy
        r'\d{2}/\d{2}/\d{2}',  # mm/dd/yy
        r'\d{4}年\d{2}月\d{2}日',  # yyyy年mm月dd日
        r'\d{2}年\d{2}月\d{2}日',  # yy年mm月dd日
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # yyyy-mm-dd hh:mm:ss
        r'\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # yy-mm-dd hh:mm:ss
        r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',  # mm/dd/yyyy hh:mm:ss
        r'\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}',  # mm/dd/yy hh:mm:ss
        r'\d{4}年\d{2}月\d{2}日 \d{2}:\d{2}:\d{2}',  # yyyy年mm月dd日 hh:mm:ss
        r'\d{2}年\d{2}月\d{2}日 \d{2}:\d{2}:\d{2}',  # yy年mm月dd日
    ]
    for pattern in patterns:
        if re.match(pattern, str(column_data)):
            return True
    
    # 判断数据范围和分布
    # 这里可以根据具体情况添加其他判断条件

    return False

def is_datetime_column(column_data: pd.Series):
    # 检查数据中是否有90%以上的数据符合时间数据的特征
    total_count = len(column_data)
    datetime_count = sum(check_datetime_column(value) for value in column_data)
    return safe_div(datetime_count, total_count) >= 0.9

def check_timeliness(df: pd.DataFrame) -> float:
    """
    检查是否符合时效性
    计算数据中时间戳列与当前时间的差值，相差3天以上开始扣分，每多1天扣1分。
    """
    # print("检查数据时效性中……")
    current_date = pd.Timestamp.now()
    # print("当前日期为：", current_date)
    timeliness_scores = []
    timeliness_count = 0
    
    for column in df.columns:
        # 检查是否为时间戳列
        if is_datetime_column(df[column]):
            # 将字符串数据转换为日期时间格式
            df[column] = pd.to_datetime(df[column], errors='coerce')

            # 计算每个数据的时间差（以天为单位）
            lags = (current_date - df[column]).dt.days
            if lags > 3:
                timeliness_count += 1
            # 计算每个数据的得分
            scores = 100 - (lags - 3).clip(lower=0).clip(upper=97)  # 最多扣除97分，以保证最低分为0

            # 计算该列数据的平均得分
            column_score = scores.mean()
            timeliness_scores.append(column_score)
    # print("超过3天未更新的数据量：", timeliness_count)
    # 计算所有列的平均得分
    if len(timeliness_scores) > 0:
        overall_timeliness_score = sum(timeliness_scores) / len(timeliness_scores)
    else:
        overall_timeliness_score = 0.
    # print("时效性得分为：", overall_timeliness_score)
    return overall_timeliness_score


def check_validity(df: pd.DataFrame) -> float:
    """
    检查是否符合规范性
    关于允许类型（字符串、整数、浮点等）、格式（长度、位数等）和范围（最小值、最大值或包含在一组允许值内）的数据库、元数据或文档规则。
    """
    # print("检查数据规范性中……")
    total_values = df.size  # 总数据数量
    invalid_values_count = 0  # 不符合规范的数据数量

    for column in df.columns:
        for index, value in df[column].items():
            # 检查字符串类型
            if isinstance(value, str):
                # 检查字符串格式
                if not re.match(r'^[A-Za-z0-9_]+$', value):
                    invalid_values_count += 1
            # 检查日期类型
            elif pd.api.types.is_datetime64_any_dtype(value):
                # 检查日期格式
                try:
                    pd.to_datetime(value, format='%Y-%m-%d', errors='raise')
                except ValueError:
                    invalid_values_count += 1
    # print("不符合规范性的数据量：", invalid_values_count)
    invalid_ratio = safe_div(invalid_values_count, total_values)
    validity_score = (1 - invalid_ratio) * 100
    # print("规范性得分为：", validity_score)
    return validity_score

def is_numeric(s):
    # 判断是否为数值型数据
    pattern = r'^[+-]?\d*\.?\d*(?:[+-:]\d*\.?\d*)?%?$'  # 匹配数值+符号的正则表达式
    return bool(re.match(pattern, str(s)))

def has_letters(s):
    # 判断是否包含字母
    # return bool(re.search('[a-zA-Z]', str(s)))
    try:
        float(s)
        return False
    except ValueError:
        return True

def split_numeric_text_data(df: pd.DataFrame):
    # 将数据分为数值和文本两部分
    numeric_data = df.select_dtypes(include=['number'])
    text_data = df.select_dtypes(include=['object'])
    return numeric_data, text_data

def detect_numeric_outliers(data: pd.DataFrame):
    """
    使用 K-means 聚类算法检测异常值
    Args:
    - data: 数据集，numpy 数组或 pandas DataFrame
    - n_clusters: 聚类数，默认为 2
    Returns:
    - outliers: 异常值的索引列表
    """
    outliers_num_count = 0
    data_ = data.copy().dropna()
    for column in data_.columns:
        column_values = data_[column]
        for index, value in column_values.items():
            if(has_letters(value)): # 删除含有字符的数值，并将其视为异常值
                outliers_num_count += 1
                column_values.drop(index, inplace=True)
                data_.drop(index, inplace=True)
        # 对删除后的数据进行聚类
        if len(column_values) > 0:
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(column_values.values.reshape(-1, 1))
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            # 计算每个样本到其所属簇中心的距离
            cleaned_column_values = pd.to_numeric(column_values, errors='coerce').dropna()
            distances = np.linalg.norm(cleaned_column_values.values.reshape(-1, 1) - centers[labels], axis=1)
                
            # 根据距离判断异常值
            threshold = np.percentile(distances, 95)  # 取距离的 95% 分位数作为阈值
            outliers = np.where(distances > threshold)[0]
            outliers_num_count += len(outliers)  # 将异常值数量加到outliers_num_count中
    
    return outliers_num_count


def iqr_numeric_outliers(data: pd.DataFrame, gap=1.5):
    """
    使用 IQR 法检测异常值
    Args:
    - data: 数据集，numpy 数组或 pandas DataFrame
    Returns:
    - outliers_num_count: 异常值的数量
    """
    outliers_num_count = 0
    data_ = data.copy().fillna(0)
    for column in data_.columns:
        column_values = data_[column]
        q1 = np.percentile(column_values, 25)
        q3 = np.percentile(column_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - gap * iqr
        upper_bound = q3 + gap * iqr
        outliers = column_values[(column_values < lower_bound) | (column_values > upper_bound)]
        outliers_num_count += len(outliers)
        column_values.drop(outliers.index, inplace=True)
        data_.drop(outliers.index, inplace=True)
    return outliers_num_count

def detect_text_outliers(text_data: pd.DataFrame, word2vec_model):
    """
    利用词向量模型进行语义分析，检测文本数据中的异常值
    Args:
    - text_data: 文本数据，DataFrame 格式，每列是文本数据
    Returns:
    - outliers_ratio: 异常值在总数据中的比率
    """
    # 计算文本数据的平均词向量
    def calculate_mean_vector(text):
        text = str(text)
        words = text.split()
        word_vectors = [word2vec_model[word] for word in words if word in word2vec_model]
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(word2vec_model.vector_size)

    total_data_points = text_data.size
    outliers_count = 0
    
    # 遍历每列数据
    for column in text_data.columns:
        text_vectors = text_data[column].apply(calculate_mean_vector)
            
        # 使用DBSCAN算法进行聚类
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(StandardScaler().fit_transform(text_vectors.tolist()))

        # 计算异常值数量
        outliers_count += np.sum(clusters == -1)

    return outliers_count

def check_accuracy(df: pd.DataFrame) -> float:
    # print("检查数据准确性中……")
    # 分离数值和文本数据
    numeric_data, text_data = split_numeric_text_data(df)

    # 检测数值数据异常值
    # numeric_outliers = detect_numeric_outliers(numeric_data)
    numeric_outliers = iqr_numeric_outliers(numeric_data)

    # 检测文本数据异常值
    # word2vec_model = api.load("word2vec-google-news-300")
    # text_outliers = detect_text_outliers(text_data, word2vec_model)
    outliers = numeric_outliers # + text_outliers
    # print("异常数据量：", outliers)
    total_data_count = df.shape[0] * df.shape[1]
    outliers_radio = safe_div(outliers, total_data_count)
    accuracy_score = (1 - outliers_radio) * 100
    # print("准确性得分为：",accuracy_score)
    return accuracy_score

def check_consistency(df: pd.DataFrame):
    """
    检查是否符合一致性
    将事物的两个或多个表示与定义进行比较时，没有差异。
    应该是对于两个或多个表之间的数据是否一致进行评判。
    """    


def check_data_quality(df: pd.DataFrame) -> Dict[str, Union[float]]:
    """
    数据质量检测
    对于所有数据集进行完整性、规范性、准确性的检测
    对于有时间列的数据集，增加时效性的检测
    """

    results = {}
    # 检查是否符合完整性
    results['completenessScore'] = check_completeness(df)
    # 检查是否符合规范性
    results['validityScore'] = check_validity(df)
    # 检查是否符合准确性
    results['accuracyScore'] = check_accuracy(df)
    # 检查是否符合时效性
    if 'timestamp_column' in df.columns:
        results['timelinessScore'] = check_timeliness(df)
        results['totalScore'] = (results['completenessScore'] + results['validityScore'] + results['accuracyScore'] + results['timelinessScore']) / 4.
    else:
        results['timelinessScore'] = None
        results['totalScore'] = (results['completenessScore'] + results['validityScore'] + results['accuracyScore']) / 3.
    # print(results)
    return results

if __name__ == '__main__':
    args = parse_args()
    # 加载数据文件
    file_path = args.data
    data = load_data(file_path)
    df = pd.DataFrame(data)

    # 数据质量检测
    quality_results = check_data_quality(df)
    
    print(json.dumps(quality_results))
