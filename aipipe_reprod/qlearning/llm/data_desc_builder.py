import os
import pandas as pd
import numpy as np
import time

def extract_dataset_features(dataset_path: str):
    """提取数据集特征"""
    start_time = time.time()
    features = {'path': dataset_path if isinstance(dataset_path, str) else ''}
    
    try:
        if isinstance(dataset_path, str):
            # 检查文件存在性和格式
            if not os.path.exists(dataset_path):
                return {'error': 'File not found', 'path': dataset_path}
            
            # 加载数据集
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
                df = pd.read_excel(dataset_path)
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)
            else:
                return {'error': 'Unsupported file format', 'path': dataset_path}
        elif isinstance(dataset_path, pd.DataFrame):
            df = dataset_path
        else:
            return {'error': 'Invalid dataset path', 'path': dataset_path}

        # 基本统计特征
        features['rows'] = df.shape[0]
        features['columns'] = df.shape[1]
        features['column_names'] = list(df.columns)
        features['memory_usage'] = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        
        # 列类型分析
        dtype_counts = {}
        features['column_dtypes'] = {}
        features['categorical_columns'] = []
        features['numeric_columns'] = []
        features['datetime_columns'] = []
        features['text_columns'] = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            features['column_dtypes'][col] = dtype
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            
            # 识别列类型
            if pd.api.types.is_numeric_dtype(df[col]):
                features['numeric_columns'].append(col)
            elif pd.api.types.is_datetime64_dtype(df[col]):
                features['datetime_columns'].append(col)
            # elif pd.api.types.is_categorical_dtype(df[col]):  # Deprecated since 2.2.0
            elif isinstance(df[col].dtype, pd.CategoricalDtype):
                features['categorical_columns'].append(col)
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                # 检查是否是分类特征还是文本特征
                unique_ratio = df[col].nunique() / df[col].count() if df[col].count() > 0 else 0
                if unique_ratio < 0.5 and df[col].nunique() < 100:
                    features['categorical_columns'].append(col)
                else:
                    features['text_columns'].append(col)
        
        features['dtype_counts'] = dtype_counts
        
        # 缺失值分析
        missing_counts = df.isna().sum()
        features['missing_values_total'] = int(missing_counts.sum())
        features['missing_values_ratio'] = float(missing_counts.sum() / (df.shape[0] * df.shape[1]))
        features['columns_with_missing'] = [col for col in df.columns if missing_counts[col] > 0]
        features['missing_by_column'] = {col: int(missing_counts[col]) for col in features['columns_with_missing']}
        
        # 数值特征统计
        features['numeric_stats'] = {}
        for col in features['numeric_columns']:
            features['numeric_stats'][col] = {
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'skew': float(df[col].skew()) if not pd.isna(df[col].skew()) else None,
                'unique_ratio': float(df[col].nunique() / df[col].count()) if df[col].count() > 0 else 0
            }
        
        # 分类特征统计
        features['categorical_stats'] = {}
        for col in features['categorical_columns']:
            features['categorical_stats'][col] = {
                'unique_values': int(df[col].nunique()),
                'top_value': str(df[col].mode()[0]) if not df[col].mode().empty else None,
                'top_freq': int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0,
                'unique_ratio': float(df[col].nunique() / df[col].count()) if df[col].count() > 0 else 0
            }
        
        # 相关性分析
        if 1 < len(features['numeric_columns']) <= 80:
            try:
                corr_matrix = df[features['numeric_columns']].corr().abs()
                # 找出高相关特征对
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_pairs = [(col1, col2, corr_matrix.loc[col1, col2]) 
                                 for col1 in upper_tri.index 
                                 for col2 in upper_tri.columns 
                                 if upper_tri.loc[col1, col2] > 0.8]
                features['high_correlation_pairs'] = [
                    {'col1': col1, 'col2': col2, 'corr': float(corr)} 
                    for col1, col2, corr in high_corr_pairs
                ]
            except Exception as e:
                features['correlation_error'] = str(e)
        
        # 数据分布特征
        features['skewed_columns'] = [
            col for col in features['numeric_columns'] 
            if abs(df[col].skew()) > 1.0 and not pd.isna(df[col].skew())
        ]
        
        # 异常值检测 (使用IQR方法)
        features['columns_with_outliers'] = []
        for col in features['numeric_columns']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers_count > 0:
                features['columns_with_outliers'].append({
                    'column': col,
                    'outliers_count': int(outliers_count),
                    'outliers_ratio': float(outliers_count / df.shape[0])
                })
        
        # 特征工程需求指标 (根据数据特征推断可能需要的处理步骤)
        features['needs_imputation'] = len(features['columns_with_missing']) > 0
        features['needs_scaling'] = any(
            features['numeric_stats'][col]['max'] > 10 * features['numeric_stats'][col]['min'] 
            for col in features['numeric_stats'] 
            if features['numeric_stats'][col]['min'] is not None and features['numeric_stats'][col]['max'] is not None
        )
        features['needs_encoding'] = len(features['categorical_columns']) > 0
        features['needs_outlier_treatment'] = len(features['columns_with_outliers']) > 0
        features['needs_normalization'] = len(features['skewed_columns']) > 0
        
        # 处理时间计算
        features['extraction_time'] = time.time() - start_time
        
        return features
        
    except Exception as e:
        return {
            'error': str(e),
            'path': dataset_path,
            'extraction_time': time.time() - start_time
        }


def create_dataset_description_en(features: dict, limit_k=3):
    """Create descriptive dataset summary"""
    description = []

    description.append(f"Dataset contains {features.get('rows', 0)} rows and {features.get('columns', 0)} columns.")

    description.append(
        f'Column type distribution: {len(features.get('numeric_columns', []))} numeric, '
        f"{len(features.get('categorical_columns', []))} categorical, " if len(features.get('categorical_columns', [])) > 0 else ""
        f"{len(features.get('datetime_columns', []))} datetime, " if len(features.get('datetime_columns', [])) > 0 else ""
        f"{len(features.get('text_columns', []))} text." if len(features.get('text_columns', [])) > 0 else ""
    )

    # Missing values
    if features.get('missing_values_total', 0) > 0:
        description.append(f"\nDataset has {features.get('missing_values_total')} missing values "
                            f"({features.get('missing_values_ratio', 0) * 100:.2f}%).")
        description.append("Columns with missing values:")
        cnt = 0
        for col, count in features.get('missing_by_column', {}).items():
            description.append(f"- {col}: {count} missing values")
            cnt += 1
            if cnt >= 20:
                break
    else:
        description.append("\nNo missing values in dataset.")

    # Outliers
    if features.get('columns_with_outliers'):
        description.append("\nColumns with outliers:")
        for item in features.get('columns_with_outliers', [])[:limit_k]:
            description.append(f"- {item['column']}: {item['outliers_count']} outliers "
                                f"({item['outliers_ratio'] * 100:.2f}%)")
        if len(features.get('columns_with_outliers')) > limit_k:
            description.append(f"...totally {len(features.get('columns_with_outliers'))} columns with outliers")

    # Skewed distribution
    if features.get('skewed_columns'):
        description.append("\nColumns with skewed distribution:")
        for col in features.get('skewed_columns', [])[:limit_k]:
            description.append(f"- {col}")
        if len(features.get('skewed_columns')) > limit_k:
            description.append(f"...totally {len(features.get('skewed_columns'))} columns with skewed distribution")

    # High correlation
    if features.get('high_correlation_pairs'):
        description.append("\nHighly correlated feature pairs:")
        for pair in features.get('high_correlation_pairs', [])[:limit_k]:
            description.append(f"- {pair['col1']} and {pair['col2']}: correlation {pair['corr']:.2f}")
        if len(features.get('high_correlation_pairs')) > limit_k:
            description.append(f"...totally {len(features.get('high_correlation_pairs'))} highly correlated feature pairs")

    # Feature engineering needs
    description.append("\nFeature engineering requirements:")
    eng_needs = []
    if features.get('needs_imputation'): eng_needs.append("missing value imputation")
    if features.get('needs_scaling'): eng_needs.append("scaling")
    if features.get('needs_encoding'): eng_needs.append("categorical encoding")
    if features.get('needs_outlier_treatment'): eng_needs.append("outlier treatment")
    if features.get('needs_normalization'): eng_needs.append("normalization")
    
    description.append("\nSimple feature engineering requirements: " + 
                      ("None" if not eng_needs else ", ".join(eng_needs)))

    # All column names
    description.append("\nColumn name list:")
    for col_type, cols in [
        ('Numeric columns', features.get('numeric_columns', [])),
        ('Categorical columns', features.get('categorical_columns', [])),
        ('Datetime columns', features.get('datetime_columns', [])),
        ('Text columns', features.get('text_columns', []))
    ]:
        if cols:
            description.append(f"\n{col_type}:")
            if len(cols) < limit_k:
                description.append(", ".join(list(map(lambda s: str(s), cols))))
            else:
                description.append(", ".join(list(map(lambda s: str(s), cols[:limit_k])) 
                                             + [f"...totally {len(cols)} columns"]))

    # # stats for numeric columns
    # if features.get('numeric_stats'):
    #     description.append("\nNumeric column statistics:")
    #     for col, stats in list(features.get('numeric_stats', {}).items())[:limit_k]:
    #         _mean = f"{stats.get('mean'):.3f}" if stats.get('mean') is not None else 'N/A'
    #         _std = f"{stats.get('std'):.3f}" if stats.get('std') is not None else 'N/A'
    #         _min = f"{stats.get('min'):.3f}" if stats.get('min') is not None else 'N/A'
    #         _median = f"{stats.get('median'):.3f}" if stats.get('median') is not None else 'N/A'
    #         _max = f"{stats.get('max'):.3f}" if stats.get('max') is not None else 'N/A'
    #         _unique_ratio = f"{stats.get('unique_ratio'):.3f}" if stats.get('unique_ratio') is not None else 'N/A'
    #         description.append(
    #             f"- {col}: "
    #             f"Mean: {_mean}, "
    #             f"Std: {_std}, "
    #             f"Min: {_min}, "
    #             f"Median: {_median}, "
    #             f"Max: {_max}, "
    #             f"Unique ratio: {_unique_ratio}"
    #         )
    #     if len(features.get('numeric_stats')) > limit_k:
    #         description.append(f"...totally {len(features.get('numeric_stats'))} numeric columns")

    # # stats for categorical columns
    # if features.get('categorical_stats'):
    #     description.append("\nCategorical column statistics:")
    #     for col, stats in list(features.get('categorical_stats', {}).items())[:limit_k]:
    #         _unique_ratio = f"{stats.get('unique_ratio'):.3f}" if stats.get('unique_ratio') is not None else 'N/A'
    #         description.append(
    #             f"- {col}: "
    #             f"Unique values: {stats.get('unique_values', 'N/A')}, "
    #             f"Top value: {stats.get('top_value', 'N/A')}, "
    #             f"Top frequency: {stats.get('top_freq', 'N/A')}, "
    #             f"Unique ratio: {_unique_ratio}"
    #         )
    #     if len(features.get('categorical_stats')) > limit_k:
    #         description.append(f"...totally {len(features.get('categorical_stats'))} categorical columns")

    return "\n".join(description)
