import pandas as pd
import numpy as np
import os
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class DataFeatures:
    FEATURE_GROUPS = {
        'basic': [
            'has_missing',
            'global_missing_rate',
            'max_column_missing_rate',
            'numeric_variance_mean',
            'categorical_cardinality_mean',
            'duplicate_row_rate',
            'df_complexity',
            'df_categorical_entropy',
            'num_cols',
        ],
        'numeric_stats': [
            'num_numeric_cols',
            'numeric_mean_of_means',
            'numeric_mean_of_stds', 
            'numeric_mean_of_skew',
        ],
        'categorical_stats': [
            'num_categorical_cols',
            'mean_unique_values',
        ],
        'correlation_stats': [
            'mean_absolute_correlation',
            'correlation_std',
        ]
    }

    @classmethod
    def FEATURE_METHODS(cls):
        return [
            *cls.FEATURE_GROUPS['basic'],
            *cls.FEATURE_GROUPS['numeric_stats'],
            *cls.FEATURE_GROUPS['categorical_stats'],
            *cls.FEATURE_GROUPS['correlation_stats']
        ]
    
    @classmethod
    def has_missing(cls, df: pd.DataFrame):
        return df.isna().any().any()

    @classmethod
    def global_missing_rate(cls, df: pd.DataFrame):
        """Calculate the global missing rate of a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            float: Percentage of missing values across the entire DataFrame
        """
        if df.empty:
            return 0.0
            
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        
        return missing_cells / total_cells

    @classmethod
    def max_column_missing_rate(cls, df: pd.DataFrame):
        """Calculate the maximum missing rate among all columns in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            float: Maximum percentage of missing values in any single column
        """
        if df.empty:
            return 0.0
            
        # Calculate missing rate for each column
        missing_rates = df.isna().mean()
        
        # Return the maximum missing rate
        return missing_rates.max()

    @classmethod
    def numeric_variance_mean(cls, df: pd.DataFrame):
        """Calculate the mean variance of numeric columns in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            float: Mean variance across all numeric columns
        """
        if df.empty:
            return 0.0
            
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            return 0.0
            
        # Calculate variance for each numeric column
        variances = []
        for col in numeric_cols:
            try:
                variance = df[col].var()
                if not np.isnan(variance):
                    variances.append(variance)
            except:
                continue
                
        # Return mean variance if we have any valid variances
        return np.mean(variances) if variances else 0.0

    @classmethod
    def df_categorical_entropy(cls, df: pd.DataFrame):
        """Calculate the mean entropy of categorical columns in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            float: Mean entropy across all categorical columns
        """
        if df.empty:
            return 0.0
            
        # Get categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) == 0:
            return 0.0
            
        # Calculate entropy for each categorical column
        entropies = []
        for col in cat_cols:
            try:
                # Get value counts and calculate probabilities
                value_counts = df[col].value_counts(normalize=True)
                # Calculate entropy: -sum(p * log2(p))
                entropy = -np.sum(value_counts * np.log2(value_counts))
                if not np.isnan(entropy):
                    entropies.append(entropy)
            except:
                continue
                
        # Return mean entropy if we have any valid entropies
        return np.mean(entropies) if entropies else 0.0


    @classmethod
    def df_complexity(cls, df: pd.DataFrame):
        """Calculate the complexity of a DataFrame using PCA.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            float: Number of principal components needed to explain 95% variance
        """
        if df.empty:
            return 0.0
            
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            return 0.0
            
        try:
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols])
            
            # Perform PCA
            pca = PCA()
            pca.fit(scaled_data)
            
            # Calculate cumulative explained variance ratio
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            
            # Find number of components needed for 95% variance
            n_components = np.argmax(cumsum >= 0.95) + 1
            
            return float(n_components)
            
        except:
            return 0.0

    @classmethod
    def num_rows(cls, df: pd.DataFrame) -> int:
        """Calculate the number of rows in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            int: Number of rows
        """
        return df.shape[0]

    @classmethod
    def num_cols(cls, df: pd.DataFrame) -> int:
        """Calculate the number of columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            int: Number of columns
        """
        return df.shape[1]

    @classmethod
    def num_numeric_cols(cls, df: pd.DataFrame) -> int:
        """Calculate the number of numeric columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            int: Number of numeric columns
        """
        if df.empty:
            return 0
        return df.select_dtypes(include=['number']).shape[1]

    @classmethod
    def num_categorical_cols(cls, df: pd.DataFrame) -> int:
        """Calculate the number of categorical columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            int: Number of categorical columns (object or category dtype)
        """
        if df.empty:
            return 0
        return df.select_dtypes(include=['object', 'category']).shape[1]

    @classmethod
    def categorical_cardinality_mean(cls, df: pd.DataFrame) -> float:
        """Calculate the mean cardinality (number of unique values) of categorical columns.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            float: Mean cardinality across all categorical columns
        """
        if df.empty:
            return 0.0

        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) == 0:
            return 0.0

        cardinalities = []
        for col in cat_cols:
            try:
                cardinality = df[col].nunique()
                cardinalities.append(cardinality)
            except Exception: # Handle potential errors during nunique calculation
                continue

        return np.mean(cardinalities) if cardinalities else 0.0

    @classmethod
    def duplicate_row_rate(cls, df: pd.DataFrame) -> float:
        """Calculate the proportion of duplicate rows in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            float: Rate of duplicate rows (between 0 and 1)
        """
        if df.empty:
            return 0.0

        num_duplicates = df.duplicated().sum()
        total_rows = len(df)

        return num_duplicates / total_rows if total_rows > 0 else 0.0

    @classmethod
    def num_numeric_cols(cls, df: pd.DataFrame) -> int:
        """Calculate the number of numeric columns in the DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame
        Returns:
            int: Number of numeric columns
        """
        if df.empty:
            return 0
        return df.select_dtypes(include=['number']).shape[1]

    @classmethod
    def numeric_mean_of_means(cls, df: pd.DataFrame) -> float:
        """Calculate the mean of the means of numeric columns.
        Args:
            df (pd.DataFrame): Input DataFrame
        Returns:
            float: Mean of the means of numeric columns
        """
        if df.empty:
            return 0.0
        numeric_cols = df.select_dtypes(include=['number']).columns
        return df[numeric_cols].mean().mean() if len(numeric_cols) > 0 else 0.0

    @classmethod
    def numeric_mean_of_stds(cls, df: pd.DataFrame) -> float:
        """Calculate the mean of the standard deviations of numeric columns.
        Args:
            df (pd.DataFrame): Input DataFrame
        Returns:
            float: Mean of the standard deviations of numeric columns
        """
        if df.empty:
            return 0.0
        numeric_cols = df.select_dtypes(include=['number']).columns
        return df[numeric_cols].std().mean() if len(numeric_cols) > 0 else 0.0

    @classmethod
    def numeric_mean_of_skew(cls, df: pd.DataFrame) -> float:
        """Calculate the mean of the skewness of numeric columns.
        Args:
            df (pd.DataFrame): Input DataFrame
        Returns:
            float: Mean of the skewness of numeric columns
        """
        if df.empty:
            return 0.0
        numeric_cols = df.select_dtypes(include=['number']).columns
        return df[numeric_cols].skew().mean() if len(numeric_cols) > 0 else 0.0

    @classmethod
    def num_categorical_cols(cls, df: pd.DataFrame) -> int:
        """Calculate the number of categorical columns in the DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame
        Returns:
            int: Number of categorical columns
        """
        if df.empty:
            return 0
        return df.select_dtypes(include=['object', 'category']).shape[1]

    @classmethod
    def mean_unique_values(cls, df: pd.DataFrame) -> float:
        """Calculate the mean number of unique values across all categorical columns.
        Args:
            df (pd.DataFrame): Input DataFrame
        Returns:
            float: Mean number of unique values across all categorical columns
        """
        if df.empty:
            return 0.0
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        return df[cat_cols].nunique().mean() if len(cat_cols) > 0 else 0.0

    @classmethod
    def mean_absolute_correlation(cls, df: pd.DataFrame) -> float:
        """Calculate the mean absolute correlation between numeric columns.
        Args:
            df (pd.DataFrame): Input DataFrame
        Returns:
            float: Mean absolute correlation between numeric columns
        """
        if df.empty:
            return 0.0
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            return 0.0
        corr_matrix = df[numeric_cols].corr()
        return corr_matrix.abs().mean().mean()

    @classmethod
    def correlation_std(cls, df: pd.DataFrame) -> float:
        """Calculate the standard deviation of correlations between numeric columns.
        Args:
            df (pd.DataFrame): Input DataFrame
        Returns:
            float: Standard deviation of correlations between numeric columns
        """
        if df.empty:
            return 0.0
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            return 0.0
        corr_matrix = df[numeric_cols].corr()
        return corr_matrix.abs().std().mean()

    @classmethod
    def all_features(cls, df: pd.DataFrame) -> np.ndarray:
        """Extract all available features from the DataFrame and combine them into a single vector.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            np.ndarray: Vector containing all extracted features
        """
        features = []
        
        for method_name in cls.FEATURE_METHODS():
            method = getattr(cls, method_name)
            try:
                feature_value = method(df)
                features.append(feature_value)
            except Exception as e:
                logging.error(f"Error extracting feature {method_name}: {e}")
                features.append(0.0)  # Default to 0.0 if an error occurs
        
        feature_vector = np.array(features)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
        
        return feature_vector


class Feature2RuleTable:
    target_col_name = 'operator_id'
    def __init__(self, 
                 load_from_file: bool = False, 
                 filepath: str = "save/data_rule_qtables/rule_table.csv", 
                 initial_data=None):
        self.table = pd.DataFrame(columns=DataFeatures.FEATURE_METHODS() + [self.target_col_name])
        if load_from_file:
            self.load(filepath)

        # 使用初始数据进行初始化
        if initial_data is not None and len(self.table) == 0:
            for df, operator_id in initial_data:
                feature_vector = DataFeatures.all_features(df)
                self.put(feature_vector, operator_id)

    def __len__(self):
        return len(self.table)

    def put(self, feature_vector: np.ndarray, suggested_pipe: list[int]):
        """Add a new rule to the table."""
        if len(feature_vector) != len(DataFeatures.FEATURE_METHODS()):
            raise ValueError(f"Feature vector length {len(feature_vector)} does not match expected length {len(DataFeatures.FEATURE_METHODS())}")
        
        if len(self.table) >= 10000:
            # randomly remove 10% of the rules
            self.table = self.table.sample(frac=0.9)

        new_rule = pd.DataFrame([feature_vector], columns=DataFeatures.FEATURE_METHODS())
        ### 直接用 list[int] 报错
        new_rule[self.target_col_name] = self.encode_suggested_pipe(suggested_pipe)
        self.table = pd.concat([self.table, new_rule], ignore_index=True)
        self.table.drop_duplicates(inplace=True)

    def encode_suggested_pipe(self, pipe: list[int]):
        """Encode the suggested pipe into a single integer."""
        return '|'.join(map(str, pipe))
    
    def decode_suggested_pipe(self, encoded_pipe: int) -> list[int]:
        """Decode the encoded pipe back into a list of integers."""
        return [int(op_id) for op_id in str(encoded_pipe).split('|')]

    def get_one(self, feature_vector: np.ndarray, k: int = 10):
        """Find the closest rule in the table based on feature vector."""
        if len(feature_vector) != len(DataFeatures.FEATURE_METHODS()):
            raise ValueError(f"Feature vector length {len(feature_vector)} does not match expected length {len(DataFeatures.FEATURE_METHODS())}")
        if self.table.empty:
            # >>> return -1
            return [-1] * k
        # Calculate the Euclidean distance between the feature vector and all rules in the table
        distances = np.linalg.norm(self.table[DataFeatures.FEATURE_METHODS()].values - feature_vector, axis=1)
        # Find the index of the rule with the smallest distance
        closest_rule_index = np.argmin(distances)
        # Return the operator_id of the closest rule
        suggested_pipe_str =  self.table.iloc[closest_rule_index][self.target_col_name]
        return self.decode_suggested_pipe(suggested_pipe_str)

    def get_top_k(self, feature_vector: np.ndarray, k: int = 10) -> list[int]:
        """Find the top-k closest rules in the table based on feature vector."""
        if len(feature_vector) != len(DataFeatures.FEATURE_METHODS()):
            raise ValueError(f"Feature vector length {len(feature_vector)} does not match expected length {len(DataFeatures.FEATURE_METHODS())}")
        if self.table.empty:
            return [-1] * k
        # TODO 后续还得进行向量搜索的优化以及大语言模型和 RAG 的结合
        # Calculate the Euclidean distance between the feature vector and all rules in the table
        distances = np.linalg.norm(self.table[DataFeatures.FEATURE_METHODS()].values - feature_vector, axis=1)
        # Find the indices of the top-k rules with the smallest distances
        top_k_indices = np.argsort(distances)[:k]
        # Return the operator_ids of the top-k rules
        ### 按照上面那样 put 导致这里出问题了，后续要改得记得这里要同步一下
        return self.table.iloc[top_k_indices][self.target_col_name].tolist()

    def save(self, filepath="save/data_rule_qtables/rule_table.csv"):
        """Saves the rule table to a file."""
        try:
            # Create parent directories if they don't exist
            parent_dir = os.path.dirname(filepath)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            self.table.to_csv(filepath, index=False)
            logger.info(f"Rule table saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving rule table: {e}")

    def load(self, filepath="save/data_rule_qtables/rule_table.csv"):
        """Loads the rule table from a file."""
        if os.path.exists(filepath):
            try:
                # self.table = pd.read_csv(filepath)
                self.table = pd.read_csv(filepath)
                logger.info(f"Rule table loaded from {filepath} with shape {self.table.shape}")
            except Exception as e:
                logger.error(f"Error loading rule table: {e}. Reinitializing.")
                self.table = pd.DataFrame(columns=DataFeatures.FEATURE_METHODS() + [self.target_col_name])
        else:
            logger.warning(f"Rule table file not found at {filepath}. Initializing new rule table.")
            self.table = pd.DataFrame(columns=DataFeatures.FEATURE_METHODS() + [self.target_col_name])
