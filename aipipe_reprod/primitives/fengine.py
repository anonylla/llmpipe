from loguru import logger
from .primitive import Primitive
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA, TruncatedSVD, FastICA, NMF, LatentDirichletAllocation
from sklearn.ensemble import RandomTreesEmbedding
import pandas as pd
import numpy as np
from copy import deepcopy
import warnings
from typing import Tuple, Dict, List, Any, Optional

from .encoders import split_cat_and_num_cols

np.random.seed(42)


class PolynomialFeaturesPrim(Primitive):
    def __init__(self, random_state: int = 0) -> None:
        super(PolynomialFeaturesPrim, self).__init__(name='PolynomialFeatures')
        self.id = 1
        self.gid = 17
        self.hyperparams: List[Any] = []
        self.type = 'FeatureEngine'
        self.description = ("Generate polynomial and interaction features. Generate a new feature "
                            "matrix consisting of all polynomial combinations of the features with "
                            "degree less than or equal to the specified degree. For example, if an input "
                            "sample is two dimensional and of the form [a, b], the degree-2 polynomial "
                            "features are [1, a, b, a^2, ab, b^2].")
        self.scaler = PolynomialFeatures(include_bias=False)
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data: pd.DataFrame) -> bool:
        if data.shape[1] > 100:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data: pd.DataFrame) -> bool:
        return True

    def transform(self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        if train_x.shape[1] > 100:
            return train_x, test_x, train_y
        
        cat_train_x, num_train_x = split_cat_and_num_cols(train_x)
        cat_test_x = test_x[cat_train_x.columns]
        num_test_x = test_x[num_train_x.columns]
        # cat_test_x, num_test_x = split_cat_and_num_cols(test_x)
        
        num_train_x.fillna(1, inplace=True)
        num_test_x.fillna(1, inplace=True)
        self.scaler.fit(num_train_x)

        num_train_x = pd.DataFrame(self.scaler.transform(num_train_x)).reset_index(drop=True).infer_objects()
        num_test_x = pd.DataFrame(self.scaler.transform(num_test_x)).reset_index(drop=True).infer_objects()

        if cat_train_x.shape[1] == 0:
            train_data_x = num_train_x.reset_index(drop=True)
            test_data_x = num_test_x.reset_index(drop=True)
        else:
            # 合并处理后的数值列和原始分类列
            train_data_x = pd.concat([
                cat_train_x.reset_index(drop=True), 
                num_train_x.reset_index(drop=True)
            ], axis=1)
            
            test_data_x = pd.concat([
                cat_test_x.reset_index(drop=True),
                num_test_x.reset_index(drop=True)
            ], axis=1)

        train_data_x = train_data_x.loc[:, ~train_data_x.columns.duplicated()]

        test_data_x = test_data_x.loc[:, ~test_data_x.columns.duplicated()]
        return train_data_x, test_data_x, train_y


class InteractionFeaturesPrim(Primitive):
    def __init__(self, random_state: int = 0) -> None:
        super(InteractionFeaturesPrim, self).__init__(name='InteractionFeatures')
        self.id = 2
        self.gid = 18
        self.hyperparams: List[Any] = []
        self.type = 'FeatureEngine'
        self.description = "Generate interaction features. [1, a, b, ab]"
        self.scaler = PolynomialFeatures(interaction_only=True, include_bias=False)
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data: pd.DataFrame) -> bool:
        if data.shape[1] > 100:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data: pd.DataFrame) -> bool:
        return True

    def transform(self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        if train_x.shape[1] > 100:
            return train_x, test_x, train_y
        
        cat_train_x, num_train_x = split_cat_and_num_cols(train_x)
        cat_test_x = test_x[cat_train_x.columns]
        num_test_x = test_x[num_train_x.columns]
        # cat_test_x, num_test_x = split_cat_and_num_cols(test_x)
        
        num_train_x.fillna(1, inplace=True)
        num_test_x.fillna(1, inplace=True)
        self.scaler.fit(num_train_x)

        num_train_x = pd.DataFrame(self.scaler.transform(num_train_x)).reset_index(drop=True).infer_objects()
        num_test_x = pd.DataFrame(self.scaler.transform(num_test_x)).reset_index(drop=True).infer_objects()

        if cat_train_x.shape[1] == 0:
            train_data_x = num_train_x.reset_index(drop=True)
            test_data_x = num_test_x.reset_index(drop=True)
        else:
            # 合并处理后的数值列和原始分类列
            train_data_x = pd.concat([
                cat_train_x.reset_index(drop=True), 
                num_train_x.reset_index(drop=True)
            ], axis=1)
            
            test_data_x = pd.concat([
                cat_test_x.reset_index(drop=True),
                num_test_x.reset_index(drop=True)
            ], axis=1)


        train_data_x = train_data_x.loc[:, ~train_data_x.columns.duplicated()]

        test_data_x = test_data_x.loc[:, ~test_data_x.columns.duplicated()]
        return train_data_x, test_data_x, train_y


class PCA_AUTO_Prim(Primitive):
    def __init__(self, random_state: int = 0) -> None:
        super(PCA_AUTO_Prim, self).__init__(name='PCA_AUTO')
        self.id = 3
        self.gid = 19
        self.PCA_AUTO_Prim: List[Any] = []
        self.type = 'FeatureEngine'
        self.description = ("LAPACK principal component analysis (PCA). Linear dimensionality "
                            "reduction using Singular Value Decomposition of the data to project "
                            "it to a lower dimensional space. It uses the LAPACK implementation "
                            "of the full SVD or a randomized truncated SVD by the method of Halko "
                            "et al. 2009, depending on the shape of the input data and the number of components to extract.")
        self.pca = PCA(svd_solver='auto')  # n_components=0.9
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data: pd.DataFrame) -> bool:
        can_num = len(data.columns) > 4
        return self.can_accept_c(data) and can_num

    def is_needed(self, data: pd.DataFrame) -> bool:
        return True

    def transform(self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        # 分离数值列和分类列
        cat_train_x, num_train_x = split_cat_and_num_cols(train_x)
        cat_test_x = test_x[cat_train_x.columns]
        num_test_x = test_x[num_train_x.columns]
        # cat_test_x, num_test_x = split_cat_and_num_cols(test_x)
        
        # 检查数值列数量是否足够进行PCA
        if num_train_x.shape[1] <= 2:
            return train_x, test_x, train_y
            
        # 处理缺失值
        num_train_x.fillna(0, inplace=True)
        num_test_x.fillna(0, inplace=True)
        
        try:
            # 只对数值列进行PCA变换
            self.pca.fit(num_train_x)
            num_train_x = pd.DataFrame(self.pca.transform(num_train_x), columns=[f'PC{i+1}' for i in range(self.pca.n_components_)])
            num_test_x = pd.DataFrame(self.pca.transform(num_test_x), columns=[f'PC{i+1}' for i in range(self.pca.n_components_)])
            
            if cat_train_x.shape[1] == 0:
                train_data_x = num_train_x.reset_index(drop=True)
                test_data_x = num_test_x.reset_index(drop=True)
            else:
                # 合并处理后的数值列和原始分类列
                train_data_x = pd.concat([
                    cat_train_x.reset_index(drop=True), 
                    num_train_x.reset_index(drop=True)
                ], axis=1)
                
                test_data_x = pd.concat([
                    cat_test_x.reset_index(drop=True),
                    num_test_x.reset_index(drop=True)
                ], axis=1)
            
            return train_data_x, test_data_x, train_y
        except Exception as e:
            logger.error(f'PCA error: {e}')
            return train_x, test_x, train_y


class IncrementalPCA_Prim(Primitive):
    def __init__(self, random_state: int = 0) -> None:
        super(IncrementalPCA_Prim, self).__init__(name='IncrementalPCA')
        self.id = 5
        self.gid = 20
        self.PCA_LAPACK_Prim: List[Any] = []
        self.type = 'FeatureEngine'
        self.description = ("Incremental principal components analysis (IPCA). Linear dimensionality "
                            "reduction using Singular Value Decomposition of centered data, keeping only "
                            "the most significant singular vectors to project the data to a lower dimensional space. "
                            "Depending on the size of the input data, this algorithm can be much more memory "
                            "efficient than a PCA. This algorithm has constant memory complexity.")
        self.hyperparams_run = {'default': True}
        self.pca = IncrementalPCA()
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data: pd.DataFrame) -> bool:
        return self.can_accept_c(data)

    def is_needed(self, data: pd.DataFrame) -> bool:
        return data.shape[1] > 4

    def transform(self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        cat_train_x, num_train_x = split_cat_and_num_cols(train_x)
        cat_test_x = test_x[cat_train_x.columns]
        num_test_x = test_x[num_train_x.columns]
        # cat_test_x, num_test_x = split_cat_and_num_cols(test_x)
        
        if num_train_x.shape[1] <= 2:
            return train_x, test_x, train_y
            
        if num_train_x.shape[1] <= 2:
            return train_x, test_x, train_y
            
        num_train_x.fillna(0, inplace=True)
        num_test_x.fillna(0, inplace=True)
        
        try:
            self.pca.fit(num_train_x)
            num_train_x = pd.DataFrame(self.pca.transform(num_train_x), columns=[f'PC{i+1}' for i in range(self.pca.n_components_)])
            num_test_x = pd.DataFrame(self.pca.transform(num_test_x), columns=[f'PC{i+1}' for i in range(self.pca.n_components_)])
            
            if cat_train_x.shape[1] == 0:
                train_data_x = num_train_x.reset_index(drop=True)
                test_data_x = num_test_x.reset_index(drop=True)
            else:
                # 合并处理后的数值列和原始分类列
                train_data_x = pd.concat([
                    cat_train_x.reset_index(drop=True), 
                    num_train_x.reset_index(drop=True)
                ], axis=1)
                
                test_data_x = pd.concat([
                    cat_test_x.reset_index(drop=True),
                    num_test_x.reset_index(drop=True)
                ], axis=1)
            
            return train_data_x, test_data_x, train_y
        except Exception as e:
            logger.error(f'IncrementalPCA_Prim: {e}')
            return train_x, test_x, train_y


class KernelPCA_Prim(Primitive):
    def __init__(self, random_state: int = 0) -> None:
        super(KernelPCA_Prim, self).__init__(name='KernelPCA')
        self.id = 6
        self.gid = 21
        self.PCA_LAPACK_Prim: List[Any] = []
        self.type = 'FeatureEngine'
        self.description = "Kernel Principal component analysis (KPCA). Non-linear dimensionality reduction through the use of kernels"
        self.n_components = 2
        self.pca = KernelPCA(n_components=self.n_components)  # n_components=5
        self.accept_type = 'c_t_krnl'
        self.random_state = random_state
        self.need_y = False

    def can_accept(self, data: pd.DataFrame) -> bool:
        if data.shape[1] <= 2:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data: pd.DataFrame) -> bool:
        return data.shape[1] > 4

    def transform(self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        try:
            cat_train_x, num_train_x = split_cat_and_num_cols(train_x)
            cat_test_x = test_x[cat_train_x.columns]
            num_test_x = test_x[num_train_x.columns]
            # cat_test_x, num_test_x = split_cat_and_num_cols(test_x)
            
            if num_train_x.shape[1] <= 2:
                return train_x, test_x, train_y
                
            num_train_x.fillna(0, inplace=True)
            num_test_x.fillna(0, inplace=True)
            
            self.pca.fit(num_train_x)

            num_train_x = pd.DataFrame(self.pca.transform(num_train_x), columns=[f'PC{i+1}' for i in range(self.n_components)])
            num_test_x = pd.DataFrame(self.pca.transform(num_test_x), columns=[f'PC{i+1}' for i in range(self.n_components)])
            
            if cat_train_x.shape[1] == 0:
                train_data_x = num_train_x.reset_index(drop=True)
                test_data_x = num_test_x.reset_index(drop=True)
            else:
                # 合并处理后的数值列和原始分类列
                train_data_x = pd.concat([
                    cat_train_x.reset_index(drop=True), 
                    num_train_x.reset_index(drop=True)
                ], axis=1)
                
                test_data_x = pd.concat([
                    cat_test_x.reset_index(drop=True),
                    num_test_x.reset_index(drop=True)
                ], axis=1)

            return train_data_x, test_data_x, train_y
        except Exception as e:
            logger.error(f'KernelPCA_Prim error: {e}')
            return train_x, test_x, train_y


class TruncatedSVD_Prim(Primitive):
    def __init__(self, random_state: int = 0) -> None:
        super(TruncatedSVD_Prim, self).__init__(name='TruncatedSVD')
        self.id = 7
        self.gid = 22
        self.PCA_LAPACK_Prim: List[Any] = []
        self.type = 'FeatureEngine'
        self.description = ("Dimensionality reduction using truncated SVD (aka LSA). "
                            "This transformer performs linear dimensionality reduction by means "
                            "of truncated singular value decomposition (SVD). Contrary to PCA, "
                            "this estimator does not center the data before computing the singular "
                            "value decomposition. This means it can work with scipy.sparse matrices "
                            "efficiently. In particular, truncated SVD works on term count/tf-idf "
                            "matrices as returned by the vectorizers in sklearn.feature_extraction.text. "
                            "In that context, it is known as latent semantic analysis (LSA). This estimator "
                            "supports two algorithms: a fast randomized SVD solver, and a \"naive\" algorithm "
                            "that uses ARPACK as an eigensolver on (X * X.T) or (X.T * X), whichever is more efficient.")
        self.hyperparams_run = {'default': True}
        self.n_components = 2
        self.pca = TruncatedSVD(n_components=self.n_components)
        self.accept_type = 'c_t_krnl'
        self.need_y = False

    def can_accept(self, data: pd.DataFrame) -> bool:
        if data.shape[1] <= 2:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data: pd.DataFrame) -> bool:
        return data.shape[1] > 4

    def transform(self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        cat_train_x, num_train_x = split_cat_and_num_cols(train_x)
        cat_test_x = test_x[cat_train_x.columns]
        num_test_x = test_x[num_train_x.columns]
        # cat_test_x, num_test_x = split_cat_and_num_cols(test_x)
        
        if num_train_x.shape[1] <= 2:
            return train_x, test_x, train_y
            
        num_train_x.fillna(0, inplace=True)
        num_test_x.fillna(0, inplace=True)
        
        try:
            self.pca.fit(num_train_x)
            num_train_x = pd.DataFrame(self.pca.transform(num_train_x), columns=[f'PC{i+1}' for i in range(self.n_components)])
            num_test_x = pd.DataFrame(self.pca.transform(num_test_x), columns=[f'PC{i+1}' for i in range(self.n_components)])
            
            if cat_train_x.shape[1] == 0:
                train_data_x = num_train_x.reset_index(drop=True)
                test_data_x = num_test_x.reset_index(drop=True)
            else:
                # 合并处理后的数值列和原始分类列
                train_data_x = pd.concat([
                    cat_train_x.reset_index(drop=True), 
                    num_train_x.reset_index(drop=True)
                ], axis=1)
                
                test_data_x = pd.concat([
                    cat_test_x.reset_index(drop=True),
                    num_test_x.reset_index(drop=True)
                ], axis=1)

            return train_data_x, test_data_x, train_y
        except Exception as e:
            logger.error(f'TruncatedSVD error: {e}')
            return train_x, test_x, train_y


class RandomTreesEmbeddingPrim(Primitive):
    def __init__(self, random_state: int = 0) -> None:
        super(RandomTreesEmbeddingPrim, self).__init__(name='RandomTreesEmbedding')
        self.id = 8
        self.gid = 23
        self.PCA_LAPACK_Prim: List[Any] = []
        self.type = 'FeatureEngine'
        self.description = ("An ensemble of totally random trees. "
                            "An unsupervised transformation of a dataset to a high-dimensional sparse representation. "
                            "A datapoint is coded according to which leaf of each tree it is sorted into. "
                            "Using a one-hot encoding of the leaves, this leads to a binary coding with as many ones as there are trees in the forest.")
        self.hyperparams_run = {'default': True}
        self.pca = RandomTreesEmbedding(random_state=random_state)
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data: pd.DataFrame) -> bool:
        return self.can_accept_c(data)

    def is_needed(self, data: pd.DataFrame) -> bool:
        return data.shape[1] < 10

    def transform(self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        if train_x.shape[1] > 100:
            return train_x, test_x, train_y
        
        cat_train_x, num_train_x = split_cat_and_num_cols(train_x)
        cat_test_x = test_x[cat_train_x.columns]
        num_test_x = test_x[num_train_x.columns]
        # cat_test_x, num_test_x = split_cat_and_num_cols(test_x)
        
        # 只对数值列进行处理
        num_train_x.fillna(0, inplace=True)
        num_test_x.fillna(0, inplace=True)
        
        # 检查是否有数值列
        if num_train_x.shape[1] == 0:
            return train_x, test_x, train_y
            
        self.pca.fit(num_train_x)
        
        # 转换数值列
        train_embeddings = self.pca.transform(num_train_x).toarray()
        test_embeddings = self.pca.transform(num_test_x).toarray()
        
        # 创建新列名
        new_cols = [f'embed_{i}' for i in range(train_embeddings.shape[1])]
        
        if cat_train_x.shape[1] == 0:
            train_data_x = pd.DataFrame(train_embeddings, columns=new_cols)
            test_data_x = pd.DataFrame(test_embeddings, columns=new_cols)
        else:
            # 合并处理后的数值列和原始分类列
            train_data_x = pd.concat([
                cat_train_x.reset_index(drop=True),
                pd.DataFrame(train_embeddings, columns=new_cols)
            ], axis=1)
            
            test_data_x = pd.concat([
                cat_test_x.reset_index(drop=True),
                pd.DataFrame(test_embeddings, columns=new_cols)
            ], axis=1)
        
        return train_data_x, test_data_x, train_y


class PCA_ARPACK_Prim(Primitive):
    def __init__(self, random_state: int = 0) -> None:
        super(PCA_ARPACK_Prim, self).__init__(name='PCA_ARPACK')
        self.id = 4
        self.gid = 24
        self.PCA_LAPACK_Prim: List[Any] = []
        self.type = 'FeatureEngine'
        self.description = "ARPACK principal component analysis (PCA). Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract."
        self.n_components = 2
        self.pca = PCA(svd_solver='arpack', n_components=self.n_components)
        self.accept_type = 'c_t_arpck'
        self.need_y = False

    def can_accept(self, data: pd.DataFrame) -> bool:
        can_num = len(data.columns) > 4
        return self.can_accept_c(data) and can_num

    def is_needed(self, data: pd.DataFrame) -> bool:
        return data.shape[1] > 4

    def transform(self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        cat_train_x, num_train_x = split_cat_and_num_cols(train_x)
        cat_test_x = test_x[cat_train_x.columns]
        num_test_x = test_x[num_train_x.columns]
        # cat_test_x, num_test_x = split_cat_and_num_cols(test_x)
        
        if num_train_x.shape[1] <= 2:
            return train_x, test_x, train_y
            
        num_train_x.fillna(0, inplace=True)
        num_test_x.fillna(0, inplace=True)
        try:
            self.pca.fit(num_train_x)
            num_train_x = pd.DataFrame(self.pca.transform(num_train_x), columns=[f'PC{i+1}' for i in range(self.n_components)])
            num_test_x = pd.DataFrame(self.pca.transform(num_test_x), columns=[f'PC{i+1}' for i in range(self.n_components)])
            
            if cat_train_x.shape[1] == 0:
                train_data_x = num_train_x.reset_index(drop=True)
                test_data_x = num_test_x.reset_index(drop=True)
            else:
                # 合并处理后的数值列和原始分类列
                train_data_x = pd.concat([
                    cat_train_x.reset_index(drop=True), 
                    num_train_x.reset_index(drop=True)
                ], axis=1)
                
                test_data_x = pd.concat([
                    cat_test_x.reset_index(drop=True),
                    num_test_x.reset_index(drop=True)
                ], axis=1)

            return train_data_x, test_data_x, train_y
        except Exception as e:
            logger.error(f'PCA_ARPACK_Prim: {e}')
            return train_x, test_x, train_y

class PCA_LAPACK_Prim(Primitive):
    def __init__(self, random_state: int = 0) -> None:
        super(PCA_LAPACK_Prim, self).__init__(name='PCA_LAPACK')
        self.id = 3
        self.gid = 25
        self.PCA_LAPACK_Prim: List[Any] = []
        self.type = 'FeatureEngine'
        self.description = "LAPACK principal component analysis (PCA). Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract."
        self.pca = PCA(svd_solver='full')  # n_components=0.9
        self.accept_type = 'c_t'
        self.need_y = False

    def can_accept(self, data: pd.DataFrame) -> bool:
        can_num = len(data.columns) > 4
        return self.can_accept_c(data) and can_num

    def is_needed(self, data: pd.DataFrame) -> bool:
        return data.shape[1] > 4

    def transform(self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series):
        train_x.columns = train_x.columns.astype(str)
        test_x.columns = test_x.columns.astype(str)
        cat_train_x, num_train_x = split_cat_and_num_cols(train_x)
        cat_test_x = test_x[cat_train_x.columns]
        num_test_x = test_x[num_train_x.columns]
        # cat_test_x, num_test_x = split_cat_and_num_cols(test_x)
        
        if num_train_x.shape[1] <= 2:
            return train_x, test_x, train_y
            
        num_train_x.fillna(0, inplace=True)
        num_test_x.fillna(0, inplace=True)
        
        try:
            self.pca.fit(num_train_x)
            num_train_x = pd.DataFrame(self.pca.transform(num_train_x), columns=[f'PC{i+1}' for i in range(self.pca.n_components_)])
            num_test_x = pd.DataFrame(self.pca.transform(num_test_x), columns=[f'PC{i+1}' for i in range(self.pca.n_components_)])
            
            if cat_train_x.shape[1] == 0:
                train_data_x = num_train_x.reset_index(drop=True)
                test_data_x = num_test_x.reset_index(drop=True)
            else:
                # 合并处理后的数值列和原始分类列
                train_data_x = pd.concat([
                    cat_train_x.reset_index(drop=True), 
                    num_train_x.reset_index(drop=True)
                ], axis=1)
                
                test_data_x = pd.concat([
                    cat_test_x.reset_index(drop=True),
                    num_test_x.reset_index(drop=True)
                ], axis=1)

            return train_data_x, test_data_x, train_y
        except Exception as e:
            logger.error(f'PCA_LAPACK_Prim: {e}')
            return train_x, test_x, train_y

