import pickle
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import warnings

from ..types import DatasetType as _Dataset


def convert_list(serie: pd.Series):
    """Converts lists in a pandas serie into a dataframe
    where which element of a list is a column
    Parameters
    ----------
    serie : pandas Serie
    The serie you want to cast into a dataframe
    Returns
    -------
    pandas DataFrame
    The converted dataframe
    """

    import numpy
    import pandas

    if (serie.apply(lambda x: type(x) == list).sum() > 0):

        serie = serie.apply(lambda x: [x] if type(x) != list else x)

        cut = int(numpy.percentile(serie.apply(len), 90))  # TODO: To test

        serie = serie.apply(lambda x: x[:cut])

        return pandas.DataFrame(serie.tolist(),
                                index=serie.index,
                                columns=[serie.name + "_item" + str(i + 1)
                                         for i in range(cut)]
                                )

    else:

        return serie


def convert_float_and_dates(serie: pd.Series):
    """Converts into float if possible and converts dates.
    Creates timestamp from 01/01/2017, year, month, day, day_of_week and hour
    Parameters
    ----------
    serie : pandas Serie
        The serie you want to convert
    Returns
    -------
    pandas DataFrame
        The converted dataframe
    """

    import pandas

    # dtype is already a date

    if (serie.dtype == 'datetime64[ns]'):

        df = pandas.DataFrame([], index=serie.index)

        df[serie.name + "_TIMESTAMP"] = (pandas.DatetimeIndex(serie) -
                                         pandas.datetime(2017, 1, 1)
                                         ).total_seconds()

        df[serie.name + "_YEAR"] = pandas.DatetimeIndex(serie).year.astype(
            float)

        df[serie.name + "_MONTH"] = pandas.DatetimeIndex(serie).month.astype(
            float)

        df[serie.name + "_DAY"] = pandas.DatetimeIndex(serie).day.astype(
            float)

        df[serie.name + "_DAYOFWEEK"] = pandas.DatetimeIndex(
            serie).dayofweek.astype(float)

        df[serie.name + "_HOUR"] = pandas.DatetimeIndex(serie).hour.astype(
            float) + pandas.DatetimeIndex(serie).minute.astype(float)/60. + \
            pandas.DatetimeIndex(serie).second.astype(float)/3600.

        return df

    else:

        # Convert float

        try:
            serie = serie.apply(float)

        except Exception:
            pass

        # Cleaning/converting dates

        if (serie.dtype != 'object'):
            return serie

        else:
            # trying to cast into date
            df = pandas.DataFrame([], index=serie.index)

            try:

                serie_to_df = pandas.DatetimeIndex(pd.to_datetime(serie))

                df[serie.name + "_TIMESTAMP"] = (serie_to_df -
                                                 pandas.datetime(2017, 1, 1)
                                                 ).total_seconds()

                df[serie.name + "_YEAR"] = serie_to_df.year.astype(
                    float)  # TODO: be careful with nan ! object or float??

                df[serie.name + "_MONTH"] = serie_to_df.month.astype(
                    float)  # TODO: be careful with nan ! object or float??

                df[serie.name + "_DAY"] = serie_to_df.day.astype(
                    float)  # TODO: be careful with nan ! object or float??

                df[serie.name + "_DAYOFWEEK"] = serie_to_df.dayofweek.astype(
                    float)  # TODO: be careful with nan ! object or float??

                df[serie.name + "_HOUR"] = serie_to_df.hour.astype(float) + \
                    serie_to_df.minute.astype(float)/60. + \
                    serie_to_df.second.astype(float) / 3600.

                return df

            except Exception:

                return serie


def profile_summary(dataset: pd.DataFrame, plot=False):

    pf = pd.DataFrame({'Attribute': "",
                       'Type': "",
                       'Num. Missing Values': [],
                       'Num. Unique Values': [],
                       'Sknewness': [],
                       'Kurtosis': []
                       })

    rows = []

    for attribute in list(dataset.select_dtypes(include=[
            np.number]).columns.values):

        att_type = dataset[attribute].dtype

        unique_values = pd.unique(dataset[attribute])

        num_missing = sum(pd.isnull(dataset[attribute]))

        sk = skew(dataset[attribute].values, axis=None, nan_policy='omit')

        ct = kurtosis(dataset[attribute].values, axis=None, nan_policy='omit')

        row = [attribute, att_type, num_missing, len(unique_values), sk, ct]

        rows.append(row)

    for attribute in list(dataset.select_dtypes(exclude=[
            np.number]).columns.values):

        att_type = dataset[attribute].dtype

        unique_values = pd.unique(dataset[attribute])

        num_missing = sum(pd.isnull(dataset[attribute]))

        sk = "N/A"

        ct = "N/A"

        row = [attribute, att_type, num_missing, len(unique_values), sk, ct]

        rows.append(row)

    for row in rows:

        pf.loc[len(pf)] = row

        if plot:

            print("Frequency plot per attribute")

            for attribute in dataset.columns:

                unique_values = pd.unique(dataset[attribute])

                num_missing = sum(pd.isnull(dataset[attribute]))

                print('Attribute: %s\nNumber of unique values: %d\nNumber '
                      'of missing values: '
                      '%d\nUnique values:' %
                      (attribute, len(unique_values), num_missing))

                print('\nFrequency plot:\n')

                d = (pd.DataFrame(dataset[attribute].value_counts()))

                ax = sns.barplot(x="index", y=attribute,
                                 data=(d).reset_index())

                ax.set(xlabel=attribute, ylabel='count')

                ax.grid(b=True, which='major', color='w', linewidth=1.0)

                ax.set_xticklabels(
                    labels=d.sort_index().index.values, rotation=90)

                plt.show()

    print("Profiling datasets")

    print(pf.to_string())


class Reader():

    """Reads and profile the data

    Parameters
    ----------
    * sep : str, defaut = None
        Delimiter to use when reading a csv file.

    * header : int or None, default = 0.
        If header=0, the first line is considered as a header.
        Otherwise, there is no header.
        Useful for csv and xls files.

    * to_hdf5 : bool, default = True
        If True, dumps each file to hdf5 format.

    * to_path : str, default = "save"
        Name of the folder where files and encoders are saved.

    * verbose : bool, defaut = False
        Verbose mode

    * encoding: Boolean, default = 'False' otherwise convert
        categorical variable target for classification to int64
    """

    def __init__(self, sep=None, header=0, to_hdf5=False,
                 to_path="./save",
                 verbose=False,
                 encoding=False):

        self.sep = sep

        self.header = header

        self.to_hdf5 = to_hdf5

        self.to_path = to_path

        self.verbose = verbose

        self.encoding = encoding

    def profile(self, path, plot=False):
        """Reads and profile data (accepted formats : csv, xls,
            json and h5):
        - del Unnamed columns
        - casts lists into variables
        - try to cast variables into float
        - format dates and extracts timestamp from 01/01/2017,
        year, month, day, day_of_week and hour
        - count numbers of missing values et list unique values
        - plot the value frequency histogram per attribute
        Parameters
        ----------
        path : str
            The path to the dataset.
        plot : bool, default = False

        Returns
        -------
        pandas dataframe
            Formated dataset.
        """

        ##############################################################
        #                           Reading
        ##############################################################

        start_time = time.time()

        if (path is None):

            raise ValueError("You must specify the path to load the data")

        else:

            type_doc = path.split(".")[-1]

            if (type_doc == 'csv'):

                if (self.sep is None):

                    raise ValueError("You must specify the separator "
                                     "for a csv file")
                else:

                    if (self.verbose):

                        print("")

                        print("Reading csv : " + path.split("/")[-1] + " ...")

                    df = pd.read_csv(path,
                                     sep=self.sep,
                                     header=self.header,
                                     engine='c',
                                     on_bad_lines='warn',
                                     encoding='ISO-8859-1')

            elif (type_doc == 'xls'):

                if (self.verbose):

                    print("")

                    print("Reading xls : " + path.split("/")[-1] + " ...")

                df = pd.read_excel(path, header=self.header)

            elif (type_doc == 'h5'):

                if (self.verbose):

                    print("")

                    print("Reading hdf5 : " + path.split("/")[-1] + " ...")

                df = pd.read_hdf(path)

            elif (type_doc == 'json'):

                if (self.verbose):

                    print("")

                    print("Reading json : " + path.split("/")[-1] + " ...")

                df = pd.read_json(path)

            else:

                raise ValueError("The document extension cannot be handled")

        # Deleting unknown column (That is, the first index column)

        try:

            del df["Unnamed: 0"]

        except Exception:

            pass

        ##############################################################
        #             Cleaning lists, floats and dates
        ##############################################################

        if (self.verbose):
            print("Reading data ...")

            df = pd.concat(Parallel(n_jobs=-1)(delayed(convert_list)(df[col])
                           for col in df.columns), axis=1)

            df = pd.concat(Parallel(n_jobs=-1)(
                           delayed(convert_float_and_dates)(df[col])
                           for col in df.columns),
                           axis=1)

        if (self.verbose):

            print("CPU time: %s seconds" % (time.time() - start_time))
            profile_summary(df, plot=False)

        return df

    def train_test_split(self, Lpath, target_name=None, encoding=False) -> _Dataset:
        """Creates train and test datasets
        Given a list of several paths and a target name,
        automatically creates and cleans train and test datasets.
        IMPORTANT: a dataset is considered as a test set if it does
        not contain the target value. Otherwise it is
        considered as part of a train set.
        Also determines the task and encodes the target (classification
            problem only).
        Finally dumps the datasets to hdf5, and eventually the target
        encoder.
        Parameters
        ----------
        Lpath : list, defaut = None
            List of str paths to load the data
        target_name : str, default = None
            The name of the target. Works for both classification
            (multiclass or not) and regression.
        Returns
        -------
        dict
            Dictionnary containing :
            - 'train' : pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : encoded pandas Serie for the target
            on train set (with dtype='float' for a regression or dtype='int'
                for a classification)
        """

        col = []

        col_train = []

        col_test = []

        df_train = dict()

        df_test = dict()

        y_train = dict()

        y_test = dict()

        if (type(Lpath) != list):

            raise ValueError("You must specify a list of paths "
                             "to load all the data")

        elif (self.to_path is None):

            raise ValueError("You must specify a path to save your data "
                             "and make sure your files are not already saved")
        else:

            ##############################################################
            #                    Reading the files
            ##############################################################
            if (len(Lpath) == 1):

                from sklearn.model_selection import train_test_split

                df = self.profile(Lpath[0])

                if (target_name in df.columns):

                    df_train, df_test, y_train, y_test = train_test_split(
                        df, df[target_name], test_size=0.33)
                    
                    df_train: pd.DataFrame = df_train.infer_objects()
                    df_test: pd.DataFrame = df_test.infer_objects()
                    y_train: pd.DataFrame = y_train.infer_objects()
                    y_test: pd.DataFrame = y_test.infer_objects()

                elif (target_name is None):

                    df = df

                else:

                    raise ValueError(
                        "Please check that the target name is correct.")

            elif (len(Lpath) == 2):

                for path in Lpath:

                    # Reading each file

                    df = self.profile(path)

                    # Checking if the target exists to split into
                    # test and train
                    if (target_name in df.columns):

                        is_null = df[target_name].isnull()

                        df_train[path] = df[~is_null].drop(target_name, axis=1)
                        y_test[path] = df[target_name][is_null]
                        df_test[path] = df[~is_null].drop(target_name, axis=1)

                        y_train[path] = df[target_name][~is_null]

                        # y_test[path] = y_train[path]
                        # print(df_train[path].shape[0],df_test[path].shape[0])

                    elif (target_name is None):

                        raise ValueError(
                            "The target name cannot be 'None' if you "
                            "provide a training and a testing dataset. ")

                    else:

                        df_test[path] = df

                del df

            # Finding the common subset of features

                for i, df in enumerate(df_train.values()):

                    if (i == 0):

                        col_train = df.columns

                    else:

                        col_train = list(set(col_train) & set(df.columns))

                for i, df in enumerate(df_test.values()):

                    if (i == 0):

                        col_test = df.columns

                    else:

                        col_test = list(set(col_test) & set(df.columns))

            # Subset of common features

                col = sorted(list(set(col_train) & set(col_test)))

                if (self.verbose):

                    print("")

                    print("> Number of common features : " + str(len(col)))

                    print("")

                    print("gathering and crunching for train and "
                          "test datasets ...")

                df_test = pd.concat([df[col] for df in df_test.values()]).infer_objects()

                df_train = pd.concat([df[col] for df in df_train.values()]).infer_objects()

                y_train = pd.concat([y for y in y_train.values()]).infer_objects()

                y_test = pd.concat([y for y in y_test.values()]).infer_objects()

                if (type(y_train) == pd.core.frame.DataFrame):

                    raise ValueError("Your target contains more than "
                                     "two columns!"
                                     " Please check that only one column "
                                     "is named " + target_name)

                elif (type(y_test) == pd.core.frame.DataFrame):

                    raise ValueError("Your target contains more than "
                                     "two columns!"
                                     " Please check that only one column "
                                     "is named " + target_name)

                else:

                    pass

                # Handling indices
                if (self.verbose) and (target_name is not None):

                    print("reindexing for train and test datasets ...")

                if (df_train.index.nunique() < df_train.shape[0]):

                    df_train.index = range(df_train.shape[0])

                if (df_test.index.nunique() < df_test.shape[0]):

                    df_test.index = range(df_test.shape[0])

                if (y_train.index.nunique() < y_train.shape[0]):

                    y_train.index = range(y_train.shape[0])

                if (y_test.index.nunique() < y_test.shape[0]):

                    y_test.index = range(y_test.shape[0])

                if (df.index.nunique() < df.shape[0]):

                    df.index = range(df.shape[0])

            else:

                raise ValueError(
                    "Please check that the path is correct (two files max.).")

                ##############################################################
                #          Creating train, test and target dataframes
                ##############################################################

            if (target_name is None):

                sparse_features = (df.isnull().sum() *
                                   100. / df.shape[0]
                                   ).sort_values(ascending=False)
                sparse = True

                if(sparse_features.max() == 0.0):

                    sparse = False

            else:

                sparse_features = (df_train.isnull().sum() *
                                   100. / df_train.shape[0]
                                   ).sort_values(ascending=False)

                sparse = True

                if(sparse_features.max() == 0.0):

                    sparse = False

            # Print information
            if (self.verbose):

                if (len(Lpath) == 2):

                    print("")

                    print("> Number of categorical features "
                          "in the training set:"
                          " " + str(len(df_train.dtypes[df_train.dtypes ==
                                                        'object'].index)))

                    print("> Number of numerical features "
                          "in the training set:"
                          " " + str(len(df_train.dtypes[df_train.dtypes !=
                                                        'object'].index)))

                    print("> Number of training samples : " +
                          str(df_train.shape[0]))

                    print("> Number of test samples : " +
                          str(df_test.shape[0]))

                    if(sparse):

                        print("")

                        print("> Top sparse features "
                              "(% missing values on train set):")

                        print(np.round(sparse_features[sparse_features >
                              0.0][:5], 1))

                    else:

                        print("")

                        print("> You have no missing values on train set...")

                elif (len(Lpath) == 1):

                    print("")

                    print("> Number of categorical features "
                          "in the training set:"
                          " " + str(len(df.dtypes[df.dtypes ==
                                                  'object'].index)))

                    print("> Number of numerical features in the training set:"
                          " " + str(len(df.dtypes[df.dtypes !=
                                                  'object'].index)))

                    print("> Number of data samples : " + str(df.shape[0]))

                    if(sparse):

                        print("")

                        print("> Top sparse features "
                              "(% missing values on dataset set):")

                        print(np.round(sparse_features[sparse_features
                                                       > 0.0][:5], 1))

                    else:

                        print("")

                        print("> You have no missing values in the dataset...")

                else:

                    pass

            ##############################################################
            #                    Encoding target
            ##############################################################
            # if (len(Lpath) ==1):
            #     df_test={}

            if (target_name is None):

                task = "clustering"

                df_train = df.select_dtypes(['number'])

                y_train = {}

                y_test = {}

                df_test = {}

            else:

                task = "regression"

                if (y_train.nunique() <= 2):

                    task = "classification"
                # else:

                if (y_train.dtype == object):

                    task = "classification"

                else:

                    # no needs to convert into float
                    pass

            if (self.verbose):

                print("")

                print("> Task : " + task)

            if (task == "classification"):

                if (self.verbose):

                    print(y_train.value_counts())

                    print("")

                    if self.encoding:  # and (y_train.dtype == object)

                        enc = LabelEncoder()

                        print("Encoding target...")

                        y_train = y_train.astype(str)

                        y_train = pd.Series(enc.fit_transform(y_train.values),
                                            index=y_train.index,
                                            name=target_name,
                                            dtype='int')

                        y_test = y_test.astype(str)

                        y_test = pd.Series(enc.fit_transform(y_test.values),
                                           index=y_test.index,
                                           name=target_name,
                                           dtype='int')

                        print("Encoding target done...")

                        df_train[target_name] = y_train.values

                        if (target_name in df_test.columns.values):
                            df_test[target_name] = y_test.values

            elif (task == "regression"):

                if (self.verbose):

                    print("Target description:\n ", target_name,
                          "\n ", y_train.describe())

            else:

                if (self.verbose):

                    print("Numerical feature description:\n ",
                          df_train.describe())
            ##############################################################
            #                         Dumping
            ##############################################################

            # Creating a folder to save the files and target encoder
            try:

                os.mkdir(self.to_path)

            except OSError:

                pass

            if (self.to_hdf5):

                start_time = time.time()

                if (self.verbose):

                    print("")

                    print("dumping files into directory : " + self.to_path)

                # Temp adding target to dump train file...

                df_train[target_name] = y_train.values

                df_train.to_hdf(self.to_path + '/df_train.h5', 'train')

                del df_train[target_name]

                if (self.verbose):

                    print("train dumped")

                df_test.to_hdf(self.to_path + '/df_test.h5', 'test')

                if (self.verbose):

                    print("test dumped")

                    print("CPU time: %s seconds" % (time.time() - start_time))

            else:

                pass

            if (task == "classification"):

                fhand = open(self.to_path + '/target_encoder.obj', 'wb')

                enc = LabelEncoder()

                pickle.dump(enc, fhand)

                fhand.close()

            else:

                pass

            return {"train": df_train,
                    "test": df_test,
                    "target": y_train,
                    "target_test": y_test}

    def train_test_split2(self, path, target_name, to_drop=None, shuffle=True, rand_state=42) -> _Dataset:
        if isinstance(path, str):
            df = pd.read_csv(path, sep=self.sep, header=self.header).infer_objects()
        elif isinstance(path, list):
            # merge all files into one
            df = pd.concat([pd.read_csv(p, sep=self.sep, header=self.header) for p in path], axis=0)
        else:
            raise ValueError("You must specify a path or a list of paths to load the data")

        if (to_drop is not None):
            df = df.drop(to_drop, axis=1)

        # Get all non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        
        # Apply LabelEncoder to each non-numeric column
        # for col in non_numeric_cols:
        #     enc = LabelEncoder()
        #     df[col] = enc.fit_transform(df[col].astype(str))

        if target_name is None:
            target_name = df.columns[-1]

        y = df[target_name]
        y = pd.Series(pd.factorize(y)[0])

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(df.drop(target_name, axis=1), y, test_size=0.2, random_state=rand_state, shuffle=shuffle)

        # X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        X_train: pd.DataFrame = X_train.reset_index(drop=True)
        X_test: pd.DataFrame = X_test.reset_index(drop=True)
        y_train: pd.Series = y_train.reset_index(drop=True)
        y_test: pd.Series = y_test.reset_index(drop=True)

        # X_validation: pd.DataFrame = X_validation.reset_index(drop=True)
        # y_validation: pd.Series = y_validation.reset_index(drop=True)

        task = "regression"
        count = y_train.nunique()

        if (count <= 2):
            task = "classification"

        else:
            if (y_train.dtype == object):
                task = "classification"
            else:
                # no needs to convert into float
                pass
    
        if (task == "classification"):
            if (self.verbose):
                print(y_train.value_counts())
                print("")
                print("encoding target ...")
            # enc = LabelEncoder()
            # y_train = pd.Series(enc.fit_transform(y_train.values),
            #                     index=y_train.index,
            #                     name=target_name,
            #                     dtype='int')
            # y_test = pd.Series(enc.transform(y_test.values),
            #                     index=y_test.index,
            #                     name=target_name,
            #                     dtype='int')
            # y_validation = pd.Series(enc.transform(y_validation.values),
            #                     index=y_validation.index,
            #                     name=target_name,
            #                     dtype='int')

            if count == 1:
                warnings.warn("Your target set has only one class ! Please check it is correct, "
                                "otherwise there is no need to use this tool...")

        else:
            if (self.verbose):
                print(y_train.describe())

        # if (task == "classification"):
        #     fhand = open(self.to_path + '/target_encoder.obj', 'wb')
        #     pickle.dump(enc, fhand)
        #     fhand.close()

        return {
            "train": X_train,
            "test": X_test,
            'target': y_train,
            'target_test': y_test,
            'target_name': target_name,
            # 'validation': X_validation,
            # 'validation_target': y_validation,
        }
    