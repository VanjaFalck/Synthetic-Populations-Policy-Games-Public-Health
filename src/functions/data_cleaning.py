# -*- coding: utf-8 -*-
"""
Preprocess datasets for synthetic population generation

Deliver a pandas dataframe ready for input in deep generative
models

"""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import yaml


class DataClean:
    """Class DataClean

    All preps of the data as pandas dataframe for running
    synthetic population generation.

    Configuration: yaml file with variable names to include in data preprocessing

    Parameters to constructor:
    datafile .................. the name/path of the original dataset or a prepared recategorised df
    configuration_file ........ yaml file: variable names set in set_configuration()
    limit ..................... limit for missing column values. Columns with more than
                                limit missing values are removed. Default = 0.5 (50 percent)
    prepared .................. When True, no imputation is done and a prepared datafile should be passed
                                Default = False

    Public functions:
    get_data()................. return pandas dataframe with cleaned data

    """
    def __init__(self,
                 # Path to data and configuration files:
                 datafile=None,  # must be a recategorised df if used from previous prepare
                 configuration_file=None,
                 prepared=False,  # must then use datafile = df_original_data
                 # Limit for missing values in columns
                 limit=0.5):
        self.datafile = datafile
        self.configuration_file = configuration_file
        self.cfg = None
        self.data = None
        self.data_original = None
        self.max_bins = None
        self.min_values = None
        self.ordinal = None
        self.categorical = None
        self.new_categorical_variables = None
        self.scale = None
        self.binary = None
        self.binary_original = None
        self.numerical_float = None
        self.numerical_int = None
        self.binary_optional = None
        self.new_binary_variables = None
        self.new_scale_variables = None
        self.new_categorical_variables = None
        self.income = None
        self.benefits = None
        self.missing_99 = None
        self.missing_9 = None
        self.not_impute = None
        self.hot_columns = None
        self.include_columns = None
        self.analysis = None
        self.housholdId = None
        self.remove_one_hots = None
        self.prepared = prepared
        self.set_configurations()
        self.limit = limit  # Remove columns with more than limit missing values
        self.initiate()

    def set_configurations(self):
        """Read and set values from yaml configuration file

        Parameters
        ----------
        self

        Returns
        -------
        None

        """
        with open(self.configuration_file, 'r') as file:
            self.cfg = yaml.safe_load(file)
        # read configurations
        variables = self.cfg["variables"]
        self.limit = self.cfg["limit"]
        self.ordinal = list(variables["ordinal"])
        self.categorical = list(variables["categorical_nominal"])
        self.scale = list(variables["categorical_ordinal"])
        self.binary_original = list(variables["binary_original"])
        self.numerical_float = list(variables["numerical_float"])
        self.numerical_int = list(variables["numerical_int"])
        self.binary_optional = list(variables["binary_optional"])
        self.new_categorical_variables = list(variables["new_categorical_variables"])
        self.new_binary_variables = list(variables["new_binary_variables"])
        self.new_scale_variables = list(variables["new_scale_variables"])
        self.income = list(variables["income"])
        self.benefits = list(variables["benefits"])
        self.missing_99 = list(variables["missing_99"])
        self.missing_9 = list(variables["missing_9"])
        self.not_impute = list(variables["not_impute"])
        self.analysis = self.cfg["analysis"]
        self.remove_one_hots = list(variables["remove_one_hots"])
        self.binary = []

    def initiate(self):
        """Initiates datasets

        Parameters
        ----------
        self

        Returns
        -------
        None

        """
        self.data = pd.read_csv(self.datafile)
        self.data_original = self.data.copy()
        if self.include_columns is not None:
            self.data = self.data[self.include_columns].copy()
        # Remove columns with all missing values
        self.data_remove_all_nan()
        self.include_columns = list(self.data.columns)
        if self.prepared:
            print("Using provided preprocessed file " + self.datafile)
            self.binary = self.binary_optional + self.new_binary_variables
            self.categorical = self.categorical + self.new_categorical_variables
            self.scale = self.scale + self.new_scale_variables 
            # Create pandas series with max_bins (the max value of each scale variable)
            all_categorical = self.scale + self.categorical
            # Keep only relevant column-data
            self.categorical = [x for x in set(self.categorical) if x in set(self.include_columns)]
            self.scale = [x for x in set(self.scale) if x in set(self.include_columns)]
            self.binary_original = [x for x in set(self.binary_original)
                                    if x in set(self.include_columns)]
            self.numerical_float = [x for x in set(self.numerical_float)
                                    if x in set(self.include_columns)]
            self.numerical_int = [x for x in set(self.numerical_int) if x in set(self.include_columns)]
            to_bins = all_categorical + self.binary
            self.max_bins = self.data[to_bins].max(axis=0)
            self.min_values = self.data[to_bins].min(axis=0)
        else:
            print("Start preparing raw data from scratch! Will take about 3 minutes")
            # Keep only relevant column-data
            self.categorical = [x for x in set(self.categorical) if x in set(self.include_columns)]
            self.scale = [x for x in set(self.scale) if x in set(self.include_columns)]
            self.binary_original = [x for x in set(self.binary_original)
                                    if x in set(self.include_columns)]
            self.numerical_float = [x for x in set(self.numerical_float)
                                    if x in set(self.include_columns)]
            self.numerical_int = [x for x in set(self.numerical_int) if x in set(self.include_columns)]
            # Replace scale values 9 and 99 with NaN
            self.values_to_nan()
            # Create pandas series with max_bins (the max value of each scale variable)
            all_categorical = self.scale + self.categorical
            self.max_bins = self.data[all_categorical].max(axis=0)
            self.min_values = self.data[all_categorical].min(axis=0)
            # Normalise categorical (by dividing max-value --> kept in max_bins)
            self.data_clean_impute()
            self.data_original = self.data.copy()  # Keep data for testing data conversions
            self.calculate_new_variables()
            self.clean_binary_variables()
            self.data[self.categorical] = self.data[self.categorical].astype(int)
            self.data[self.binary] = self.data[self.binary].astype(int)
            if self.analysis == "Categorical":
                self.categorical = list(set(self.categorical + self.scale))
                self.scale = []
        self.one_hot_encode()
        self.sort_columns()  # to make sure the outputs always have same sequence of variables
        self.data = self.data[sorted(self.data.columns)] 
        self.max_bins.sort_index()  # sort pandas Series
        self.min_values.sort_index()

    def sort_columns(self):
        """Sort variables with columns information

        Parameters
        ----------
        self

        Returns
        -------
        None

        """
        if self.categorical is not None:
            self.categorical.sort()
        if self.numerical_float is not None:
            self.numerical_float.sort()
        if self.numerical_int is not None:
            self.numerical_int.sort()
        if self.hot_columns is not None:
            self.hot_columns.sort()
        if self.income is not None:
            self.income.sort()
        if self.benefits is not None:
            self.benefits.sort()
        if self.scale is not None:
            self.scale.sort()
        if self.binary is not None:
            self.binary.sort()
        if self.include_columns is not None:
            self.include_columns.sort()

    def data_clean_impute(self):
        """Normalise and impute original variables

        Parameters
        ----------
        self

        Returns
        -------
        None

        """
        if self.analysis == "Mixed":
            self.normalise_scale()
        self.impute()

    def clean_binary_variables(self):
        """Reconstruct original variables to one-columns binaries

        Parameters
        ----------
        self

        Returns
        -------
        None

        """
        data = self.data.copy()
        # TODO: fix hard coding
        if self.binary:
            self.binary.sort()
        if self.binary_original:
            self.binary_original.sort()
        optionals = ["isFemale", "hasIllness", "hasFriend", "getHelp"]
        assert len(self.binary_original) == len(optionals)
        # isFemale
        data["isFemale"] = np.floor(data["PB150"] / 2)
        # Create pure binaries where 1 = yes (keep only one variable)
        for i, val in enumerate(self.binary_original):
            if i == 0:
                continue
            data[optionals[i]] = 0
            data[optionals[i]].where(self.data[val] > 1.0, 1, inplace=True)
        self.binary = list(set(self.binary + optionals))
        for remove in self.binary_original:
            if remove in self.binary:
                self.binary.remove(remove)
        data[self.binary] = data[self.binary].astype(int)
        data = data.drop(self.binary_original, axis=1)
        for col in self.binary:
            self.max_bins[col] = 1
            self.min_values[col] = 0
        data[self.binary] = data[self.binary].astype(int)
        self.data = data.copy()

    def calculate_new_variables(self):
        """Calculate prepared (new) variables from original variables

        Parameters
        ----------
        self

        Returns
        -------
        None

        """
        data = self.data.copy()
        # TODO: fix hard coding
        # ADD NEW VARIABLES
        # HouseholdID and FamilyPosition from personalID (PB030)
        data["HouseholdID"] = data["PB030"] // 100
        self.housholdId = data["HouseholdID"].copy()
        # Age to 5 categories from year of birth
        data["Age"] = pd.cut(x=data["PB140"],
                             bins=[1931, 1944, 1957, 1970, 1983, 1996],
                             labels=[5, 4, 3, 2, 1])
        data["Age"] = data["Age"].astype(int)
        # hasIncome (to binary)
        data["hasIncome"] = np.where(data[self.income].sum(axis=1) > 0, True, False)
        # hasBenefits (to binary)
        data["hasBenefits"] = np.where(data[self.benefits].sum(axis=1) > 0, True, False)
        self.binary = self.binary + ["hasBenefits", "hasIncome"]
        # householdSize
        data["householdSize"] = 0
        # pandas Series
        households = data["HouseholdID"].value_counts()
        for key, val in households.items():
            data["householdSize"] = np.where(data["HouseholdID"] == int(key),
                                             val, data["householdSize"])
        # Reduce to max 5 categories
        data["householdSize"] = np.where(data["householdSize"] > 4, 5, data["householdSize"])
        # data["PB030"].value_counts()[1404398] ---> 8 (number of members in household
        # CLEAN
        # Add new variables to type of variables list
        self.binary = list(set(self.binary + ["hasIncome", "hasBenefits"]))
        # Update max_bins with new variables
        self.max_bins["Age"] = data["Age"].max(axis=0)
        self.min_values["Age"] = data["Age"].min(axis=0)
        # self.max_bins["FamilyPosition"] = data["FamilyPosition"].max(axis=0)
        self.max_bins["householdSize"] = data["householdSize"].max(axis=0)
        self.min_values["householdSize"] = data["householdSize"].min(axis=0)
        if self.analysis == "Mixed":
            # Normalise late new scale variables:
            data["Age"] = data["Age"] / self.max_bins["Age"]
            data["householdSize"] = data["householdSize"]/data["householdSize"].max(axis=0)
            self.scale = list(set(self.scale + ["Age"] + ["householdSize"]))
            assert not data["householdSize"].isnull().values.any()
        if self.analysis == "Categorical":
            self.categorical = list(set(self.categorical + ["Age", "householdSize"]))
        # Drop transformed variables
        data = data.drop(["PB030", "PB140"], axis=1)
        income_benefits = list(set(self.income + self.benefits))
        data = data.drop(income_benefits, axis=1)
        self.data = data.copy()

    def data_remove_all_nan(self):
        """Removes columns with nan for all examples

        Parameters
        ----------
        data .... pandas dataframe

        Returns
        -------
        df ...... cleaned data as pandas dataframe

        """
        self.data = self.data.dropna(axis=1, how="all").copy()

    def data_remove_limit_nan(self):
        """Removes columns with a limit of nans in examples

        Parameters
        ----------
        data .... pandas dataframe

        Returns
        -------
        df ...... cleaned data as pandas dataframe

        """
        # Find columns with more than limit missing data
        data = self.data.copy()
        total = data.shape[0]
        is_nan = data.isnull().sum()
        is_nan = np.where(is_nan > self.limit * total, 1, 0)
        take_out_columns = np.array(data.columns)[is_nan.astype(bool)]
        data = data.drop(take_out_columns, axis=1)
        self.data = data.copy()

    def impute(self, columns=None):
        """Replace NaN in original data

        Parameters
        ----------
        columns .... optional: impute only variables in "columns"

        Returns
        -------
        None

        """
        data = self.data.copy()
        if columns is None:
            columns = self.categorical + self.scale + self.numerical_float + self.binary_original
        scale_col = [x for x in columns if x in set(self.scale)]  # scale
        nominal_col = [x for x in columns if x in set(self.categorical)]  # nominal
        numerical_col = [x for x in columns if x in set(self.numerical_float)]  # numerical floats
        binary_col = [x for x in columns if x in set(self.binary_original)]  # binaries
        knn_col = list(set(binary_col + nominal_col + scale_col))
        imputer = IterativeImputer(max_iter=20,
                                   skip_complete=True,
                                   random_state=42,
                                   missing_values=np.NaN)
        knn = KNNImputer(n_neighbors=7,
                         missing_values=np.NaN
                         )
        df = data[columns].copy()
        numerical_col = [x for x in numerical_col if x not in self.not_impute]
        knn_col = [x for x in knn_col if x not in self.not_impute]
        temp_data = imputer.fit_transform(df[numerical_col])  # Impute iterative
        # temp_data_knn = knn.fit_transform(df[nominal_col])  # Impute knn
        temp_data_knn = knn.fit_transform(df[knn_col])  # Impute knn
        # Create pandas dataframe of imputed dataset
        df_imputed = pd.DataFrame(temp_data, columns=numerical_col)
        df_imputed_knn = pd.DataFrame(temp_data_knn, columns=knn_col)
        data[numerical_col] = df_imputed.copy()
        # data[nominal_col] = df_imputed_knn.copy()
        data[knn_col] = df_imputed_knn.copy()
        self.data = data

    def one_hot_encode(self, columns=None):
        """Replace values in list "values" with NaN

        Parameters
        ----------
        values .... list of values to replace with NaN

        Returns
        -------
        None

        """
        data = self.data.copy()
        if columns is None:
            columns = self.categorical
        hot = OneHotEncoder(categories="auto",
                            dtype=np.int8)
        temp_one_hot = hot.fit_transform(data[columns])
        self.hot_columns = hot.get_feature_names_out()
        df_one_hot = pd.DataFrame(temp_one_hot.toarray(), columns=self.hot_columns)
        # Reconstruct new data-set
        data = pd.concat([df_one_hot.copy(),
                          data[self.scale].copy()], axis=1)
        data = pd.concat([data, self.data[self.binary]], axis=1)
        self.data = data

    def values_to_nan(self):
        """Replace 99 and 9 as missing values with NaN

        Parameters
        ----------
        self

        Returns
        -------
        None

        """
        self.data[self.missing_99] = self.data[self.missing_99].replace(99, np.nan)
        self.data[self.missing_9] = self.data[self.missing_9].replace(9, np.nan)

    def normalise_scale(self):
        """Normalise scale columns [0, 1]

        Parameters
        ----------
        columns ... list of column-names to include (all selected if None)

        Returns
        -------
        None

        """
        self.data[self.scale] = self.data[self.scale] / self.max_bins[self.scale]

    def normalise_numerical(self):
        """Normalise numerical columns [0, 1] or z-score

        Parameters
        ----------
        self

        Returns
        -------
        None

        """
        scaler = StandardScaler()
        scaler.fit(self.data[self.numerical_float])
        self.data[self.numerical_float] = scaler.transform(self.data[self.numerical_float])

    def de_normalise_scale(self, df, columns=None):
        """De-normalise scale columns
        back from [0, 1] to scale categorical values

        Parameters
        ----------
        df ........ pandas dataframe
        columns ... list of column-names to include (all selected if None)

        Returns
        -------
        df ...... dataframe with (selected) columns with de-normalised values

        """
        all_columns = self.scale
        if columns is None:
            # new_columns = [x for x in set(df.columns) if x in set(all_categorical)]
            new_columns = [x for x in set(all_columns) if x in set(df.columns)]
        else:
            new_columns = [x for x in set(all_columns) if x in set(columns)]
        df = df[new_columns].copy() * self.max_bins[new_columns]
        return df

    def get_argmax_score(self, df):
        """Argmax for a single variables one-hot-encoded options.
        One-hot encoded variables are named NNN_<number>, where the
        number represent the score on the variable.

        Parameters
        ----------
        df ........ pandas dataframe with all options for one variable

        Returns
        -------
        score ...... a string formatted number extracted from the option
                     with the highest sigmoid output
        """
        argmax_name = df.idxmax(axis=1)
        score = argmax_name.str.split("_").str[-1]
        # score is a string with a number
        return score

    def get_data(self, columns=None):
        """Get cleaned data as a pandas dataframe.
        Used as input in deep generative networks.

        Parameters
        ----------
        columns ...... column names as a list

        Returns
        -------
        data ...... dataframe with one-hots [0, 1],
                    normalised scale and
                    fixed binary variables [0, 1]

        """
        if columns is None:
            columns = self.data.columns
        df = self.data[columns].copy()
        return df

    def get_data_recategorised(self):
        df = self.data.copy()  # binaries are by default fixed in self.data
        if self.categorical:
            df = self.re_categorise_hot(df)
        if self.scale:
            df = self.re_categorise_scale(df)
        return df
    
    def get_data_causal(self):
        """Get a representation of the data prepared for causal analysis. One category
        for each one-hot-encoded variable is removed.

        Parameters
        ----------
        
        Returns
        -------
        df_c ...... return self.data with one category for each one-hot-encoded
                    variable removed

        """
        df_causal = self.data.copy()
        df_causal = df_causal.drop(self.remove_one_hots, axis=1)
        return df_causal

    def synthetic_raw_data(self, synthetic, columns=None):
        """From synthetic output to a pandas dataframe representation.

        Parameters
        ----------
        synthetic ........ output from decoder (a tensorflow object)

        Returns
        -------
        df ...... dataframe with synthetic data

        """
        if columns is None:
            print("get_synthetic_raw() no columns")
            columns = self.data.columns
        x = synthetic.shape[0]
        y = synthetic.shape[1]
        df_synthetic = pd.DataFrame(np.array(synthetic).reshape(x, y))
        df_synthetic.columns = columns
        df_synthetic = df_synthetic[columns]
        return df_synthetic.copy()

    def get_synthetic(self, synthetic, columns=None):
        """From synthetic output to a pandas dataframe representation.
        Reconstruct binary variables by single sigmoid and
        threshold 0.5. Reconstruct one-hot-encoded categorical variables
        by drawing argmax from the variables internal categories. Keep
        scaled-variables (= use the sigmoid outputs as variable value).

        Parameters
        ----------
        synthetic ........ output from decoder (a tensorflow object)

        Returns
        -------
        df ...... dataframe with synthetic data on the same form as class data

        """
        if columns is None:
            print("get_synthetic() no columns")
            columns = self.data.columns
        df = self.get_synthetic_recategorised(synthetic, columns)
        # Convert synthetic data on same form as class data
        convert_columns = list(set(list(self.categorical + list(self.new_categorical_variables))))
        df_convert, columns_convert = self.hot_encode(df[convert_columns], convert_columns)
        drop_columns = list(set(list(self.categorical) + list(self.new_categorical_variables)))
        df = df.drop(drop_columns, axis=1)
        df = pd.concat([df, df_convert], axis=1)
        diff_columns = list(set(self.hot_columns) - set(columns_convert))
        if diff_columns:
            print("Do detect {} difference!!!".format(len(diff_columns)))
            array = np.zeros((df.shape[0], len(diff_columns)))
            df_diff = pd.DataFrame(data=array, columns=diff_columns)
            df_diff = df_diff.astype(int)
            print("Old data shape {}\nNew dataframe shape {}".format(df.shape, df_diff.shape))
            df_concat = pd.concat([df, df_diff], axis=1)
            print("Returned dataframe has shape {}".format(df_concat.shape))
            df = df_concat
        return df[columns]

    def get_synthetic_recategorised(self, synthetic, columns=None):
        """From synthetic output to a pandas dataframe representation.
        Recategorise all variables to a number on each variable for
        both nominal and ordinal (scaleable) variables.

        Parameters
        ----------
        synthetic ........ output from decoder (a tensorflow object)
        columns .......... optional: add columns to extract from class data

        Returns
        -------
        df ...... dataframe with synthetic data similar to original data input

        """

        if columns is None:
            print("get_synthetic_recategorised() no columns")
            columns = self.data.columns
        df = self.synthetic_raw_data(synthetic, columns)
        df = df[columns]
        # Convert one-hot-encoded variables in df_synthetic
        if self.categorical:
            df = self.re_categorise_hot(df)
        if self.binary:
            df = self.re_categorise_binary(df)
        if self.scale:
            df = self.re_categorise_scale(df)
        df = df.astype(int)
        return df
    
    def get_synthetic_causal(self, synthetic, columns=None):
        """Get a representation of synthetic data prepared for causal analysis. One category
        for each one-hot-encoded variable is removed.

        Parameters
        ----------
        synthetic .... a tensorflow representation of synthetic data
        Returns
        -------
        df_causal ...... return synthetic data with one category for each one-hot-encoded
                         variable removed

        """
        if columns is None:
            print("get_synthetic_causal() no columns")
            columns = self.data.columns
        df_causal = self.get_synthetic(synthetic, columns)
        use_columns = [x for x in columns if x not in self.remove_one_hots]
        df_causal = df_causal[use_columns].copy()
        return df_causal[use_columns]

    def re_categorise_binary(self, df=None):
        if df is None:
            df = self.data.copy()
        convert_columns = [x for x in self.binary if x in df.columns]
        df[convert_columns] = np.where(df[convert_columns] >= 0.5, 1, 0)
        df[convert_columns] = df[convert_columns].astype(int)
        return df

    def re_categorise_scale(self, df=None):
        if df is None:
            df = self.data.copy()
        convert_columns = [x for x in self.scale if x in df.columns]
        df[convert_columns] = df[convert_columns] * self.max_bins[convert_columns]
        df[convert_columns] = df[convert_columns].astype(int)
        return df

    def re_categorise_hot(self, df=None):
        """Reconstruct one-hot-encoded columns in
        synthetic data by calculating argmax on each variable's options.
        Is also used on original data to convert back from one-hot to
        categorical number.

        Parameters
        ----------
        data ..... reconstructed pandas dataframe from tensorflow object

        Returns
        -------
        df_max ........ reconstruction of all one-hot-encoded to scale
        hot_columns ... names of the one-hot-variables converted

        """
        if df is None:
            df = self.data.copy()
        collect_column_names = {}
        # hot_columns = [x for x in self.hot_columns if x in df.columns]
        hot_columns = [x for x in df.columns if x in self.hot_columns]
        for col in hot_columns:
            if col.isalpha():
                # Should only be the scaled or binary variables
                continue
            else:
                # Should only be the one-hot encoded variables
                name = col.split("_")[0]
                if collect_column_names.get(name) is None:
                    collect_column_names[name] = []
                collect_column_names[name].append(col)
        # Select argmax for each one-hot-encoded variable
        arg_max_all = {}
        for col_ in collect_column_names:
            # Extract argmax for each hot-encoded variable
            arg_max_all[col_] = self.get_argmax_score(df[collect_column_names[col_]]
                                                      ).astype(int)
        # Create a new dataframe with the reconstructed columns
        df_hot = pd.DataFrame.from_dict(arg_max_all)
        df_hot = df_hot.astype(int)
        df = pd.concat([df, df_hot], axis=1)
        df = df.drop(hot_columns, axis=1)
        return df

    def hot_encode(self, x, hot_list, na=False):
        column_list = []
        for h in hot_list:
            df = pd.get_dummies(x[h], prefix=h, dummy_na=na, dtype=np.int8)
            x = pd.concat([x, df], axis=1)
            column_list.append(df.columns.tolist())
        # Flatten the lists within lists of one-hot-column names
        flat_list = [item for sublist in column_list for item in sublist]
        x = x.drop(hot_list, axis=1)
        return x, flat_list
