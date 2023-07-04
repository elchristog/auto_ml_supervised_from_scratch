import pandas as pd
import numpy as np
import awswrangler as wr
import category_encoders as ce
import matplotlib.pyplot as plt
import os
import seaborn as sns
import kds
import shap
import umap
import boto3
from typing import Dict, Any


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cluster import DBSCAN

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import KMeansSMOTE
from sklearn.neural_network import MLPClassifier

from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.calibration import CalibratedClassifierCV


from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from missingpy import MissForest
from sklearn.feature_selection import SelectKBest, f_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime



from statsmodels.compat.python import Literal, lzip
from pandas_profiling import ProfileReport
import warnings
warnings.filterwarnings("ignore")



def read_table(db_name: str, table_name: str) -> pd.DataFrame:
    """
    Description: read_table function reads using Athena the ETL processed master table
    Input:
    - db_name: Name of the data base in Athena
    - table_name: Name of the master table in Athena
    Output:
    - df: readed DataFrame
    """
    df = wr.athena.read_sql_query(sql = "SELECT * FROM " + db_name + "." + table_name,
                                  database= db_name,
                                  s3_output = 's3://itau-my-us-east-1-077002895655-athena/',
                                  ctas_approach = False) 
    return df


def descriptive(df: pd.DataFrame, label: str, output_dir: str = None, columns: list = None) -> None:
    """
    Description: descriptive function to generate a pandas profiling report and a pairplot
    Input: 
        - dataframe: dataframe to be analyzed
        - label: target column
        - output_dir: directory to save the report
        - columns: list of columns to be used in the analysis
    Output: pandas profiling report and pairplot
    """

    if columns:
        df = df[columns]

    profile = ProfileReport(df)
    if output_dir:
        profile.to_file(output_dir + "/profile_report.html")
    else:
        profile.to_file("profile_report.html")
    sns.pairplot(df, hue=label)
    plt.show()

    return profile


def stratified_ad_hoc(df: pd.DataFrame, stratified_variables: np.ndarray, size_percentage: float = 0.3) -> pd.DataFrame:
    """
    Description: stratified_ad_hoc functions generate a stratified sample of the dataframe.
    Input:
        - df : pd.DataFrame 
            Dataframe to be stratified.
        - label_name : str
            Name of the label column.
        - stratified_variables : np.ndarray
            Array of the variables to be stratified.
        - size_percentage : float
            Percentage of the dataframe to be returned.
    Output:
        - df_stratified : pd.DataFrame
            Stratified sample of the dataframe.
    """

    df_stratified, df_test = train_test_split(df, test_size = (1-size_percentage), stratify = df[stratified_variables])
    
    return df_stratified


def cross_validation_sampling(df: pd.DataFrame, label: str, cv: int, scoring: str, month_feature: str, scoring_month: str, validation_month: str):
    """
    Description: cross_validation_sampling function to generate a cross validation sampling
    Input:
    - df: dataframe
    - label: label column name
    - cv: number of folds
    - scoring: scoring metric
    - month_feature: month feature name
    - scoring_month: scoring month
    - validation_month: validation month
    Output:
    - train_df: dataframe with the training data
    - validation_df: dataframe with the validation data
    - scoring_df: dataframe with the scoring data
    - cv: cross validation object
    """

    train_df = df[(df[month_feature] != scoring_month) & (df[month_feature] != validation_month)]
    validation_df = df[df[month_feature] == validation_month]
    scoring_df = df[df[month_feature] == scoring_month]
    cv = KFold(n_splits=cv, shuffle=True, random_state=1)

    return train_df, validation_df, scoring_df, cv


def nulls_per_df(train_df: pd.DataFrame, validation_df: pd.DataFrame, scoring_df: pd.DataFrame, numerical_features: list):
    """
    Description: nulls_per_df function 
    Input:
    - train_df: Dataframe with the training data
    - validation_df: Dataframe with the validation data
    - scoring_df: Dataframe with the scoring data
    - numerical_features: List with the numerical features
    Output: print with the number of nulls per dataframe and plot with the kde distribution per numerical_feature
    """
    
    train_nulls = train_df.isnull().sum()
    validation_nulls = validation_df.isnull().sum()
    scoring_nulls = scoring_df.isnull().sum()
    nulls_df = pd.DataFrame({'train_nulls': train_nulls, 'validation_nulls': validation_nulls, 'scoring_nulls': scoring_nulls})
    
    dtypes_train = train_df.dtypes
    dtypes_validation = validation_df.dtypes
    dtypes_scoring = scoring_df.dtypes
    dtypes_df = pd.DataFrame({'dtypes_train': dtypes_train, 'dtypes_validation': dtypes_validation, 'dtypes_scoring': dtypes_scoring})

    for feature in numerical_features:
        plt.figure(figsize=(10,5))
        sns.kdeplot(train_df[feature], label='train')
        sns.kdeplot(validation_df[feature], label='validation')
        sns.kdeplot(scoring_df[feature], label='scoring')
        plt.title(feature)
        plt.legend()
        plt.show()

    return nulls_df, dtypes_df


def replace_outliers_with_nan(df, num_cols, threshold=3):
    """
    Description: This function replaces the outliers in the numerical columns of a dataframe with NaNs
    Input: 
        - df: Pandas dataframe
        - num_cols: list of numerical columns in the dataframe
        - threshold: the number of standard deviations from the mean beyond which a data point is considered an outlier
    Output:
        - df_no_outliers: The input dataframe with the outliers replaced by NaNs
    """
    
    df_no_outliers = df.copy()
    
    for col in num_cols:
        col_mean, col_std = np.mean(df[col]), np.std(df[col])
        cut_off = col_std * threshold
        lower, upper = col_mean - cut_off, col_mean + cut_off
        outliers = df_no_outliers[(df_no_outliers[col] < lower) | (df_no_outliers[col] > upper)].index
        df_no_outliers.loc[outliers, col] = np.nan
        
    return df_no_outliers



def imputation_comparison(train_df, label_name, numerical_features, cv, miss_forest=False) -> pd.DataFrame:
    """
    Description: This function tests all methods of imputing missing values for numerical features and compares their performance using cross-validation with logistic regression and Lasso regularization.
    Input: 
        - train_df: Pandas dataframe with the data to be imputed and tested
        - label_name: Name of the column with the label
        - numerical_features: List of names of the numerical features to be imputed
        - cv: KFold object with the cross-validation parameters
        - miss_forest: Boolean indicating whether to include the missForest method (default False)
    Output:
        - None
    """
    
    imputer_list = {'mean': SimpleImputer(strategy='mean'),
                    'median': SimpleImputer(strategy='median'),
                    'most_frequent': SimpleImputer(strategy='most_frequent'),
                    'knn': KNNImputer()}
    
    if miss_forest:
        imputer_list['missForest'] = MissForest()
    
    results = []
    names = []
    df = train_df[numerical_features + [label_name]]

    for name, imputer in imputer_list.items():
        pipeline = make_pipeline(imputer, MinMaxScaler(), LogisticRegression(penalty='l1', solver='liblinear'))
        scores = cross_val_score(pipeline, df[numerical_features], df[label_name], cv=cv, scoring='roc_auc')
        results.append(scores)
        names.append(name)
        msg = "%s: %f (%f)" % (name, scores.mean(), scores.std())
        print(msg)

    fig = plt.figure()
    fig.suptitle('Imputation Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()



def apply_imputation(train_df, numerical_features, imputer_name):
    """
    Description: This function applies a selected imputation methodology to the dataframe.
    Input: 
        - train_df: Pandas dataframe with the data to be imputed.
        - imputer_name: Name of the imputation method to be applied.
        - numerical_features: List of names of the numerical features to be imputed.
    Output:
        - df_imputed: Dataframe with the imputed data.
    """
    
    imputer_list = {'mean': SimpleImputer(strategy='mean'),
                    'median': SimpleImputer(strategy='median'),
                    'most_frequent': SimpleImputer(strategy='most_frequent'),
                    'knn': KNNImputer(),
                    'missForest': MissForest()}
    
    imputer = imputer_list[imputer_name]
    
    df_imputed = train_df.copy()
    df_imputed[numerical_features] = imputer.fit_transform(df_imputed[numerical_features])
    
    return df_imputed


def standarization_comparison(train_df :pd.DataFrame, cv, label_name : str, numerical_features : list, standarization_list = {'StandardScaler':StandardScaler(),'MinMaxScaler':MinMaxScaler(),'MaxAbsScaler':MaxAbsScaler(),
                                                                                         'RobustScaler':RobustScaler(),'Normalizer':Normalizer(),'PowerTransformer':PowerTransformer()}, 
                              scoring_metric = 'recall'):
    """
    Description: standarization_comparison functions compare the different standarization techniques and generate the boxplots of the metric in a kflod using a logistic regression with Lasso regularization.
    Input:
        - train_df: Data frame with the data.
        - label_name: Name of the column with the label.
        - numerical_features: List with the names of the numerical features.
        - standarization_list: Dictionary with the standarization techniques to be compared.
        - scoring_metric: Metric to be used in the cross validation.
        - cv_folds: Number of folds in the cross validation.
        - cv_seed: Seed for the cross validation.
    Output:
        - standarization_comparison: Data frame with the results of the cross validation.
    """
    results = []
    names = []
    df = train_df[numerical_features + [label_name]]
    
    for name, standarization in standarization_list.items():
        kfold = cv
        cv_results = cross_val_score(LogisticRegression(penalty='l1', solver='liblinear'), standarization.fit_transform(df.drop(label_name, axis=1)), df[label_name], cv=kfold, scoring=scoring_metric)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    fig = plt.figure()
    fig.suptitle('Standarization Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()



def compare_encoders(train_df: pd.DataFrame(), cv, label_name: str, categorical_features: list, encoders= {'onehot': ce.OneHotEncoder(), 'target': ce.TargetEncoder(), 'binary':ce.BinaryEncoder(), 'ordinal':ce.OrdinalEncoder(),
                                                                   'hashing': ce.HashingEncoder(),
                                                                   'helmert':ce.HelmertEncoder(),'sum': ce.SumEncoder(),'polynomial':ce.PolynomialEncoder(),
                                                                   'backward':ce.BackwardDifferenceEncoder(),'base': ce.BaseNEncoder(),
                                                                   'catboost':ce.CatBoostEncoder(),'james': ce.JamesSteinEncoder(),'mestimator':ce.MEstimateEncoder(),
                                                                   'woe':ce.WOEEncoder(),'count': ce.CountEncoder(),'sum':ce.SumEncoder(),'polynomial':ce.PolynomialEncoder(),
                                                                   'backward':ce.BackwardDifferenceEncoder(),'base': ce.BaseNEncoder(),'leave':ce.LeaveOneOutEncoder(),
                                                                   'catboost':ce.CatBoostEncoder(),
                                                                   'mestimator': ce.MEstimateEncoder(),'woe':ce.WOEEncoder(),'count':ce.CountEncoder()}, scoring_metric='recall'):
    """
    Description: compare_encoders function to compare different encoders and show the metrics
    Input:
        - train_df: pd.DataFrame with the training data
        - cv: KFold object
        - label_name: str with the name of the label column
        - categorical_features: list with the names of the categorical features
        - encoders: dict with the encoders to compare
        - scoring_metric: str with the metric to use
    Output:
        - df: pd.DataFrame with the results
    """
    results = []
    names = []
    avg_results = []    

    df = train_df[categorical_features + [label_name]]
    
    for name, data in encoders.items():
        model = LogisticRegression(penalty='l1',solver='liblinear')
        kfold = cv
        scoring_metric = scoring_metric
        if (name =='target') | (name =='catboost') | (name =='james') | (name =='mestimator') | (name =='woe') | (name =='count') | (name =='leave'):
            cv_results = cross_val_score(model,data.fit_transform(df.drop(label_name,axis=1),df[label_name]),df[label_name],cv=kfold,scoring=scoring_metric)
        else: 
            cv_results = cross_val_score(model, data.fit_transform(df.drop(label_name, axis=1)),df[label_name],cv=kfold,scoring=scoring_metric)
        results.append(cv_results)
        names.append(name)
        avg_results.append(cv_results.mean())
        msg = "%s:%f (%f)" % (name,cv_results.mean(), cv_results.std())
        print(msg)

    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()



def compare_balancing(train_df: pd.DataFrame, cv,  label_name: str, numerical_features : list, categorical_features: list, selected_standarizer: str, selected_encoder: str, balancing= {'random':RandomOverSampler(),'smote':SMOTE(),'adasyn':ADASYN(),'borderline':BorderlineSMOTE(),'svm':SVMSMOTE(),
                                                                     'kmeans':KMeansSMOTE(),
                                                                     'smotetomek':SMOTETomek()}, metric='recall'):
    """
    Description: compare_balancing function to compare different balancing methods and show the metrics
    Input:
        - train_df: pd.DataFrame with the data
        - cv: KFold object
        - label_name: str with the name of the label column
        - numerical_features: list with the names of the numerical features
        - categorical_features: list with the names of the categorical features
        - selected_standarizer: str with the name of the selected standarizer
        - selected_encoder: str with the name of the selected encoder
        - balancing: dict with the balancing methods to compare
        - cv: int with the number of cross validation
        - metric: str with the metric to use
    Output:
        df: pd.DataFrame with the results
    """
    results = []
    names = []
    avg_results = []


    if selected_standarizer == 'MinMaxScaler':
        standarizer = MinMaxScaler()
    elif selected_standarizer == 'StandardScaler':
        standarizer = StandardScaler()
    elif selected_standarizer == 'RobustScaler':
        standarizer = RobustScaler()
    elif selected_standarizer == 'PowerTransformer':
        standarizer = PowerTransformer()
    elif selected_standarizer == 'Normalizer':
        standarizer = Normalizer()
    elif selected_standarizer == 'MaxAbsScaler':
        standarizer = MaxAbsScaler()

    if selected_encoder == 'onehot':
        encoder = ce.OneHotEncoder()
    elif selected_encoder == 'target':
        encoder = ce.TargetEncoder()
    elif selected_encoder == 'binary':
        encoder = ce.BinaryEncoder()
    elif selected_encoder == 'ordinal':
        encoder = ce.OrdinalEncoder()
    elif selected_encoder == 'hashing':
        encoder = ce.HashingEncoder()
    elif selected_encoder == 'helmert':
        encoder = ce.HelmertEncoder()
    elif selected_encoder == 'sum':
        encoder = ce.SumEncoder()
    elif selected_encoder == 'polynomial':
        encoder = ce.PolynomialEncoder()
    elif selected_encoder == 'backward':
        encoder = ce.BackwardDifferenceEncoder()
    elif selected_encoder == 'base':
        encoder = ce.BaseNEncoder()
    elif selected_encoder == 'leave':
        encoder = ce.LeaveOneOutEncoder()
    elif selected_encoder == 'catboost':
        encoder = ce.CatBoostEncoder()
    elif selected_encoder == 'mestimator':
        encoder = ce.MEstimateEncoder()
    elif selected_encoder == 'woe':
        encoder = ce.WOEEncoder()
    elif selected_encoder == 'count':
        encoder = ce.CountEncoder()
    elif selected_encoder == 'james':
        encoder = ce.JamesSteinEncoder()
    elif selected_encoder == 'none':
        encoder = None

    df_numerical = train_df[numerical_features]
    df_categorical = train_df[categorical_features]
    df_label = train_df[label_name].reset_index(drop=True)

    if selected_standarizer is not None:
        df_numerical = pd.DataFrame(standarizer.fit_transform(df_numerical),columns=df_numerical.columns).reset_index(drop=True)
    if selected_encoder is not None:
        if selected_encoder == 'target':
            df_categorical = pd.DataFrame(encoder.fit_transform(df_categorical,df_label)).reset_index(drop=True)
        else:
            df_categorical = pd.DataFrame(encoder.fit_transform(df_categorical)).reset_index(drop=True)
    df = pd.concat([df_numerical,df_categorical,df_label],axis=1)

    for name, data in balancing.items():
        model = LogisticRegression(penalty='l1',solver='liblinear')
        kfold = cv
        scoring_metric = metric
        X, y = data.fit_resample(df.drop(label_name,axis=1),df[label_name])
        cv_results = cross_val_score(model, X, y,cv=kfold, scoring=scoring_metric)
        results.append(cv_results)
        names.append(name)
        avg_results.append(cv_results.mean())
        msg = "%s:%f (%f)" % (name,cv_results.mean(), cv_results.std())
        print(msg)

    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()



def data_preparation(train_df: pd.DataFrame, label_name: str, validation_df: pd.DataFrame, scoring_df: pd.DataFrame, numerical_features : list, categorical_features: list, selected_standarizer: str, selected_encoder: str, selected_balancing: str):
    """
    Description: data_preparation function to prepare the data for the model
    Input:
        - train_df: pd.DataFrame with the data
        - label_name: str with the name of the label column
        - validation_df: pd.DataFrame with the data
        - scoring_df: pd.DataFrame with the data
        - numerical_features: list with the names of the numerical features
        - categorical_features: list with the names of the categorical features
        - selected_standarizer: str with the name of the selected standarizer
        - selected_encoder: str with the name of the selected encoder
        - selected_balancing: str with the name of the selected balancing
    Output:
        X_train: pd.DataFrame with the data
        y_train: pd.DataFrame with the data
        X_validation: pd.DataFrame with the data
        y_validation: pd.DataFrame with the data
        X_scoring: pd.DataFrame with the data
        y_scoring: pd.DataFrame with the data
    """
    if selected_standarizer == 'MinMaxScaler':
        standarizer = MinMaxScaler()
    elif selected_standarizer == 'StandardScaler':
        standarizer = StandardScaler()
    elif selected_standarizer == 'RobustScaler':
        standarizer = RobustScaler()
    elif selected_standarizer == 'PowerTransformer':
        standarizer = PowerTransformer()
    elif selected_standarizer == 'Normalizer':
        standarizer = Normalizer()
    elif selected_standarizer == 'MaxAbsScaler':
        standarizer = MaxAbsScaler()

    if selected_encoder == 'onehot':
        encoder = ce.OneHotEncoder()
    elif selected_encoder == 'target':
        encoder = ce.TargetEncoder()
    elif selected_encoder == 'binary':
        encoder = ce.BinaryEncoder()
    elif selected_encoder == 'ordinal':
        encoder = ce.OrdinalEncoder()
    elif selected_encoder == 'hashing':
        encoder = ce.HashingEncoder()
    elif selected_encoder == 'helmert':
        encoder = ce.HelmertEncoder()
    elif selected_encoder == 'sum':
        encoder = ce.SumEncoder()
    elif selected_encoder == 'polynomial':
        encoder = ce.PolynomialEncoder()
    elif selected_encoder == 'backward':
        encoder = ce.BackwardDifferenceEncoder()
    elif selected_encoder == 'base':
        encoder = ce.BaseNEncoder()
    elif selected_encoder == 'leave':
        encoder = ce.LeaveOneOutEncoder()
    elif selected_encoder == 'catboost':
        encoder = ce.CatBoostEncoder()
    elif selected_encoder == 'mestimator':
        encoder = ce.MEstimateEncoder
    elif selected_encoder == 'woe':
        encoder = ce.WOEEncoder()
    elif selected_encoder == 'count':
        encoder = ce.CountEncoder()
    elif selected_encoder == 'james':
        encoder = ce.JamesSteinEncoder()
    elif selected_encoder == 'none':
        encoder = None

    if selected_balancing == 'random':
        balancing = RandomOverSampler()
    elif selected_balancing == 'smote':
        balancing = SMOTE()
    elif selected_balancing == 'adasyn':
        balancing = ADASYN()
    elif selected_balancing == 'borderline':
        balancing = BorderlineSMOTE()
    elif selected_balancing == 'svm':
        balancing = SVMSMOTE()
    elif selected_balancing == 'kmeans':
        balancing = KMeansSMOTE()
    elif selected_balancing == 'smotetomek':
        balancing = SMOTETomek()

    df_train_numerical = train_df[numerical_features].reset_index(drop=True)
    df_train_categorical = train_df[categorical_features].reset_index(drop=True)
    df_train_label = train_df[label_name].reset_index(drop=True)

    df_validation_numerical = validation_df[numerical_features].reset_index(drop=True)
    df_validation_categorical = validation_df[categorical_features].reset_index(drop=True)
    df_validation_label = validation_df[label_name].reset_index(drop=True)

    df_scoring = scoring_df.drop(numerical_features,axis=1)
    df_scoring = df_scoring.drop(categorical_features,axis=1)
    df_scoring.columns = df_scoring.columns.str.replace('_2fw', '')

    df_scoring_numerical = scoring_df[numerical_features].reset_index(drop=True)
    df_scoring_categorical = scoring_df[categorical_features].reset_index(drop=True)
    df_scoring_label = scoring_df[label_name].reset_index(drop=True)

    if selected_standarizer is not None:
        df_train_numerical = pd.DataFrame(standarizer.fit_transform(df_train_numerical)).reset_index(drop=True)
        df_validation_numerical = pd.DataFrame(standarizer.transform(df_validation_numerical)).reset_index(drop=True)
        df_scoring_numerical = pd.DataFrame(standarizer.transform(df_scoring_numerical)).reset_index(drop=True)
        df_train_numerical.columns = numerical_features
        df_validation_numerical.columns = numerical_features
        df_scoring_numerical.columns = numerical_features

    if selected_encoder is not None:
        df_train_categorical = pd.DataFrame(encoder.fit_transform(df_train_categorical)).reset_index(drop=True)
        df_validation_categorical = pd.DataFrame(encoder.transform(df_validation_categorical)).reset_index(drop=True)
        df_scoring_categorical = pd.DataFrame(encoder.transform(df_scoring_categorical)).reset_index(drop=True)

    X_train_pre_balancing = pd.concat([df_train_numerical, df_train_categorical], axis=1)
    y_train_pre_balancing = df_train_label

    X_validation = pd.concat([df_validation_numerical, df_validation_categorical], axis=1)
    y_validation = df_validation_label

    X_scoring = pd.concat([df_scoring_numerical, df_scoring_categorical], axis=1)

    if selected_balancing is not None:
        X_train, y_train = balancing.fit_resample(X_train_pre_balancing, y_train_pre_balancing)   

    counts_pre = y_train_pre_balancing.value_counts()
    counts_post = y_train.value_counts()
    print('Pre-balancing')
    print(counts_pre)
    print('Post-balancing')
    print(counts_post)


    plt.show()
    


    return X_train, y_train, X_validation, y_validation, X_scoring




def umap_dbscan_outliers(df, num_cols, n_neighbors=15, min_samples=5, eps=0.5):
    """
    Description: This function performs UMAP dimensionality reduction on the input data and detects outliers using DBSCAN clustering.
                 It also returns the input dataframe with the outliers removed.
    Input: 
        - df: Pandas dataframe with the data to be reduced and clustered
        - num_cols: list of numerical columns in the dataframe
        - n_neighbors: The number of nearest neighbors used in the UMAP algorithm
        - min_samples: The minimum number of samples in a neighborhood for a point to be considered a core point in DBSCAN
        - eps: The maximum distance between two samples for them to be considered as in the same neighborhood in DBSCAN
    Output:
        - cluster_labels: Array of cluster labels assigned by DBSCAN (-1 indicates an outlier)
        - reduced_data: Array of reduced data points with two dimensions
        - df_no_outliers: The input dataframe with the outliers removed
    """
    
    data = df[num_cols].values
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, metric='euclidean')
    embedding = reducer.fit_transform(data)
    
    clusterer = DBSCAN(min_samples=min_samples, eps=eps)
    cluster_labels = clusterer.fit_predict(embedding)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=cluster_labels, ax=ax[0])
    ax[0].set_title('UMAP')
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=df['target'], ax=ax[1])
    plt.show()
    
    reduced_data = embedding
    cluster_labels = cluster_labels
    
    df_no_outliers = df[cluster_labels != -1]
    
    return cluster_labels, reduced_data, df_no_outliers




def train_mvp_logistic_lasso_regression_random_search(X_train: pd.DataFrame, y_train: pd.DataFrame, X_validation: pd.DataFrame, y_validation: pd.DataFrame, label_name: str, cv) -> LogisticRegression:
    """
    Train a logistic regression model with lasso regularization using random search
    Args:
        X_train: pd.DataFrame with the data
        y_train: pd.DataFrame with the labels
        X_validation: pd.DataFrame with the data
        y_validation: pd.DataFrame with the labels
        label_name: str with the name of the label column
        cv: KFold object
    Returns:
        LogisticRegression model
        model_metrics: dict with the model metrics
    """
    model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    param_grid = {
        'C': np.logspace(-4, 4, 20),
        'class_weight': [None, 'balanced']
    }
    random_search = RandomizedSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_iter=100, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    print('Best hyperparameters: {}'.format(random_search.best_params_))
    print('Best score: {}'.format(random_search.best_score_))
    print('Best estimator: {}'.format(random_search.best_estimator_))

    model = random_search.best_estimator_
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_validation)
    print('Accuracy: {}'.format(accuracy_score(y_validation, y_pred)))
    print('Precision: {}'.format(precision_score(y_validation, y_pred)))
    print('Recall: {}'.format(recall_score(y_validation, y_pred)))
    print('F1: {}'.format(f1_score(y_validation, y_pred)))
    print('ROC AUC: {}'.format(roc_auc_score(y_validation, y_pred)))
    print('Confusion matrix: {}'.format(confusion_matrix(y_validation, y_pred)))
    print('Overfitting: {}'.format(roc_auc_score(y_train, model.predict(X_train)) - roc_auc_score(y_validation, y_pred)/roc_auc_score(y_validation, y_pred)))

    model_metrics = {
        'accuracy': accuracy_score(y_validation, y_pred),
        'precision': precision_score(y_validation, y_pred),
        'recall': recall_score(y_validation, y_pred),
        'f1': f1_score(y_validation, y_pred),
        'roc_auc': roc_auc_score(y_validation, y_pred),
        'confusion_matrix': confusion_matrix(y_validation, y_pred),
        'overfitting': roc_auc_score(y_train, model.predict(X_train)) - roc_auc_score(y_validation, y_pred)/roc_auc_score(y_validation, y_pred)
    }
    
    return model, model_metrics



def model_training(X_train: pd.DataFrame, y_train: pd.DataFrame, X_validation: pd.DataFrame, y_validation: pd.DataFrame, model_name, label_name: str, scoring_metric: str, cv_df):
    """
    Train a model using cross validation
    Args:
        X_train: pd.DataFrame with the data
        y_train: pd.DataFrame with the labels
        X_validation: pd.DataFrame with the data
        y_validation: pd.DataFrame with the labels
        model_name: model to be trained
        label_name: str with the name of the label column
        scoring_metric: str with the scoring metric to be used
        cv_df: pd.DataFrame with the cross validation results
    Returns:
        model: trained model
        model_metrics: dict with the model metrics
    """

    if model_name == 'XGB':
        model = XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
    elif model_name == 'LR':
        model = LogisticRegression(random_state=42, solver='liblinear')
        param_grid = {
            'C': np.logspace(-4, 4, 20),
            'penalty': ['l1', 'l2'],
            'class_weight': [None, 'balanced']
        }
    elif model_name == 'RF':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'criterion': ['gini', 'entropy'],
            'class_weight': [None, 'balanced']
        }
    elif model_name == 'GBM':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    elif model_name == 'NB':
        model = GaussianNB()
        param_grid = {
            'var_smoothing': np.logspace(0,-9, num=100)
        }
    elif model_name == 'DT':
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'class_weight': [None, 'balanced']
        }
    elif model_name == 'NN':
        model = MLPClassifier(random_state=42)
        param_grid = {
            'hidden_layer_sizes': [(10, 10, 10), (20, 20, 20), (30, 30, 30), (40, 40, 40), (50, 50, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }
    else:
        raise Exception('Model not found')
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validation)
    print('Accuracy: {}'.format(accuracy_score(y_validation, y_pred)))
    print('Precision: {}'.format(precision_score(y_validation, y_pred)))
    print('Recall: {}'.format(recall_score(y_validation, y_pred)))
    print('F1: {}'.format(f1_score(y_validation, y_pred)))
    print('ROC AUC: {}'.format(roc_auc_score(y_validation, y_pred)))
    print('Confusion matrix: {}'.format(confusion_matrix(y_validation, y_pred)))
    print('Overfitting: {}'.format(roc_auc_score(y_train, model.predict(X_train)) - roc_auc_score(y_validation, y_pred)/roc_auc_score(y_validation, y_pred)))

    model_metrics = {
        'accuracy': accuracy_score(y_validation, y_pred),
        'precision': precision_score(y_validation, y_pred),
        'recall': recall_score(y_validation, y_pred),
        'f1': f1_score(y_validation, y_pred),
        'roc_auc': roc_auc_score(y_validation, y_pred),
        'confusion_matrix': confusion_matrix(y_validation, y_pred),
        'overfitting': roc_auc_score(y_train, model.predict(X_train)) - roc_auc_score(y_validation, y_pred)/roc_auc_score(y_validation, y_pred)
    }
    
    return model, model_metrics



def shap_values_importance(model: pd.DataFrame(), X_train: pd.DataFrame(), X_validation: pd.DataFrame(), y_train: pd.DataFrame(), y_validation: pd.DataFrame()) -> None:
    """
    Calculate the shap values for the model
    Args:
        model: model to calculate the shap values
        X_train: pd.DataFrame with the data
        X_validation: pd.DataFrame with the data
        y_train: pd.DataFrame with the labels
        y_validation: pd.DataFrame with the labels
    Returns:
    None
    """
    kmeans = shap.kmeans(X_train, 10)
    explainer = shap.KernelExplainer(model.predict_proba, kmeans)
    shap_values = explainer.shap_values(X_validation)
    shap.summary_plot(shap_values, X_validation, plot_type="bar")
    plt.show()



def mvp_predictions(model, scoring_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions with the model
    Args:
        model: model to make the predictions
        scoring_df: pd.DataFrame with the data
    Returns:
        pd.DataFrame with the predictions
    """
    predictions = model.predict(scoring_df)
    probabilities = model.predict_proba(scoring_df)

    return pd.DataFrame({'prediction': predictions, 'probability': probabilities[:,1]})



def piloto_control(X_scoring: pd.DataFrame, predictions: pd.DataFrame, effect_numerical_features: list):
    """
    Description: piloto_control function 
    Input:
    - X_scoring: DataFrame with the sample for scoring
    - predictions: Dataframe with the vector predicted by the model
    - effect_numerical_features: List with the numerical features to be analyzed
    Output:
    - dataframe of selected rows to piloto and control
    - dataframe of means in piloto and control
    """
        
    X_scoring['predictions'] = predictions.prediction
    X_scoring['probability'] = predictions.probability

    X_scoring = X_scoring[X_scoring['predictions'].notnull()]

    effect_categorical_features = [col for col in X_scoring.columns if col not in effect_numerical_features]

    for feature in effect_numerical_features:
        X_scoring[feature + '_decile'] = pd.qcut(X_scoring[feature], 10, labels=False, duplicates='drop')
    
    X_scoring['decile_probability'] = pd.qcut(X_scoring['probability'], 10, labels=False, duplicates='drop')
    decile_columns = [col for col in X_scoring.columns if 'decile' in col]

    X_scoring = X_scoring.sort_values(by=['decile_probability'] + decile_columns + effect_categorical_features, ascending=False)
    X_scoring.reset_index(inplace=True)
    X_scoring['piloto_control'] = np.where(X_scoring.index % 2 == 0, 'piloto', 'control')

    piloto_control_means = X_scoring.groupby(['piloto_control']).agg({'probability': 'mean'})
    for feature in effect_numerical_features:
        piloto_control_means[feature] = X_scoring.groupby(['piloto_control'])[feature].mean()
    piloto_control_means = piloto_control_means.reset_index()
    piloto_control_means = piloto_control_means.rename(columns={'probability': 'mean_probability'})

    return X_scoring, piloto_control_means



def save_model_metrics(model_name, model_version, model_metrics, model_description, selected_imputer, selected_encoder, selected_balancing, used_variables, model):
    """
    Description: save_model_metrics function 
    Input:
    - model_name: Name of the model
    - model_version: Version of the model
    - model_metrics: Dictionary with the model metrics
    - model_description: Description of the model
    - selected_imputer: Selected imputer for the model
    - selected_encoder: Selected encoder for the model
    - selected_balancing: Selected balancing for the model
    - used_variables: List with the variables used in the model
    - model: Model to save
    Output: excel file with the model metrics
    """
    
    if not os.path.isfile('metricas.xlsx'):
        df = pd.DataFrame(columns=['Modelo', 'Versión', 'Descripción', 'Imputación', 'Codificación', 'Balanceo', 'ROC AUC', 'Precision', 'Recall', 'F1', 'Accuracy', 'Confusion Matrix', 'Overfitting', 'Variables', 'Modelo_ajustado', 'Fecha'])
        df.to_excel('metricas.xlsx', index=False)
    
    df = pd.read_excel('metricas.xlsx')
    
    model_metrics_dict = {'Modelo': model_name, 'Versión': model_version, 'Descripción': model_description, 'Imputación': selected_imputer, 'Codificación': selected_encoder, 'Balanceo': selected_balancing, 'ROC AUC': model_metrics['roc_auc'], 'Precision': model_metrics['precision'], 'Recall': model_metrics['recall'], 'F1': model_metrics['f1'], 'Accuracy': model_metrics['accuracy'], 'Confusion Matrix': model_metrics['confusion_matrix'], 'Overfitting': model_metrics['overfitting'], 'Variables': used_variables, 'Modelo_ajustado': model, 'Fecha': datetime.now()}
    
    df = df.append(model_metrics_dict, ignore_index=True)
    df.to_excel('metricas.xlsx', index=False)
    
    return print('Modelo guardado en el excel')



def confusion_mtx(y_test: pd.DataFrame, y_predict: pd.DataFrame):
    
    """
    Description: confusion_mtx function generates confusion matrix from predictions and true labels
    Input:
    - y_test: Dataframe with the true vector of the test data
    - y_predict: Dataframe with the vector predicted by the model
    Output: confusion matrix plot
    """
    
    cm = confusion_matrix(y_test, y_predict)
    sns.heatmap(cm, annot=True, cbar=None, cmap='Oranges')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class'), plt.xlabel('Predict Class')
    plt.show()




def cum_gain(y_test: pd.DataFrame, y_test_prob: pd.DataFrame):
    """
    Description: model_gain function 
    Input:
    - test_set: DataFrame with the sample for testing
    - probability: Dataframe with the vector of model probabilities
    Output:
    - fig: Cumulative gain plot
    - msg_gain: Message with the gain level for the model
    """
    
    fig = plt.figure(figsize=(8,8))
    kds.metrics.plot_cumulative_gain(y_test.iloc[:,0], y_test_prob)
    plt.legend(('model', 'Wizard', 'Random'))
    plt.show()

    gain_total = kds.metrics.decile_table(y_test.iloc[:,0], y_test_prob, labels=False)
    msg_gain = print('The model gain is ' + str(float(gain_total.loc[1, 'cum_resp_pct']))[:4] + '%')
    
    return msg_gain


def polynomial_features(X_train: pd.DataFrame, X_validation: pd.DataFrame, X_test: pd.DataFrame, degree: int) -> tuple:
    """
    Description: polynomial_features function generates polynomial features for the dataset
    Input:
    - X_train: Dataframe with the features
    - y_train: Dataframe with the target
    - X_validation: Validation dataframe
    - y_validation: Validation target
    - X_test: Test dataframe
    - degree: Degree of the polynomial
    Output:
    - X_train_poly: Dataframe with the polynomial features for the train set
    - X_validation_poly: Dataframe with the polynomial features for the validation set
    - X_test_poly: Dataframe with the polynomial features for the test set
    """
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_validation_poly = poly.transform(X_validation)
    X_test_poly = poly.transform(X_test)

    X_train_poly = pd.DataFrame(X_train_poly, columns=poly.get_feature_names(X_train.columns))
    X_validation_poly = pd.DataFrame(X_validation_poly, columns=poly.get_feature_names(X_validation.columns))
    X_test_poly = pd.DataFrame(X_test_poly, columns=poly.get_feature_names(X_test.columns))
    
    return X_train_poly, X_validation_poly, X_test_poly

def vif_filter(X_train: pd.DataFrame, y_train: pd.DataFrame, X_validation: pd.DataFrame, y_validation: pd.DataFrame, cv, threshold) -> list:
    """
    Description: vif_filter function calculates the variance inflation factor for each feature in the dataset.
    Input:
    - X_train: Dataframe with the features
    - y_train: Dataframe with the target
    - X_validation: Validation dataframe
    - y_validation: Validation target
    - threshold: Threshold for the variance inflation factor
    - cv: Kfold cross validation
    Output:
    - selected_features: List with the selected features
    """
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns
    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    selected_features = vif[vif['VIF'] >= threshold]['Features'].tolist() # Revisar

    plt.figure(figsize=(10, 5))
    plt.plot(vif['Features'], vif['VIF'], 'o')
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.xticks(rotation=90)
    plt.show()

    lasso = LassoCV(cv=cv, random_state=0)
    lasso.fit(X_train, y_train)
    print('Original data: Lasso AUC score in kfold test: ', roc_auc_score(y_train, lasso.predict(X_train)))
    print('Original data: Lasso AUC score in validation: ', roc_auc_score(y_validation, lasso.predict(X_validation)))

    lasso = LassoCV(cv=cv, random_state=0)
    lasso.fit(X_train[selected_features], y_train)
    print('VIF reduced data: Lasso AUC score in kfold test: ', roc_auc_score(y_train, lasso.predict(X_train[selected_features])))
    print('VIF reduced data: Lasso AUC score in validation: ', roc_auc_score(y_validation, lasso.predict(X_validation[selected_features])))
        
    return selected_features


def f_filter(X_train: pd.DataFrame, y_train: pd.DataFrame, X_validation: pd.DataFrame, y_validation: pd.DataFrame, cv, k_features = 10):
    """
    Description: Uses f_filter selectkbest method to select the best features.
    Input:
        - X_train: dataframe with the features.
        - y_train: dataframe with the target.
        - X_validation: validation dataframe.
        - y_validation: validation target.
        - cv: kfold cross validation.
        - k_features: support of the selectkbest method.
    Output:
        - selected_features: list with the selected features.
    """
    f_filter = SelectKBest(score_func=f_classif, k=k_features)
    f_filter.fit(X_train, y_train)
    selected_features = X_train.columns[f_filter.get_support()]

    plt.figure(figsize=(10, 5))
    plt.bar([i for i in range(len(f_filter.scores_))], f_filter.scores_)
    plt.xticks([i for i in range(len(f_filter.scores_))], X_train.columns, rotation=90)
    plt.title('Feature Importance')
    plt.show()

    lasso = LassoCV(cv=cv, random_state=0)
    lasso.fit(X_train, y_train)
    print('Original data: Lasso AUC score in kfold test: ', roc_auc_score(y_train, lasso.predict(X_train)))
    print('Original data: Lasso AUC score in validation: ', roc_auc_score(y_validation, lasso.predict(X_validation)))

    lasso = LassoCV(cv=cv, random_state=0)
    lasso.fit(X_train[selected_features], y_train)
    print('F-Filter reduced data: Lasso AUC score in kfold test: ', roc_auc_score(y_train, lasso.predict(X_train[selected_features])))
    print('F-Filter reduced data: Lasso AUC score in validation: ', roc_auc_score(y_validation, lasso.predict(X_validation[selected_features])))

    return selected_features


def model_comparison(X_train: pd.DataFrame, y_train: pd.DataFrame, cv, scoring_metric = 'recall', model_list = {'XGB': XGBClassifier(), 'LR': LogisticRegression(), 'RF': RandomForestClassifier(), 'GBM': GradientBoostingClassifier(), 'KNN': KNeighborsClassifier(), 'NB': GaussianNB(), 'DT': DecisionTreeClassifier(), 'NN': MLPClassifier()}):
    """
    Description: This function compares the models in the list.
    Input: 
        - X_train: dataframe with the features.
        - y_train: dataframe with the target.
        - cv: Kfold cross validation.
        - scoring_metric: scoring metric.
        - model_list: dictionary of the models. model_list = {'model_name': model_object}
    Output:
        - model_comparison: Dataframe with the results of the comparison.
    """
    results = []
    names = []
    for name, model in model_list.items():
        kfold = cv
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring_metric)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def precision_threshold(model, X_train: pd.DataFrame, y_train: pd.DataFrame, X_validation: pd.DataFrame, y_validation: pd.DataFrame):
    """
    Description: This function calculates the precision threshold for the model, using the deciles of the predicted probabilities and looking for the decile with the highest precision.
    Input:
        - model: model object.
        - X_train: dataframe with the features.
        - y_train: dataframe with the target.
        - X_validation: validation dataframe.
        - y_validation: validation target.
        - cv: kfold cross validation.
    """
    y_train_pred_prob = model.predict_proba(X_train)[:, 1]
    y_validation_pred_prob = model.predict_proba(X_validation)[:, 1]

    df = pd.DataFrame({'y_true': y_train.values.flatten(), 'y_pred_prob': y_train_pred_prob})

    df['decile'] = pd.qcut(df['y_pred_prob'], q=10)
    
    df['value'] = df['decile'].apply(lambda x: x.right)

    total_0 = (df['y_true'] == 0).sum()
    total_1 = (df['y_true'] == 1).sum()
    perc_0 = total_0 / len(df)
    perc_1 = total_1 / len(df)

    df_grouped = df.groupby('decile').agg({'y_true': ['sum', 'count', 'mean', lambda x: x.eq(0).sum(), lambda x: x.eq(0).mean()], 'value': 'first'})
    
    df_grouped.columns = ['TP', 'Total', 'Precision', 'FP', 'FPR', 'Value']
    
    df_grouped['Count_0'] = df_grouped['FP']
    df_grouped['Count_1'] = df_grouped['TP']
    df_grouped['Perc_0'] = df_grouped['Count_0'] / total_0
    df_grouped['Perc_1'] = df_grouped['Count_1'] / total_1

    # print("Train data:")
    # print(df_grouped)

    dv = pd.DataFrame({'y_true': y_validation.values.flatten(), 'y_pred_prob': y_validation_pred_prob})

    dv['decile'] = pd.qcut(dv['y_pred_prob'], q=10)
    
    dv['value'] = dv['decile'].apply(lambda x: x.right)

    total_v0 = (dv['y_true'] == 0).sum()
    total_v1 = (dv['y_true'] == 1).sum()
    perc_v0 = total_v0 / len(dv)
    perc_v1 = total_v1 / len(dv)

    dv_grouped = dv.groupby('decile').agg({'y_true': ['sum', 'count', 'mean', lambda x: x.eq(0).sum(), lambda x: x.eq(0).mean()], 'value': 'first'})
    
    dv_grouped.columns = ['TP', 'Total', 'Precision', 'FP', 'FPR', 'Value']

    dv_grouped['Count_0'] = dv_grouped['FP']
    dv_grouped['Count_1'] = dv_grouped['TP']
    dv_grouped['Perc_0'] = dv_grouped['Count_0'] / total_v0
    dv_grouped['Perc_1'] = dv_grouped['Count_1'] / total_v1

    print("Validation data:")
    print(dv_grouped)

    
    
def model_predictions(model, scoring_df: pd.DataFrame, low_threshold: float, high_threshold: float) -> pd.DataFrame:
    """
    Description: This function predicts the target for the scoring dataframe.
    Input:
        - model: model object.
        - scoring_df: dataframe with the features.
        - low_threshold: low threshold for the target.
        - high_threshold: high threshold for the target.
    Output:
        - Dataframe with the predictions.
    """
    scoring_df['pred_prob'] = model.predict_proba(scoring_df)[:, 1]
    scoring_df['pred'] = np.where(scoring_df['pred_prob'] > high_threshold, 1, np.where(scoring_df['pred_prob'] < low_threshold, 0, np.nan))

    return  pd.DataFrame({'prediction': scoring_df['pred'], 'probability': scoring_df['pred_prob']})


def clean_data(X):
    """
    Description: This function cleans the input data by replacing NaN, infinite values and values too large for float64 with the mean.
    Input:
        - X: dataframe to clean
    Output:
        - X_clean: cleaned dataframe
    """
    X_clean = X.copy()
    X_clean[np.isinf(X_clean)] = np.nan
    X_clean = X_clean.apply(lambda x: x.fillna(x.mean()), axis=0)
    return X_clean


def calibrated_probabilities(model, X_train, y_train, X_validation, X_scoring, cv):
    """
    Description: This function applies probability calibration using CalibratedClassifierCV from sklearn.
    Input:
        - model: trained model object
        - X_train: dataframe with the train features
        - y_train: dataframe with the train target
        - X_validation: dataframe with the validation features
        - X_scoring: dataframe with the scoring features
        - cv: k-fold cross validation
    Output:
        - calibrated_probs: dictionary containing the calibrated probabilities for train, validation, and scoring sets
    """
    X_validation = X_validation[X_train.columns]
    X_scoring = X_scoring[X_train.columns]

    X_train_clean = clean_data(X_train)
    X_validation_clean = clean_data(X_validation)
    X_scoring_clean = clean_data(X_scoring)

    calibrated_model = CalibratedClassifierCV(model, cv=cv, method='sigmoid')
    calibrated_model.fit(X_train_clean, y_train)

    train_calibrated_probs = calibrated_model.predict_proba(X_train_clean)[:, 1]
    validation_calibrated_probs = calibrated_model.predict_proba(X_validation_clean)[:, 1]
    scoring_calibrated_probs = calibrated_model.predict_proba(X_scoring_clean)[:, 1]

    calibrated_probs = {
        'train': train_calibrated_probs,
        'validation': validation_calibrated_probs,
        'scoring': scoring_calibrated_probs
    }

    return calibrated_probs



def create_lambda_function(pipeline_name: str) -> Dict[str, Any]:
    """
    Description: This function creates an AWS Lambda function that triggers the execution of the given pipeline
                 periodically, every month.
    Input:
        - pipeline_name: The name of the registered pipeline that you want to execute periodically.
    Output:
        - lambda_function_info: A dictionary containing information about the created AWS Lambda function.
    """

    lambda_client = boto3.client('lambda')

    lambda_code = f"""
        import boto3

        def lambda_handler(event, context):
            client = boto3.client('sagemaker')
            response = client.start_pipeline_execution(PipelineName='{pipeline_name}')
            return response
        """

    iam_role = 'arn:aws:iam::<account_id>:role/<role_name>'

    lambda_function_name = f"{pipeline_name}_trigger"

    response = lambda_client.create_function(
        FunctionName=lambda_function_name,
        Runtime='python3.8',
        Role=iam_role,
        Handler='lambda_function.lambda_handler',
        Code={
            'ZipFile': lambda_code.encode('utf-8')
        },
        Description=f"Triggers the '{pipeline_name}' pipeline execution periodically every month.",
        Timeout=15,
        MemorySize=128
    )

    lambda_function_info = {
        'FunctionName': response['FunctionName'],
        'FunctionArn': response['FunctionArn'],
        'Version': response['Version']
    }

    eventbridge_client = boto3.client('events')

    rule_name = f"{pipeline_name}_monthly_trigger"
    rule_schedule = 'cron(0 12 1 * ? *)'  # Execute at 12:00 PM UTC on the 1st of every month

    eventbridge_client.put_rule(
        Name=rule_name,
        ScheduleExpression=rule_schedule,
        State='ENABLED',
        Description=f"Triggers the '{pipeline_name}' pipeline execution every month.",
        Targets=[
            {
                'Id': '1',
                'Arn': lambda_function_info['FunctionArn']
            }
        ]
    )

    return lambda_function_info




def create_batch_transform_endpoint(model_name: str, input_data: str, output_data: str) -> Dict[str, Any]:
    """
    Description: This function creates a Batch Transform endpoint using the trained and registered model from the
                 given pipeline.
    Input:
        - model_name: The name of the trained and registered model from the pipeline.
        - input_data: The S3 location of the input data to be used for the Batch Transform job.
        - output_data: The S3 location where the output data should be stored after the Batch Transform job.
    Output:
        - transform_job_info: A dictionary containing information about the Batch Transform job.
    """

    sagemaker_client = boto3.client('sagemaker')

    transform_job_name = f"{model_name}_batch_transform"

    response = sagemaker_client.create_transform_job(
        TransformJobName=transform_job_name,
        ModelName=model_name,
        BatchStrategy='SingleRecord',
        TransformInput={
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': input_data
                }
            },
            'ContentType': 'text/csv',
            'CompressionType': 'None',
            'SplitType': 'Line'
        },
        TransformOutput={
            'S3OutputPath': output_data,
            'Accept': 'text/csv',
            'AssembleWith': 'Line'
        },
        TransformResources={
            'InstanceType': 'ml.m5.xlarge',
            'InstanceCount': 1
        }
    )

    transform_job_info = {
        'TransformJobName': response['TransformJobName'],
        'TransformJobArn': response['TransformJobArn']
    }

    return transform_job_info
