import itertools
import warnings 
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import sympy
from scipy import stats
from collections import OrderedDict

def compute_missing_percentage(dataframe):
    total_rows = len(dataframe)
    missing_percentage = {}
    
    for column in dataframe.columns:
        missing_count = dataframe[column].isnull().sum()
        missing_percentage[column] = (missing_count / total_rows) * 100
    
    return missing_percentage

def delete_features_above_threshold(dataframe, threshold):
    missing_percentages = compute_missing_percentage(dataframe)
    features_to_delete = [column for column, percentage in missing_percentages.items() if percentage > threshold]
    dataframe.drop(features_to_delete, axis=1, inplace=True)

def fill_missing_values(dataframe):
    for column in dataframe.columns:
        if dataframe[column].dtype == 'object':
            mode_value = dataframe[column].mode().iloc[0]
            dataframe[column].fillna(mode_value, inplace=True)
        else:
            median_value = dataframe[column].median()
            dataframe[column].fillna(median_value, inplace=True)

def replace_outliers_with_whiskers(dataframe, multiplier=1.5):
    for column in dataframe.select_dtypes(include=[np.number]).columns:
        unique_values = dataframe[column].unique()
        
        # Skip outlier replacement if feature contains only 0s and 1s
        if len(unique_values) == 2 and set(unique_values) == {0, 1}:
            continue

        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        lower_whisker = q1 - multiplier * iqr
        upper_whisker = q3 + multiplier * iqr

        dataframe[column] = np.where(
            (dataframe[column] < lower_whisker) | (dataframe[column] > upper_whisker),
            np.where(dataframe[column] < lower_whisker, lower_whisker, upper_whisker),
            dataframe[column]
        )


def woe_transform_dataframe(df, target_col, event=1, bins=10, min_bin_size=0.05):
    woe_dict = {}
    
    for col in df.columns:
        if df[col].dtype == 'object' and col != target_col:
            event_total = df[target_col].sum()
            non_event_total = df[target_col].count() - event_total
            x_labels = sorted(df[col].unique())
            woe_col_dict = {}
            
            for label in x_labels:
                event_count = df.loc[df[col] == label, target_col].sum()
                non_event_count = df.loc[df[col] == label, target_col].count() - event_count
                
                # Check if event or non-event count is zero
                if event_count == 0 or non_event_count == 0:
                    # Calculate the average WoE for the feature
                    avg_woe = np.mean(list(woe_col_dict.values()))
                    woe_col_dict[label] = avg_woe
                else:
                    woe = np.log((event_count / event_total) / (non_event_count / non_event_total))
                    woe_col_dict[label] = woe
            
            woe_dict[col] = woe_col_dict
        
        elif pd.api.types.is_numeric_dtype(df[col]) and col != target_col and col != 'timestamp':
            try:
                x_bins = pd.cut(df[col], bins=bins, duplicates='drop')
            except ValueError:
                x_bins = pd.cut(df[col], duplicates='drop')
            
            bin_counts = x_bins.value_counts(normalize=True)
            valid_bins = bin_counts[bin_counts >= min_bin_size].index
            event_total = df[target_col].sum()
            non_event_total = df[target_col].count() - event_total
            woe_col_dict = {}
            
            for bin_label in valid_bins:
                event_count = df.loc[x_bins == bin_label, target_col].sum()
                non_event_count = df.loc[x_bins == bin_label, target_col].count() - event_count
                
                # Check if event or non-event count is zero
                if event_count == 0 or non_event_count == 0:
                    # Calculate the average WoE for the feature
                    avg_woe = np.mean(list(woe_col_dict.values()))
                    woe_col_dict[bin_label] = avg_woe
                else:
                    woe = np.log((event_count / event_total) / (non_event_count / non_event_total))
                    woe_col_dict[bin_label] = woe
            
            woe_dict[col] = woe_col_dict
    
    return woe_dict



def transform_to_woe(df, woe_dict, replace_outliers=True, outlier_threshold=3):
    transformed_df = df.copy()
    
    for col, woe_col_dict in woe_dict.items():
        transformed_df[col] = transformed_df[col].map(woe_col_dict)
        
        if replace_outliers:
            unique_values = transformed_df[col].unique()
            
            # Skip outlier replacement if feature contains only 0s and 1s
            if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                continue

            q1 = transformed_df[col].quantile(0.25)
            q3 = transformed_df[col].quantile(0.75)
            iqr = q3 - q1
            lower_whisker = q1 - outlier_threshold * iqr
            upper_whisker = q3 + outlier_threshold * iqr

            transformed_df[col] = np.where(
                (transformed_df[col] < lower_whisker) | (transformed_df[col] > upper_whisker),
                np.where(transformed_df[col] < lower_whisker, lower_whisker, upper_whisker),
                transformed_df[col]
            )
    
    return transformed_df

def calculate_psi(expected, observed):
    epsilon = 1e-10  # Small value to avoid division by zero
    expected = np.asarray(expected) + epsilon
    observed = np.asarray(observed) + epsilon

    # Calculate the expected and observed event rates
    expected_rate = expected / np.sum(expected)
    observed_rate = observed / np.sum(observed)

    # Calculate the PSI for each category
    psi = (observed_rate - expected_rate) * np.log(observed_rate / expected_rate)
    psi = np.sum(psi)

    return psi

def calculate_psi_dataframe(dataframe, reference_df):
    psi_results = pd.DataFrame(columns=['Feature', 'PSI'])

    for feature in dataframe.columns:
        expected_freq = reference_df[feature].values
        observed_freq = dataframe[feature].values

        psi = calculate_psi(expected_freq, observed_freq)
        psi_results = psi_results.append({'Feature': feature, 'PSI': psi}, ignore_index=True)

    return psi_results

