import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def train_test():
    # Define the folder paths
    dataset_folder = 'datasets'
    click_bait_1_folder = 'click_bait_1'
    click_bait_2_folder = 'click_bait_2'

    # Create empty lists to store datasets
    click_bait_1_data = []
    click_bait_2_data = []

    # Load datasets from click_bait_1 folder
    click_bait_1_path = os.path.join(dataset_folder, click_bait_1_folder)
    click_bait_1_files = os.listdir(click_bait_1_path)

    for file in click_bait_1_files:
        if file.endswith('.csv'):
            file_path = os.path.join(click_bait_1_path, file)
            df = pd.read_csv(file_path)
            click_bait_1_data.append(df)

    # Load datasets from click_bait_2 folder
    click_bait_2_path = os.path.join(dataset_folder, click_bait_2_folder)
    click_bait_2_files = os.listdir(click_bait_2_path)

    for file in click_bait_2_files:
        if file.endswith('.csv'):
            file_path = os.path.join(click_bait_2_path, file)
            df = pd.read_csv(file_path)
            click_bait_2_data.append(df)

    # Concatenate click_bait_1 datasets
    click_bait_1_dataset = pd.concat(click_bait_1_data)

    # Concatenate click_bait_2 datasets
    click_bait_2_dataset = pd.concat(click_bait_2_data)

    # Perform train-test splits for click_bait_1 dataset
    click_bait_1_train, click_bait_1_test = train_test_split(click_bait_1_dataset, test_size=0.2, random_state=42)

    # Perform train-test splits for click_bait_2 dataset
    click_bait_2_train, click_bait_2_test = train_test_split(click_bait_2_dataset, test_size=0.2, random_state=42)

    # Extract X_train and Y_train from click_bait_1 train set
    click_bait_1_X_train = click_bait_1_train['headline'] 
    click_bait_1_Y_train = click_bait_1_train['clickbait'] 

    # Extract X_train and Y_train from click_bait_2 train set
    click_bait_2_X_train = click_bait_2_train['headline'] 
    click_bait_2_Y_train = click_bait_2_train['clickbait']  

    # Extract X_test and Y_test from click_bait_1 test set
    click_bait_1_X_test = click_bait_1_test['headline']
    click_bait_1_Y_test = click_bait_1_test['clickbait']

    # Extract X_test and Y_test from click_bait_2 test set
    click_bait_2_X_test = click_bait_2_test['headline']
    click_bait_2_Y_test = click_bait_2_test['clickbait']


    # Concatenate X_train and Y_train from click_bait_1 and click_bait_2 train sets
    X_train = np.concatenate((click_bait_1_X_train, click_bait_2_X_train), axis=0)
    Y_train = np.concatenate((click_bait_1_Y_train, click_bait_2_Y_train), axis=0)

    # Concatenate X_test and Y_test from click_bait_1 and click_bait_2 test sets
    X_test = np.concatenate((click_bait_1_X_test, click_bait_2_X_test), axis=0)
    Y_test = np.concatenate((click_bait_1_Y_test, click_bait_2_Y_test), axis=0)
    #removing Nan Values
    X_train = [x for x in X_train if str(x) != 'nan']
    Y_train = Y_train[~np.isnan(Y_train)]

    X_test = [x for x in X_test if str(x) != 'nan']
    Y_test = Y_test[~np.isnan(Y_test)]


    return(X_train, Y_train, X_test, Y_test)


train_test()