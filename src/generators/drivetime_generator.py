import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

import os
from pathlib import Path
cwd = Path.cwd()
main_dir = cwd.parents[0]
os.chdir(main_dir)
from toolkit.etl_toolkit import preprocess_data, engineering_toolkit


class drivetime_data_generator():
    """
    Custom class for generating batches of data.
    """
    def __init__(self, 
                 columns_to_keep:list,
                 cat_columns:list,
                 fetching_table:dd,
                 zone_lookup:pd.DataFrame,
                 batch_size:int,
                 shuffle:bool = True
                ):
        """
        Instance initialization
        """
        self.columns_to_keep = columns_to_keep
        self.cat_columns = cat_columns
        self.fetching_table = fetching_table.reset_index().drop('index', 1)
        self.zone_lookup = zone_lookup
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.__on_epoch_end__()
        
    def __len__(self):
        """
        Used to define number of batches to loop through.
        """
        return(int(np.ceil(len(self.fetching_table)/self.batch_size)))
    
    def __getitem__(self, index):
        """
        Safety measure in case of batch_size and indexes length non-divisibility.
        """
        if len(self.indexes[index*self.batch_size:]) > self.batch_size:
            indexes = self.indexes[index*self.batch_size:(index + 1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:]
        "Slicing based on indexes and handling the preprocessing"
        df = self.fetching_table.loc[indexes, :]
        df = preprocess_data(df, self.zone_lookup, self.columns_to_keep, self.cat_columns)
        df = engineering_toolkit(df, ['borough_size', 'trip_type', 'season'], self.zone_lookup)
        df = df[['passenger_count', 'trip_distance', 'PUSize', 'DOSize', 'trip_type', 'PUBorough', 'DOBorough', 'season', 'drivetime']]
        return(
            #The batch of data in a format (X_train, y_train), (X_test, y_test) is returned
            batch_generator(df, bulk_model = True)
            )
    
    def __on_epoch_end__(self):
        self.indexes = list(range(len(self.fetching_table)))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def batch_generator(df:pd.DataFrame, bulk_model: bool = False):
    """
    Generates a batch of data, input should be a computed dask dataframe object, which means pandas DataFrame.
    For regular batch the input data should contain columns:
        'passenger_count', 'trip_distance', 'PUSize', 'DOSize', 'trip_type', 'drivetime'
    
    For bulk_model training the columns in the input should be of this structure:
            'passenger_count', 'trip_distance', 'PUSize', 'DOSize', 'trip_type', 'PUBorough', 'DOBorough', 'season', 'drivetime'
    """
    X_data, y_data = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
    sc = MinMaxScaler()

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size = 0.9)
    X_train[:, :4] = sc.fit_transform(X_train[:, :4])
    X_test[:, :4] = sc.fit_transform(X_test[:, :4])

    y_train = sc.fit_transform(y_train.reshape(-1, 1))
    y_test = sc.fit_transform(y_test.reshape(-1, 1))
    
    if bulk_model:
        trip_enc = OneHotEncoder(categories = (['day', 'rush_hour', 'night'], )).fit(X_train[:, -4].reshape(-1, 1))
        seasons = ['spring', 'summer', 'autumn', 'winter']
        season_enc = OneHotEncoder(categories = (seasons, )).fit(X_train[:, -1].reshape(-1, 1))
        boroughs = ['Bronx', 'Brooklyn', 'EWR', 'Manhattan', 'Queens', 'Staten Island']
        borough_encoder = OneHotEncoder(categories = (boroughs, )).fit(X_train[:, -3:-1].reshape(-1, 1))
        X_train = np.c_[X_train[:, :-4], trip_enc.transform(X_train[:, -4].reshape(-1, 1)).toarray(), X_train[:, -3:]]
        X_test = np.c_[X_test[:, :-4], trip_enc.transform(X_test[:, -4].reshape(-1, 1)).toarray(), X_test[:, -3:]]
        X_train = np.c_[X_train[:, :-1], season_enc.transform(X_train[:, -1].reshape(-1, 1)).toarray()]
        X_test = np.c_[X_test[:, :-1], season_enc.transform(X_test[:, -1].reshape(-1, 1)).toarray()]
        X_train = np.c_[X_train[:, :-6], borough_encoder.transform(X_train[:, -6].reshape(-1, 1)).toarray(), X_train[:, -5:]]
        X_test = np.c_[X_test[:, :-6], borough_encoder.transform(X_test[:, -6].reshape(-1, 1)).toarray(), X_test[:, -5:]]
        X_train = np.c_[X_train[:, :-5], borough_encoder.transform(X_train[:, -5].reshape(-1, 1)).toarray(), X_train[:, -4:]]
        X_test = np.c_[X_test[:, :-5], borough_encoder.transform(X_test[:, -5].reshape(-1, 1)).toarray(), X_test[:, -4:]]
    else: 
        drop_enc = OneHotEncoder(categories = (['day', 'rush_hour', 'night'], )).fit(X_train[:, -1].reshape(-1, 1))
        X_train = np.c_[X_train[:, :-1], drop_enc.transform(X_train[:, -1].reshape(-1, 1)).toarray()]
        X_test = np.c_[X_test[:, :-1], drop_enc.transform(X_test[:, -1].reshape(-1, 1)).toarray()]        
    return (X_train, y_train), (X_test, y_test)