from typing import Optional
from dask import dataframe as dd
import pandas as pd
import numpy as np

def engineering_toolkit(df:pd.DataFrame, features:list, zone_lookup:Optional[pd.DataFrame] = None):
    """
    Function for engineering the new features. Data frame passed as df is expected to be the output 
    of preprocess_data function.
    List of possible variables to construct: 'borough_size', 'speed', 'trip_type', 'season'.
    Arguments passed to features should be the subset of above list but the function is fault-proof in this matter.
    Zone lookup table must fit the certain shape and format. See the link below for exemplary lookup table:
    https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv    
    """
    valid_features = ['borough_size', 'speed', 'trip_type', 'season']
    assert  set(features).issubset(valid_features), 'Features chosen: ' +  ', '.join(valid_features) + ', are not in a list of valid features:' + ', .join(valid_features)}!'
    if 'borough_size' in features:
        assert isinstance(zone_lookup, pd.DataFrame), f'Please specify zone lookup table, it can be downloaded from: {"https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv"}'
        borough_ =  zone_lookup[['Borough', 'Zone']].groupby('Borough').count()
        borough_sizes = pd.Series(borough_.values[:, 0], index = borough_.index).to_dict()
        df[['PUSize', 'DOSize']] = df.apply(lambda row: pd.Series([borough_sizes[row['PUBorough']], borough_sizes[row['DOBorough']]]), 1)
    if 'speed' in features:
        df.loc[:, 'speed'] = df.apply(lambda row: row['trip_distance']/row['drivetime'], 1)
    if 'trip_type' in features:
        df.loc[:, 'trip_type'] = df['extra'].map(lambda fee: {0 : 'day', 0.5 : 'night', 1 : 'rush_hour'}[fee])
    if 'season' in features:
        def seasoner(month):
            if month in (3, 4, 5):
                return 'spring'
            if month in (6, 7, 8):
                return 'summer'
            if month in (9, 10, 11):
                return 'autumn'
            if month in (12, 1, 2):
                return 'winter'
            
        df.loc[:, 'season'] = df['tpep_dropoff_datetime'].map(lambda date: seasoner(pd.to_datetime(date).month))
    df = df.drop(['extra', 'tpep_dropoff_datetime', 'tpep_pickup_datetime'], 1)
    return df


def preprocess_data(df:dd, zone_lookup:pd.DataFrame, columns_to_keep:Optional[list] = None, cat_columns:Optional[list] = None):
        """
        Processes the data to meet the requirements stated at the end of domain understanding notebook. 
        This function cleans the data from outlying observations and anomalies. At this point all type 
        conversion narrows down to specifying the character string. 
        Zone lookup table must fit the certain shape and format. See the link below for exemplary lookup table:
        https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv
        Here the drivetime variable is engineered as it is used in cleaning the data.
        """
        
        if columns_to_keep is not None:
            df = df[columns_to_keep]
        
        if cat_columns is not None:
            type_dict = pd.Series([str for x in cat_columns], index = cat_columns).to_dict()
            df = df.astype(type_dict) 
        
        df['keep'] = np.where(df['extra'].map(lambda x: x in (0, 0.5, 1)), 1, np.nan)
        df = df[df['keep'] == 1]
        
        non_yellow_zones = tuple(zone_lookup.loc[zone_lookup['service_zone'] != 'Yellow Zone', 'LocationID'].values)
        df.loc[:, 'keep'] = np.where(df['PULocationID'].map(lambda x: x not in non_yellow_zones + (0, 263, 264, 265)), 1, np.nan)
        df = df[df['keep'] == 1]
        
        df['keep'] = np.where(df['DOLocationID'].map(lambda x: x not in non_yellow_zones + (0, 263, 264, 265)), 1, np.nan)
        df = df[df['keep'] == 1]
        
        df['keep'] = np.where(df['fare_amount'] > 0, 1, np.nan)
        df['keep'] = np.where(df['tip_amount'] > 0, 1, df['keep'])
        df = df[df['keep'] == 1]
        
        def calculate_drivetime(row):
            return (pd.to_datetime(row['tpep_dropoff_datetime']) - pd.to_datetime(row['tpep_pickup_datetime'])).seconds/60
        df.loc[:, 'drivetime'] = df.apply(calculate_drivetime, 1)
        df['keep'] = np.where(df['drivetime'] < 100, 1, np.nan)
        df = df[df['keep'] == 1]
        
        df['keep'] = np.where(df['drivetime'] > 1, 1, np.nan)
        df = df[df['keep'] == 1]
        df = df.drop('keep', 1)
        zone_dict = pd.Series(zone_lookup['Borough'], index = zone_lookup['LocationID']).to_dict()
        df[['PUBorough', 'DOBorough']] = df.apply(lambda row: pd.Series([zone_dict[row['PULocationID']], zone_dict[row['DOLocationID']]]), 1)
        return df


def ingest_data(year:list, month:Optional[str] = None):
    """
    Reads the data straight from the bucket into the dask dataframe object. Month variable is expected to be of '01, 02,..,11, 12' format.
    When reading the data from specific month, the year must be of string format, while for ingesting yearly bulks of data it can be a list.
    Exemplary use: 
                1. ingest_data('2018') for retrieving the data from whole year 2018,
                2. ingest_data('2019','06') retrieve the data from July 2018,
                3. ingest_data(['2018', '2019']) retrieve full data related to the task.
    """
    valid_months = ('0' + str(i+1) if i < 9 else str(i+1) for i in range(12)) 
    if month is not None:
        assert month in (valid_months), f'Month variable: {str(month)} is of the wrong type or format!'
        assert year in ('2018', '2019'), f'Year variable: {str(year)} is of the wrong type, format or out of range for this task!'
        return dd.read_csv(f'https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_{year}-{month}.csv')
    else:
        urls = [f'https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_{y}-{m}.csv' for y, m in zip(year, valid_months)]
        return dd.read_csv(urls)


