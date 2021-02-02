from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def evaluate_xgb(results:dict, zoom:Optional[tuple] =  None, zoom_on:Optional[str] = None):
    """
    Plotting the metrics specified in the XGBoost model definition. For now it is hardcoded RMSE for it is the sole metric used in modelling.
    Zoom allows to inspect only a chosen part of the learning process. The zoom is expected to be tuple of form: (start_epoch, finish_epoch).
    zoom_on is meant to specify whether the plot should be zoomed on training or validation losses (i.e train or test).
    """
    assert isinstance(results, dict), 'Results provided are of inproper format! Keep on mind that the results from generate_drivetime_model should be used.'
    train_loss = results['train']['rmse']
    test_loss = results['eval']['rmse']    
    epochs = range(len(results['eval']['rmse']))
    if zoom is not None:
        assert isinstance(zoom_on, str), 'Please specify if the train or test losses should be zoomed on.'
        epochs = range(zoom[0], zoom[1])
        train_loss = results['train']['rmse'][zoom[0]:zoom[1]]
        test_loss = results['eval']['rmse'][zoom[0]:zoom[1]]
    if zoom_on == 'train':
        plt.figure()
        plt.plot(epochs, train_loss, label='Train')
        plt.legend()
        plt.ylabel('RMSE')
        plt.title('XGBoost Loss')
        plt.show()
    if zoom_on == 'test':
        plt.figure()
        plt.plot(epochs, test_loss, label='Test')
        plt.legend()
        plt.ylabel('RMSE')
        plt.title('XGBoost Loss')
        plt.show()
    if zoom_on is None:
        plt.figure()
        fig, ax = plt.subplots()
        ax.plot(epochs, train_loss, label='Train')
        ax.plot(epochs, test_loss, label='Test')
        ax.legend()
        plt.ylabel('RMSE')
        plt.title('XGBoost Loss')
        plt.show()


def mape(y_pred:np.array, y_true:np.array): 
    """
    Calculates the Mean Absolute Percentage Error for a given sets:
    1. Model predictons
    2. Ground truth labels
    """
    y_true = np.where(y_true == 0, 0.1, y_true)
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100


def inspect_distribution(column:pd.Series, compare:bool = False, sister_column:Optional[pd.Series] = None):
    """
    Function for inspecting the distribution of sample.
    Optionally the sister, or any other, column can be specified for quick comparisons.
    """
    if compare:
        plt.figure(figsize = (17, 5))
        plt.subplot(1, 2, 1)
    sns.histplot(column, 
                 color = 'navy'
                 )
    plt.title(f'Histogram of {column.name}')
    if compare:
        plt.subplot(1, 2, 2)
        sns.histplot(sister_column, 
                 color = 'orange'
                 )
        plt.title(f'Histogram of sister column')


def inspect_yearly(aggregated_data:pd.DataFrame, method:str):
    """
    Creating the clustermap or heatmap from the aggregating data frame.
    The data frame is expected to contain columns: year, month, trips (number of trips) and unique_trips (count based on index).
    """
    assert method in ['heatmap', 'clustermap'], f'No implemented: method: {method} not implemented!'
    pivot_trips = aggregated_data.pivot('year', 'month', 'trips')
    pivot_trips_unique = aggregated_data.pivot('year', 'month', 'unique_trips')   
    if method == 'heatmap':
        plt.figure(figsize = (20, 10))
        plt.subplot(1, 2, 1)
        sns.heatmap(pivot_trips)
        plt.title('Number of trips overall.')
        plt.yticks(rotation = 0)
        plt.subplot(1, 2, 2)
        sns.heatmap(pivot_trips_unique)
        plt.title('Number of unique trips, based on index.')
        plt.yticks(rotation = 0)
    if method == 'clustermap':
        plt.figure(figsize = (14, 7))
        sns.clustermap(pivot_trips)
        plt.title('Clustered trips.')
        plt.yticks(rotation = 0)
        plt.figure(figsize = (14, 7))
        sns.clustermap(pivot_trips_unique)
        plt.title('Clustered unique trips.')
        plt.yticks(rotation = 0)


def corrplot(df:pd.DataFrame, columns:[list]): 
    """
    Creates the heatmap based on correlation matrix. 
    Remeber that categorical variables are not taken into account.
    """
    sns.set_theme(style="white")
    df = df[columns]
    # Compute the correlation matrix
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    x_axis_labels = columns
    y_axis_labels = columns
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.yticks(rotation = 0)


def calculate_drivetime(df:pd.DataFrame):
    """
    Calculates the drive times in the column-wise operation. Used only for analysis, not processing and modelling.
    """
    return (pd.to_datetime(df['tpep_dropoff_datetime']) - pd.to_datetime(df['tpep_pickup_datetime'])).apply(lambda x: x.seconds/60)

