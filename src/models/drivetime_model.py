import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from tqdm import tqdm
from pathlib import Path
import os

from generators.drivetime_generator import batch_generator

def generate_drivetime_model(batch:pd.DataFrame,
                             incremental_learning:bool = False,
                             grid_search:bool = False,
                             early_stopping_rounds:int = 100,
                             eval_metric:Optional = None,
                             params:Optional[dict] = None,
                             param_space:Optional[dict] = None,
                             n_trials:Optional[int] = 1000):
    """
    Generates both regular and bulk model.
    GPU support is enabled by default it may be changed by manipulationg the tree_method parameter.
    
    If the incremental learning switch is set for True:
        1. Function expects the output of the drivetime_data_generator as the batch.
        2. Then the incremental learning is performed with custom or default parameters. Verbosity equal to 0
            is recommended to supress the communicates within a training loop.
        3. Also the directory in which the model will be stored should be cleaned.
    With incremental learning set to False:
        1. The grid search may be performed, just remeber that it may be time-consuming.
        2. Otherwise the Booster will be fitted to the data provided.
        3. In this case batch of data is expected to be the output of batch_generator function
        
    Additional note:
    Exemplary parameter space for the grid search:
        param_space = {'learning_rate': hp.uniform('learning_rate', 0.01, 0.3), 
                       'n_round': scope.int(hp.quniform('n_round', 200, 3000, 100)),
                       'max_depth': scope.int(hp.quniform('max_depth', 5, 16, 1)), 
                       'gamma': hp.uniform('gamma', 0, 10), 
                       'min_child_weight': hp.uniform('min_child_weight', 0, 10),
                       'subsample': hp.uniform('subsample', 0.1, 1), 
                       'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1)
                       }
    """
    if not incremental_learning:
        (X_train, y_train), (X_test, y_test) = batch_generator(batch)
    else:
        (X_train, y_train), (X_test, y_test) = batch
        
    training_set = xgb.DMatrix(X_train, y_train)
    validation_set = xgb.DMatrix(X_test, y_test)
    
    if grid_search:
        assert isinstance(param_space, dict), """The param space dictionary needs to be specified 
                                                 when performing grid search. Please see the documentation 
                                                 to see exemplary param space."""
        def score(params):
            ps = {'learning_rate': params['learning_rate'],
                  'max_depth': params['max_depth'],
                  'gamma': params['gamma'],
                  'min_child_weight': params['min_child_weight'],
                  'subsample': params['subsample'], 
                  'colsample_bytree': params['colsample_bytree'],
                  'verbosity': 1,
                  'objective': 'reg:squarederror',
                  'eval_metric': 'rmse',
                  'tree_method': 'gpu_hist',
                  'random_state': 1110
                }

            model = xgb.train(ps,
                              training_set,
                              params['n_round'],
                              evals = [(validation_set,'eval'), (training_set,'train')],
                              early_stopping_rounds=100,
                              verbose_eval = False)
            y_pred = model.predict(validation_set)
            score = mean_squared_error(y_test, y_pred,squared=False)
            return score
        
        trials = Trials()
        assert isinstance(n_trials, int), 'Please specify number of trials (n_trials argument) as integer.'
        optimal_hp = fmin(fn = score,
                          space = param_space,
                          algo = tpe.suggest,
                          max_evals = n_trials,
                          trials = trials, 
                          )
        return optimal_hp, trials
    
    else:
        ps = {'objective': 'reg:squarederror',
              'tree_method': 'gpu_hist',
              'eval_metric': eval_metric,
              'random_state': 0,
              'n_round' : 100000,
              'verbosity' : 0}
        if params is not None:
            ps = params
        
        evals_result = {}
        if incremental_learning and Path('stored_model/xg_temp_model.model').is_file():
            model = xgb.train(ps,
                              training_set,
                              ps['n_round'],
                              evals = [(validation_set,'eval'), (training_set,'train')],
                              early_stopping_rounds = early_stopping_rounds,
                              evals_result = {},
                              verbose_eval = False,
                              xgb_model = '/stored_model/xg_temp_model.model')
        
        model = xgb.train(ps,
                          training_set,
                          ps['n_round'],
                          evals = [(validation_set,'eval'), (training_set,'train')],
                          evals_result = evals_result,
                          early_stopping_rounds = early_stopping_rounds,
                          verbose_eval = False)
        
        if incremental_learning:
            model.save_model('/stored_model/xg_temp_model.model')
            return model, evals_result
        
        return model, evals_result