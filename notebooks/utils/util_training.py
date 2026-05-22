import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
import itertools


class Util_Training:
    def __init__(self):
        pass


    #Function to combine predictions from individual models to one ensemble prediction
    def make_ensemble_prediction(self, list_of_predictions):

        #Check if the list contains single elements (not lists)
        if all(isinstance(pred, (np.floating, float)) for pred in list_of_predictions):
            #Directly return the average if it's a flat list
            return sum(list_of_predictions) / len(list_of_predictions)
        
        #Otherwise, assume it's a list of lists and transpose
        transposed = list(zip(*list_of_predictions))
        
        #Calculate the mean for each group of predictions
        ensemble_mean = np.mean(transposed, axis=1)
        return ensemble_mean
    

    #Try out all possible ensemble combinations to identify the best ensemble
    def evaluateDifferentEnsembles(self, predictions_dict, y_test):

        #Placeholder for ensemble results
        ensemble_dict = {}

        #Loop through all subsets of models of size 2 or more
        for r in range(2, len(predictions_dict) + 1):  #Start at 2 to exclude subsets of size 0 and 1
            for subset_keys in itertools.combinations(predictions_dict.keys(), r):  #Generate subsets of model names
                subset_preds = [predictions_dict[key] for key in subset_keys]  #Retrieve predictions for the subset
                y_pred_ensemble = self.make_ensemble_prediction(subset_preds)  #Create ensemble prediction
                
                #Calculate MSRE
                ensemble_msre = np.mean(((y_test - y_pred_ensemble) / y_test) ** 2)
                
                #Create a name for the combination and store MSRE
                combination_name = " + ".join(subset_keys)
                ensemble_dict[combination_name] = ensemble_msre

        #Find the best combination (minimum MSRE)
        min_value = min(ensemble_dict.values())  #Find the minimum MSRE value
        result = {k: v for k, v in ensemble_dict.items() if v == min_value}  #Filter for the best combination

        #Print the best combination and its MSRE
        for key, value in result.items():
            print(f"Best Combination: {key}, MSRE: {value}")
        
        return min_value
    

    #Function to train and predict with ridge regressor with the pretuned hyperparameters
    def train_ridge_and_predict(self, X_train, y_train, X_test, best_hyperparameters):

        #Get the best alpha
        best_alpha = best_hyperparameters.get('ridge__alpha')

        #Scale features for Ridge Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #Initialize and fit the Ridge Regressor on scaled data
        ridge_model = Ridge(alpha=best_alpha)
        ridge_model.fit(X_train_scaled, y_train)
        y_pred_ridge = ridge_model.predict(X_test_scaled)

        return y_pred_ridge
    

    #Function to train and predict with random forest with the pretuned hyperparameters
    def train_random_forest_and_predict(self, X_train, y_train, X_test, best_hyperparameters):

        #Get the best hyperparameters
        best_max_depth = best_hyperparameters.get('max_depth')
        best_max_features = best_hyperparameters.get('max_features')
        best_min_samples_leaf = best_hyperparameters.get('min_samples_leaf')
        best_min_samples_split = best_hyperparameters.get('min_samples_split')
        best_n_estimators = best_hyperparameters.get('n_estimators')

        #Train model and predict
        rf_model = RandomForestRegressor(max_depth=best_max_depth, max_features=best_max_features,
                                         min_samples_leaf=best_min_samples_leaf,
                                         min_samples_split=best_min_samples_split,
                                         n_estimators=best_n_estimators)
        
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        return y_pred_rf
    
    
    #Function to train and predict with xgb with the pretuned hyperparameters
    def train_xgb_and_predict(self, X_train, y_train, X_test, best_hyperparameters):

        #Get the best hyperparameters
        best_colsample_bytree = best_hyperparameters.get('colsample_bytree')
        best_learning_rate = best_hyperparameters.get('learning_rate')
        best_max_depth = best_hyperparameters.get('max_depth')
        best_n_estimators = best_hyperparameters.get('n_estimators')
        best_subsample = best_hyperparameters.get('subsample')

        #Train model and predict
        xgb_model = XGBRegressor(colsample_bytree = best_colsample_bytree, learning_rate = best_learning_rate,
                                 max_depth=best_max_depth, n_estimators= best_n_estimators,
                                 subsample=best_subsample)
        xgb_model.fit(X_train, y_train)
        y_pred_rf = xgb_model.predict(X_test)

        return y_pred_rf
    

    #Function to train and predict with linear regression
    def train_lin_reg_and_predict(self, X_train, y_train, X_test):

        #Train model and predict
        lin_model = LinearRegression()
        lin_model.fit(X_train, y_train)
        y_pred_rf = lin_model.predict(X_test)

        return y_pred_rf


    #Function to train and predict with sarima with the pretuned hyperparameters
    def train_sarima_and_predict(self, y_train, model_order, seasonal_order, number_of_missing_targets):
        
        #Fit the best SARIMAX model
        best_model = SARIMAX(y_train, order=model_order, seasonal_order=seasonal_order)
        results = best_model.fit(disp=False)

        #Forecast number_of_missing_targets new values
        forecast = results.forecast(steps=int(number_of_missing_targets))
        return forecast.values
    

    #Function to calculate the msre in different setting (one prediction or list of predictions)
    def calculate_msre(self, y_test, y_pred):

        #If y_test and y_pred have multiple values
        if isinstance(y_test, pd.Series) and isinstance(y_pred, np.ndarray):
            y_test = y_test.to_numpy() #Convert pandas Series to numpy array

            # Check if their lengths match
            if y_test.shape != y_pred.shape:
                raise ValueError("y_test and y_pred must have the same dimensions.")
            
            msre = np.mean(((y_test - y_pred) / y_test) ** 2)
            return msre

        #If y_test and y_pred have one value
        elif isinstance(y_test, np.float64) and (isinstance(y_pred, np.float64) or isinstance(y_pred, np.float32)):
            msre = np.mean(((y_test - y_pred) / y_test) ** 2)
            return msre
        
        else:
            raise TypeError("The types of y_test and y_pred must be compatible.")
