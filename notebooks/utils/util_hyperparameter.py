from pmdarima import auto_arima
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

class Util_Hyperparameter:
    def __init__(self):
        pass

    #Define hyperparameter search for ridge regression
    def perform_ridge_hyperparameter_search(self, X_train, y_train):
        param_grid = {
            'ridge__alpha': [1e-6, 1e-4, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e6]
        }

        ridge_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])

        grid_search = GridSearchCV(
            estimator=ridge_pipeline,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        return grid_search.best_params_
    

    #Define hyperparameter search for random forest
    def perform_rf_hyperparameter_search(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        rf = RandomForestRegressor(random_state=42)

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        return grid_search.best_params_
    

    #Define hyperparameter search for xgb
    def perform_xgb_hyperparameter_search(self, X_train, y_train):

        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        xgb = XGBRegressor()

        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        return grid_search.best_params_
    

    #Define hyperparameter search for SARIMA
    def tune_sarima_params(self, y_train):

        #Fit auto_arima to find the best parameters
        model = auto_arima(
            y_train,
            start_p=0, start_q=0, max_p=3, max_q=3,
            seasonal=True, m=12,  #Monthly data, seasonality of 12
            start_P=0, start_Q=0, max_P=2, max_Q=2,
            d=None, D=1, trace=True,
            stepwise=True,
            error_action="ignore", suppress_warnings=True
        )

        #Print and return the best parameters
        print("Best model order:", model.order)
        print("Best seasonal order:", model.seasonal_order)
        return model.order, model.seasonal_order