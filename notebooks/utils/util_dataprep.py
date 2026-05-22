import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class Util_DataPrep:
    def __init__(self):
        pass

    #TODO not used -> took too long time to execute for all missing values
    def predict_missing_middle_values_sarima(self, country_df, max_sarima_order=(1, 1, 1, 12), seasonal_order=(1, 1, 1, 12)):
            """
            Predicts and fills values that are currently 0 in the columns using SARIMA.
            Leaves values as 0 if SARIMA prediction is not possible due to lack of data.
            """
            df = country_df.copy()

            for col in df.columns:
                #Find the indices of zero values
                zero_indices = df[col][df[col] == 0].index

                #If the column is entirely zero, we can leave it as zero or attempt prediction
                if df[col].eq(0).all():
                    continue  #Nothing to do if the entire column is zero
                
                #For each zero value, check if there are valid values to base a SARIMA prediction on
                for i in zero_indices:
                    #Check if there are non-zero values after the zero, to determine if prediction is possible
                    if df[col][i:].ne(0).any():
                        non_zero_series = df[col].replace(0, float('nan')).dropna()

                        #Try fitting the SARIMA model, if possible
                        try:
                            model = SARIMAX(
                                non_zero_series,
                                order=max_sarima_order[:3],  #Non-seasonal order
                                seasonal_order=seasonal_order  #Seasonal order
                            )
                            model_fit = model.fit(disp=False)
                            #Predict the missing value
                            df[col].iloc[i] = model_fit.forecast()[0]
                        except Exception as e:
                            #If SARIMA fitting fails, leave the zero value as 0
                            #print(f"Warning: SARIMA model fitting failed for column '{col}' at index {i}. Left as 0.")
                            df[col].iloc[i] = 0
                    else:
                        #If there are no non-zero values after the zero, leave the value as 0
                        df[col].iloc[i] = 0

            return df


    #Get the number of missing months in the data for all X columns
    def getColumnsWithNumberOfMissingMonth(self, df, target_index, target_column):

        #Dictionary to store distances for the current sheet
        sheet_distances = {}
        
        #Calculate the distance between the last valid index and the target index for each column
        for column in df.columns.drop(target_column):
            #Drop NaN values to find the last valid index
            non_nan_series = df[column].dropna()
            
            #If the column has any non-missing values, calculate the distance
            if not non_nan_series.empty:
                last_valid_index = non_nan_series.index[-1]
                #Calculate the distance to the target index
                distance = target_index - last_valid_index
            else:
                #If all values are missing, set distance as target_index + 1
                distance = target_index + 1  #or any other value to indicate no data
            
            #Store the result for the column
            sheet_distances[column] = int(distance)
        
        return sheet_distances


    #Shift columns with missing values forward to the index of the current month (e.g. December 2024 == 204)
    def shift_columns(self, df, shift_dict):
        #Make a copy of the DataFrame to avoid modifying the original
        df = df.copy()
        
        #Generate new date entries if 'Date' is in the shift_dict
        if 'Date' in shift_dict:
            #Get the last date in the 'Date' column
            last_date = df['Date'].iloc[-1]
            
            #Generate new dates by adding 1 month at a time
            new_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(shift_dict['Date'])]
            
            #Append these new dates to the DataFrame
            new_df = pd.DataFrame({'Date': new_dates})
            df = pd.concat([df, new_df], ignore_index=True)
        
        #Shift each column based on the values in shift_dict
        for column, shift_value in shift_dict.items():
            if column == 'Date':
                #Date column is already handled with new entries
                continue
            elif column == 'month':
                #Extract month from the 'Date' column
                df['month'] = df['Date'].dt.month
            elif column == 'year':
                #Extract year from the 'Date' column
                df['year'] = df['Date'].dt.year
            else:
                #Shift the specified column and use bfill to fill missing values
                df[column] = df[column].shift(shift_value).bfill()

        return df
    

    #Get number of missing targets (Available to internal market) -> This equals the number of
    #predictions which have to be made
    def getNumberOfMissingTargets(self, df, target_index,target_column):
        df = df.copy()
        non_nan_series = df[target_column].dropna()
        last_valid_index = non_nan_series.index[-1]
        distance = target_index - last_valid_index
        return distance


    #For training purpose -> remove the last 4 rows because of missing y and split the data in train / test
    def remove_data_and_split(self, df, number_of_missing_targets, X_cols, y_col):
        
        #Copy dataframe
        shifted_df = df.copy()

        #Remove last 4 rows for training due to missing data
        #TODO comment out this line for final prediction
        shifted_df = shifted_df.iloc[:-4]

        #Split the data in train / test set
        shifted_df_train = shifted_df.iloc[:-number_of_missing_targets]
        shifted_df_test = shifted_df.iloc[-number_of_missing_targets:]

        X_train = shifted_df_train[X_cols].apply(
            lambda col: col.astype('float64') if col.name not in ['Date', 'month', 'year'] else col
        )
        y_train = shifted_df_train[y_col].astype('float64')
        X_test = shifted_df_test[X_cols].apply(
            lambda col: col.astype('float64') if col.name not in ['Date', 'month', 'year'] else col
        )
        y_test = shifted_df_test[y_col]

        return X_train, y_train, X_test, y_test