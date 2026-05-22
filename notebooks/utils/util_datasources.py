import glob
import os
import pandas as pd
from pandas.tseries.offsets import MonthEnd


class Util_DataSources:
    def __init__(self):
        pass


    #Load the different sheets of the CDS excel files
    def load_cds_excel_sheets(self, file_path):
        #Use glob to get a list of all CSV files in the directory
        csv_files = glob.glob(os.path.join(file_path, "*.csv"))

        #Create a list to hold each DataFrame
        dataframes_cds = {}

        #Loop over the list of CSV files and read each one into a DataFrame
        for file in csv_files:
            df = pd.read_csv(file, skiprows = 52)

            #Load Metadata for name
            metadata_df = pd.read_csv(file, nrows=10, header=None)
            series_name = metadata_df.iloc[2]
            #Clean the series name
            series_name = str(series_name).replace('### ', '').replace(' ', '').replace("0", "")  #Basic replacements
            cleaned_series_name = series_name.split('\n')[0]
            dataframes_cds[cleaned_series_name] = df

        #Print out information about each DataFrame
        for series_name, df in dataframes_cds.items():
            print(f"DataFrame for series: {series_name}")

        return dataframes_cds


    #Load the different sheets of the Eurostat excel
    def load_eurostat_excel_sheets(self, file_path):

        #Read all sheets into a dictionary where the key is the sheet name
        sheets_dict = pd.read_excel(file_path, sheet_name=None, skiprows=9)
        
        #Print the sheet names and their corresponding DataFrames
        df_list = []
        for sheet_name, df in sheets_dict.items():
            print(f"DataFrame for sheet: {sheet_name}")
            df_list.append(df)
        
        return df_list


    #Preprocess the CDS data
    def prep_cds_data(self, dataframes, country_mapping):
        dataframes_new = {}

        for key, df in dataframes.items():
            #Convert to datetime format
            df["Date"] = pd.to_datetime(df['Date'])
            #Create features based on the index: month and year
            df['month'] = df['Date'].dt.month
            df['year'] = df['Date'].dt.year
            df.rename(columns=country_mapping, inplace=True)
            columns_to_keep = list(country_mapping.values())
            columns_to_keep += ["Date", "month", "year"]
            for column in columns_to_keep:
                if column not in df.columns:
                    df[column] = 0
            df = df[columns_to_keep]
            dataframes_new[key] = df
        return dataframes_new
    

    #Eurostat data has a second column for each month to define a flag (p / e) -> drop this column
    def extract_eurostat_data(self, dataframes):

        cleaned_dfs = []

        for df in dataframes:
            #Drop columns with NaN values
            cleaned_df = df.dropna(axis=1, how='all')
            cleaned_df = cleaned_df.drop(index=0)

            #Drop unnamed columns
            columns_to_drop = [col for col in cleaned_df.columns if 'Unnamed' in col]
            cleaned_df = cleaned_df.drop(columns=columns_to_drop, errors='ignore')

            #Country data only up to line 41, after that only NaNs
            cleaned_df = cleaned_df.iloc[:41]

            cleaned_dfs.append(cleaned_df)

        return cleaned_dfs

    
    #Set the index and fill missing middle values
    def prep_eurostat_data(self, data):
        #Set TIME as the index and transpose
        data = data.set_index('TIME').T

        #Reset the index to turn TIME back into a column
        data = data.reset_index(inplace=False)

        #Convert the 'index' column (which is the former TIME) to datetime
        data.rename(columns={'index': 'Date'}, inplace=True)
        data['Date'] = pd.to_datetime(data['Date']) + MonthEnd(1)

        #Fill missing middle values with mean of previous and next
        data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').apply(
            lambda col: col.fillna((col.shift(1) + col.shift(-1)) / 2)
        )

        #Create features based on the index: month and year
        data['month'] = data['Date'].dt.month
        data['year'] = data['Date'].dt.year

        #Backfill remaining missing values instead of filling with 0 or anything else
        data = data.bfill()

        return data
    

    #Create the final dict with country name as key and all data for this country (DataFrame) as value
    def create_country_dict_with_starting_year(self, result_df, starting_year):
        
        #Filter rows where the "year" is before starting_year
        df_from_starting_year = result_df[result_df['year'] >= starting_year]

        #Create a dictionary with a DataFrame per country
        country_dataframes = {}

        #List of columns to retain in each country's DataFrame
        common_columns = ['Date', 'month', 'year']

        #Loop over each country column (ignoring "Date", "month", "year", and "type" columns)
        for country in df_from_starting_year.columns[:-4]:  #Exclude the last four columns
            #Pivot the data to set 'type' values as columns
            country_df = df_from_starting_year.pivot_table(index=common_columns, columns='type', values=country).reset_index()
            #Store in dictionary with the country name as the key
            country_dataframes[country] = country_df

        return country_dataframes