# Nowcasting Challenge

This seminar deals with the prediction of monthly electricity availability by estimating current values for the energy balance “Available to internal market”.


## Installation

1. Clone the repo
```bash
#Open a terminal, naviate to a suitable directory and type:
git clone https://github.com/Marcus633/Nowcasting-Challenge.git
```

2. Create a New Virtual Environment
```bash
#Open the repo in an IDE (e.g. VS code), open a terminal there and type:
python -m venv nowcasting_venv
```

3. Install the necessary dependencies
```bash
#Acvivate the venv
nowcasting_venv\Scripts\activate

#Install dependencies
pip install -r requirements.txt
```


## Usage
In the notebooks/execute directory there are 2 .ipynb files that can be executed (with "run all"):

1. merge_datasets.ipynb
    - Is used to preprocess and merge the datasets from the different sources
    - The result is a dict with sheets of the data for all countries saved in countries_data.xlsx

2. make_prediction.ipynb
    - Is used to train the models and make the predictions
    - The result is a dict of msre values representing the performance of the different models.


## Customizing

### Customizing the data
To replace the existing data versioned in the repo follow the steps:
1. Fetch the new data from the Eurostat page
    - Go to https://ec.europa.eu/eurostat/databrowser/view/nrg_cb_em/default/table?lang=en
    - Scroll a little down, click on the download symbol
    - For "Select data" choose: "Full dataset"
    - Keep the remaining preselected values (Spreadsheet (.xlsx) and Include flags)
    - Click on download and replace the "data\nrg_cb_em_spreadsheet.xlsx" with the new downloaded data

2. Fetch the new data from the CDS page
    - Go to https://cds.climate.copernicus.eu/datasets/sis-energy-derived-reanalysis?tab=download
    - For the variables "Meterology" and "Energy", click on "Select all"
    - For "Spatial aggregation" choose "Country level"
    - For "Energy product type" select "Energy"
    - For "Temporal aggregation" select "Monthly"
    - "Year" and "Month" are selected automatically, since the data aggregated on the various levels are stored in single files for the entire timeseries
    - Click on download and replace the fils in "data\CDS_energy" with the new downloaded data

**Important:** If the data is updated and you want to predict newer months with the data, keep in mind to update the ```month_to_predict_idx``` variable (in the notebook/execute/make_prediction.ipynb file, Cell 7). Examples:
- 202 = November 2024
- 203 = Dezember 2024
- 204 = Januar 2025
- ...

### Customize the models
For the model training there are several things that can be customized.
1. Customize the data
    - Cell 6: If you want to change the features (X) that are included in the training data, you can delete entries in the X_cols variable
    - Cell 7: If you want to reduce the number of years in the training data, you can modify the line ```shifted_df = shifted_df[shifted_df['year'] >= 2017]```
    - Cell 7: You can also adjust the train/test split defined in ```remove_data_and_split()```

2. Customize the hyperparameters
    - Default behavior is to load the pretrained hyperparameters from the notebook/hyperparameter directory
    - Cell 9: As shown for the transformer (code is commented out), you can use the notebooks/utils/util_hyperparameter.py files to find hyperparameter tuning functions for all models

3. Customize the Ensembles
    - If you want to use different models for an ensemble prediction, you can run the ```evaluateDifferentEnsembles()``` method to evaluate which models work the best together
    - Afterwards you can change the ```list_of_predictions``` variables to include the predictions of the models you want to combine
    
    
