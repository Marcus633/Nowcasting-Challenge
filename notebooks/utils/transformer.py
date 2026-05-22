import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
import torch.nn as nn
from functools import partial


#Class for training the transformer and make predictions 
class Transformer:
    def __init__(self):
        pass

    #Create datapoints (X data) of lengths sequence_length to predict one single y value
    def to_sequences(self, sequence_length, X, y):
        if len(X) <= sequence_length or len(y) <= sequence_length:
            print(f"Warning: data length ({len(X)}, {len(y)}) is not greater than sequence_length ({sequence_length}).")
        
        x_seq = []
        y_seq = []

        #Iterate over the data, making sure we get full sequences of length `sequence_length`
        for i in range(sequence_length, len(X)+1, 1):
            #Create a block of `sequence_length` elements from X
            window_X = X[(i - sequence_length):i, :]
            #The target for this sequence is the target value of the last value of the window
            after_window_y = y[i-1]  #Using the y value of the last element of the sequence block
            x_seq.append(window_X)
            y_seq.append(after_window_y)
        
        #Convert to torch tensors
        return (
            torch.tensor(x_seq, dtype=torch.float32).view(-1, sequence_length, X.shape[1]),
            torch.tensor(y_seq, dtype=torch.float32).view(-1, 1),
        )
    
    
    def create_dataloader_batches(self, X_train, y_train, sequence_length, batch_size, val_size):
        
        y_train = y_train.copy()
        X_train = X_train.copy()

        #Scale the data
        scaler_X = StandardScaler()
        X_train_seq = scaler_X.fit_transform(X_train)

        scaler_y = StandardScaler()
        y_train_seq = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()

        #Sequence Data Preparation
        X_train, y_train = self.to_sequences(sequence_length, X_train_seq, y_train_seq)

        #Split initial sequence
        initial_sequence = X_train[-1, -(sequence_length-1):, :]

        #Split in train and val for validation
        X_train = X_train[:-val_size]
        X_val = X_train[-val_size:]
        y_train = y_train[:-val_size]
        y_val = y_train[-val_size:]

        #Setup data loaders -> shuffle of training is fine, since the data consists of blocks with size
        #sequence_length which perceive the temporal structure
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(X_val, y_val)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, scaler_X, scaler_y, initial_sequence
    

    #Function to train the transformer model
    def train_transformer(self, X_train, y_train, best_hyperparameters):

        #Extract the best hyperparameter for the data (for this specific country)
        best_d_model = best_hyperparameters.get('d_model')
        best_nhead = best_hyperparameters.get('nhead')
        best_num_layers = best_hyperparameters.get('num_layers')
        best_learning_rate = best_hyperparameters.get('learning_rate')
        best_batch_size = best_hyperparameters.get('batch_size')
        best_sequence_length = best_hyperparameters.get('sequence_length')
    
        #Set up the data, split in train and test set
        train_loader, test_loader, scaler_X, scaler_y, initial_sequence = self.create_dataloader_batches(X_train, y_train, best_sequence_length, best_batch_size, val_size=14)

        #Set up the model
        model = TransformerModel(input_dim=X_train.shape[1],d_model=best_d_model, nhead=best_nhead, num_layers=best_num_layers).to("cpu")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best_learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

        #Train and evaluate the model
        epochs = 1000
        early_stop_count = 0
        min_val_loss = float('inf')

        for epoch in range(epochs):

            #Train with training data and do backpropagation to update the model weights
            model.train()
            for batch in train_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to("cpu"), y_batch.to("cpu")

                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            #Evaluate the model with test data
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in test_loader:
                    x_batch, y_batch = batch
                    x_batch, y_batch = x_batch.to("cpu"), y_batch.to("cpu")
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())

            #Calculate loss
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)

            #Early Stopping
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= 5:
                break

        #Return model and initial sequence as starting point for the prediction
        model.eval()
        return model, scaler_X, scaler_y, initial_sequence, best_sequence_length

    #Function to predict the next values
    def predict_next_values(self, model, scaler_X, scaler_y, initial_sequence, seq_length, X_test, num_predictions):
        model.eval()
        predictions = []
        sequence = initial_sequence.clone()

        #Apply the scaler_X to all X_test inputs and concatenate the initial_sequence with the test data
        X_test_scaled = scaler_X.transform(X_test).astype(np.float32)
        sequence = torch.cat([sequence, torch.tensor(X_test_scaled, dtype=torch.float32)])

        #Sequence Data Preparation of test data, create one batch for all test instances
        x_seq = []
        for i in range(seq_length, len(sequence)+1, 1):
            window_X = sequence[(i - seq_length):i, :]
            x_seq.append(window_X)
        X_prepared = torch.stack(x_seq).view(-1, seq_length, sequence.shape[1])
        dataset = TensorDataset(X_prepared)
        dataloader = DataLoader(dataset, batch_size=int(num_predictions), shuffle=False)

        #Apply model and return the prediction
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(batch[0])
                predictions.append(outputs)
        predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        return predictions


#Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


#Define model to instantiate later
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)


    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x
    

#Class to perform hyperparameter search
class HyperparameterTuningTransformer:

    def __init__(self):
        pass


    def objective(self, trial, X_train, y_train):
        #Define Hyperparameters to tune and value ranges
        d_model = trial.suggest_categorical("d_model", [64, 128])
        nhead = trial.suggest_categorical("nhead", [4])
        num_layers = trial.suggest_int("num_layers", 2,3)
        dropout = trial.suggest_float("dropout", 0.2, 0.2)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        sequence_length = trial.suggest_categorical("sequence_length", [12, 24])

        #Sequence Data Preparation
        transformer = Transformer()
        #Create train / test loader batches, use approximately 20% (170*0,2 ~ 35) of the instances for validation
        train_loader, test_loader,_,_,_ = transformer.create_dataloader_batches(X_train, y_train, sequence_length, batch_size, val_size=35)

        model = TransformerModel(input_dim=X_train.shape[1], d_model=d_model, nhead=nhead,
                                num_layers=num_layers, dropout=dropout).to("cpu")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        #Train on training data
        epochs = 50
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                x_batch, y_batch = batch
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()


        #Evaluate on validation data
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in test_loader:
                    x_batch, y_batch = batch
                    x_batch, y_batch = x_batch.to("cpu"), y_batch.to("cpu")
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        return val_loss


    #Create the study and optimize
    def perform_hyperparameter_search(self, X_train, y_train, n_trials=50):
        study = optuna.create_study(direction="minimize")
        study.optimize(partial(self.objective, X_train=X_train, y_train=y_train), n_trials=n_trials)
        print("Best hyperparameters:", study.best_params)
        return study.best_params
