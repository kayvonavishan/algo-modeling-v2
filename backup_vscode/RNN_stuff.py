

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import pickle
import gc

######################
# RNN
######################
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def train_and_save_rnn_model(
    model_path, scaler_path, master_df, columns_to_include, label_column, seq_length=26, hidden_dim=99,
    num_layers=1, num_epochs=3000, learning_rate=0.0052, prediction_horizon=1, scale_data=True
):
    print("Starting RNN model training...")

    # Step 1: Create df_clean with only the specified columns
    print("Preparing data...")
    df_clean = master_df[columns_to_include].copy()
    print(f"Data shape after selecting columns: {df_clean.shape}")
    
    # Step 2: Remove all rows that contain any NaN values
    df_clean = df_clean.dropna()
    print(f"Data shape after dropping NaN values: {df_clean.shape}")
    
    if scale_data:
        print("Scaling data...")
        # Step 3: Initialize the MinMaxScaler
        scaler = MinMaxScaler()
        
        # Step 4: Fit the scaler to the data and transform
        scaled_array = scaler.fit_transform(df_clean)

        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print("Data scaling complete.")
        
        # Step 5: Create a scaled DataFrame
        df_scaled = pd.DataFrame(scaled_array, index=df_clean.index, columns=df_clean.columns)
    else:
        print("Skipping data scaling as per user request.")
        # If not scaling, use the original data
        df_scaled = df_clean.copy()
        scaler = None  # No scaler needed

    print(f"Data ready for splitting: {df_scaled.shape}")

    # Calculate the total number of data points
    data_length = len(df_scaled)
    
    # Split the data into training and testing sets (80% train, 20% test)
    split_point = int(data_length * 0.8)
    train_data = df_scaled.iloc[:split_point]
    test_data = df_scaled.iloc[split_point:]
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    def create_sequences(data, sequence_length=10, horizon=1):
        sequences = []
        targets = []
        for i in range(len(data) - sequence_length - horizon + 1):
            seq = data.iloc[i:i+sequence_length].values
            label = data.iloc[i+sequence_length+horizon-1][label_column]
            sequences.append(seq)
            targets.append(label)
        return np.array(sequences), np.array(targets)
    
    # Creating sequences for training and testing data
    print("Creating sequences for training and testing data...")
    X_train, y_train = create_sequences(train_data, sequence_length=seq_length, horizon=prediction_horizon)
    X_test, y_test = create_sequences(test_data, sequence_length=seq_length, horizon=prediction_horizon)
    print(f"Number of training sequences: {X_train.shape[0]}")
    print(f"Number of testing sequences: {X_test.shape[0]}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Detecting the number of features in the training data
    input_dim = X_train.shape[2]
    print(f"Input dimension (number of features): {input_dim}")
    
    # Create the RNN model
    print("Initializing RNN model...")
    model = RNNModel(input_dim, hidden_dim, num_layers).to(device)
    print("Model initialized.")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Loss function and optimizer set.")
    
    # Convert data to tensors and move to configured device
    print("Converting data to tensors...")
    X_train_tensor = torch.tensor(X_train).float().to(device)
    y_train_tensor = torch.tensor(y_train).float().to(device)
    X_test_tensor = torch.tensor(X_test).float().to(device)
    y_test_tensor = torch.tensor(y_test).float().to(device)
    print("Data conversion complete.")
    
    # Initialize variables to track the best model
    best_rmse = float('inf')
    best_model = None  # Track the best model instead of just state
    
    # Training loop
    print("Starting training loop...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train_tensor)
        train_loss = criterion(train_outputs, y_train_tensor.unsqueeze(1))
        train_loss.backward()
        optimizer.step()
        
        # Evaluate on training data
        model.eval()
        with torch.no_grad():
            train_predicted = model(X_train_tensor)
            train_rmse = np.sqrt(criterion(train_predicted, y_train_tensor.unsqueeze(1)).item())
            
            # Evaluate on testing data
            test_predicted = model(X_test_tensor)
            test_rmse = np.sqrt(criterion(test_predicted, y_test_tensor.unsqueeze(1)).item())
    
            # Check if this is the best RMSE we've seen so far
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_model = model  # Save the entire model, not just the state
    
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training RMSE: {train_rmse:.7f}, Testing RMSE: {test_rmse:.7f}')
    
    # After training is complete, save the best model (architecture + weights)
    print("Training completed.")
    torch.save(best_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Optional: Print the best test RMSE
    print(f'Best Test RMSE: {best_rmse:.6f}')

    import gc
    
    print("Cleaning up...")
    # Attempt to delete 'df_clean' and 'df_scaled' if they exist
    for var in ['df_clean', 'df_scaled']:
        if var in globals():
            del globals()[var]
            print(f"Deleted variable '{var}'.")
        else:
            print(f"Variable '{var}' does not exist.")
    
    # Force garbage collection to reclaim memory
    collected = gc.collect()
    print(f"Garbage collector: collected {collected} objects.")
    
    # PLOTTING
    print("Generating predictions for plotting...")
    loaded_model = torch.load(model_path)
    loaded_model.eval()
    
    with torch.no_grad():
        best_test_predicted = loaded_model(X_test_tensor)
    print("Predictions generated.")
    
   # Convert predictions and actual values to numpy arrays
    predicted_values = best_test_predicted.cpu().detach().numpy().flatten()[:200]
    actual_values = y_test_tensor.cpu().detach().numpy().flatten()[:200]
    
    if scale_data and scaler is not None:
        print("Unscaling predictions and actual values...")
        # Get MinMaxScaler parameters for target column
        target_column_idx = columns_to_include.index(label_column)
        target_min = scaler.data_min_[target_column_idx]
        target_max = scaler.data_max_[target_column_idx]
        
        # Inverse transform using the MinMaxScaler formula: X = X_scaled * (max - min) + min
        original_predictions = predicted_values * (target_max - target_min) + target_min
        original_actuals = actual_values * (target_max - target_min) + target_min
        print("Unscaling complete.")
    else:
        print("Skipping unscaling as data was not scaled.")
        original_predictions = predicted_values
        original_actuals = actual_values

    
    # Create traces for Plotly
    print("Creating plot...")
    trace1 = go.Scatter(x=np.arange(len(original_predictions)), y=original_predictions, 
                       mode='lines', name='Predicted Values')
    trace2 = go.Scatter(x=np.arange(len(original_actuals)), y=original_actuals, 
                       mode='lines', name='Actual Values')
    
    layout = go.Layout(
        title='Comparison of Predicted and Actual Values',
        xaxis={'title': 'Index'},
        yaxis={'title': 'Values'},
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.show()
    print("Plot displayed.")


######################
# NBEATS ALL FEATURES
######################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import gc
from sklearn.preprocessing import StandardScaler

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_features):
        super(NBeatsBlock, self).__init__()
        self.fc1 = nn.Linear(input_size * num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.theta_layer = nn.Linear(hidden_size, input_size * num_features + 1)  # +1 for forecast
        self.input_size = input_size
        self.num_features = num_features

    def forward(self, x):
        batch_size = x.size(0)
        # Flatten the input
        x = x.view(batch_size, -1)  # Flatten to (batch_size, input_size * num_features)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        theta = self.theta_layer(x)
        
        backcast = theta[:, :-1]  # All but last value
        forecast = theta[:, -1].unsqueeze(1)  # Last value, make it 2D
        return backcast, forecast

class NBeatsStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, num_features):
        super(NBeatsStack, self).__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=input_size,
                hidden_size=hidden_size,
                num_features=num_features
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x, return_backcast=False):
        residuals = x
        forecast = torch.zeros(x.size(0), 1).to(x.device)
        backcasts = []
        
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals.view(residuals.size(0), -1)  # Flatten residuals
            backcast = backcast.view(backcast.size(0), -1)  # Flatten backcast
            residuals = residuals - backcast
            forecast = forecast + block_forecast
            if return_backcast:
                backcasts.append(backcast)
                
        if return_backcast:
            return forecast, backcasts
        return forecast

class NBeatsModel(nn.Module):
    def __init__(self, input_size, num_features, num_stacks, num_blocks_per_stack, hidden_size, is_binary=0):
        super(NBeatsModel, self).__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.is_binary = is_binary
        self.stacks = nn.ModuleList([
            NBeatsStack(
                input_size=input_size,
                hidden_size=hidden_size,
                num_blocks=num_blocks_per_stack,
                num_features=num_features
            )
            for _ in range(num_stacks)
        ])
        if is_binary:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_decomposition=False):
        batch_size = x.size(0)
        stack_input = x
        forecast = torch.zeros(batch_size, 1).to(x.device)
        
        if return_decomposition:
            forecasts = []
            backcasts = []
            
            for stack in self.stacks:
                stack_forecast, stack_backcasts = stack(stack_input, return_backcast=True)
                stack_input = stack_input.view(batch_size, -1)
                forecast = forecast + stack_forecast
                forecasts.append(stack_forecast)
                backcasts.append(stack_backcasts)
            
            if self.is_binary:
                forecast = self.sigmoid(forecast)
            return forecast, forecasts, backcasts
            
        for stack in self.stacks:
            stack_forecast = stack(stack_input)
            forecast = forecast + stack_forecast
        
        if self.is_binary:
            forecast = self.sigmoid(forecast)
        return forecast

def train_and_save_nbeats_model(
    model_path, scaler_path, master_df, columns_to_include, label_column, 
    seq_length=10, hidden_dim=256, num_stacks=2, num_blocks_per_stack=3,
    num_epochs=30, learning_rate=0.001, prediction_horizon=1, batch_size=512, 
    scale_data=True, is_binary=0
):
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Starting N-BEATS model training...")
    
    # Step 1: Create df_clean with only the specified columns
    print("Preparing data...")
    df_clean = master_df[columns_to_include].copy()
    df_clean = df_clean.dropna()
    print(f"Data after dropping NA: {df_clean.shape}")
    
    if scale_data:
        # Step 2: Scale the data
        print("Scaling data...")
        scaler = MinMaxScaler()
        scaled_array = scaler.fit_transform(df_clean)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print("Data scaling complete. Scaler saved.")
        df_scaled = pd.DataFrame(scaled_array, index=df_clean.index, columns=df_clean.columns)
    else:
        # If not scaling, use the original data
        print("Skipping data scaling as per user request.")
        df_scaled = df_clean.copy()
        scaler = None  # No scaler is used

    # round down
    df_scaled = df_scaled.round(2)

    # Split data
    print("Splitting data into training and testing sets...")
    split_point = int(len(df_scaled) * 0.8)
    train_data = df_scaled.iloc[:split_point]
    test_data = df_scaled.iloc[split_point:]
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    def create_sequences(data, sequence_length=10, horizon=1):
        sequences = []
        targets = []
        for i in range(len(data) - sequence_length - horizon + 1):
            seq = data.iloc[i:i+sequence_length].values  # Use all columns
            label = data.iloc[i+sequence_length+horizon-1][label_column]
            sequences.append(seq)
            targets.append(label)
        return np.array(sequences), np.array(targets)
    
    # Create sequences
    print("Creating sequences...")
    X_train, y_train = create_sequences(train_data, sequence_length=seq_length, horizon=prediction_horizon)
    X_test, y_test = create_sequences(test_data, sequence_length=seq_length, horizon=prediction_horizon)
    print(f"Number of training sequences: {X_train.shape[0]}")
    print(f"Number of testing sequences: {X_test.shape[0]}")
    
    # Determine the number of features
    num_features = X_train.shape[2]
    input_size = seq_length  # Corrected: Length of sequences
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Initializing N-BEATS model...")
    model = NBeatsModel(
        input_size=input_size,
        num_features=num_features,
        num_stacks=num_stacks,
        num_blocks_per_stack=num_blocks_per_stack,
        hidden_size=hidden_dim,
        is_binary=is_binary
    ).to(device)
    print(f"Model initialized with {num_stacks} stacks, {num_blocks_per_stack} blocks per stack")

    
    # Loss and optimizer
    criterion = nn.BCELoss() if is_binary else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Loss function and optimizer set.")
    
    # Convert to PyTorch datasets and dataloaders
    print("Preparing data loaders...")
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Adjust based on your CPU cores
    )
    print("Data loaders ready.")
    
    # Convert test data to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    best_rmse = float('inf')
    best_model_state = None
    best_metric = 0 if is_binary else float('inf')
    
    print("Starting training loop...")
    # Training loop with batches
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1
            
            # Clear GPU cache for batch
            del batch_X, batch_y, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_train_loss = total_train_loss / batch_count
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Training Loss: {avg_train_loss:.7f}")
       
        # Evaluate after every epoch
        model.eval()
        with torch.no_grad():
            test_predictions = []
            chunk_size = batch_size
            for i in range(0, len(X_test_tensor), chunk_size):
                chunk_X = X_test_tensor[i:i+chunk_size]
                chunk_pred = model(chunk_X)
                test_predictions.append(chunk_pred)
            
            test_predicted = torch.cat(test_predictions, dim=0)
            
            if is_binary:
                binary_preds = (test_predicted.squeeze() > 0.5).float()
                metric = (binary_preds == y_test_tensor).float().mean().item()
                metric_name = "accuracy"
            else:
                metric = np.sqrt(criterion(test_predicted.squeeze(), y_test_tensor).item())
                metric_name = "RMSE"
            
            if is_binary:
                is_better = metric > best_metric
            else:
                is_better = metric < best_metric
            
            if is_better:
                best_metric = metric
                best_model_state = model.state_dict().copy()
                print(f"Epoch [{epoch+1}/{num_epochs}] - New best {metric_name}: {metric:.7f}")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}] - {metric_name}: {metric:.7f}")
            
            del test_predictions, test_predicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    
    # Load best model state and save
    print("Training completed.")
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), model_path)
    
    # Conditional printing based on task type
    if is_binary:
        print(f'Best Test Accuracy: {best_metric:.6f}')
    else:
        print(f'Best Test RMSE: {best_metric:.6f}')
        
    print(f"Model saved to {model_path}")
    
    # Plotting with memory optimization
    print("Generating predictions for plotting...")
    loaded_model = model  # No need to reload, we already have it
    loaded_model.eval()
    
    # Generate predictions in chunks
    chunk_size = batch_size
    predictions = []
    actuals = y_test_tensor.cpu().numpy()[:200]
    
    with torch.no_grad():
        for i in range(0, min(200, len(X_test_tensor)), chunk_size):
            chunk = X_test_tensor[i:min(i+chunk_size, 200)]
            pred = loaded_model(chunk)
            predictions.append(pred.cpu().numpy())
    
    predicted_values = np.concatenate(predictions, axis=0).flatten()[:200]
    print("Predictions generated.")
    
    if scale_data and scaler is not None:
        print("Unscaling predictions and actual values...")
        # Get MinMaxScaler parameters for target column
        target_column_idx = columns_to_include.index(label_column)
        target_min = scaler.data_min_[target_column_idx]
        target_max = scaler.data_max_[target_column_idx]
        
        # Inverse transform using the MinMaxScaler formula: X = X_scaled * (max - min) + min
        original_predictions = predicted_values * (target_max - target_min) + target_min
        original_actuals = actuals * (target_max - target_min) + target_min
        print("Unscaling complete.")
    else:
        # No scaling was applied; use the original values
        print("Skipping unscaling as data was not scaled.")
        original_predictions = predicted_values
        original_actuals = actuals

    # Create plot
    print("Creating plot...")
    trace1 = go.Scatter(x=np.arange(len(original_predictions)), 
                       y=original_predictions, 
                       mode='lines', 
                       name='Predicted Values')
    trace2 = go.Scatter(x=np.arange(len(original_actuals)), 
                       y=original_actuals, 
                       mode='lines', 
                       name='Actual Values')
    
    layout = go.Layout(
        title='Comparison of Predicted and Actual Values',
        xaxis={'title': 'Index'},
        yaxis={'title': 'Values'},
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.show()
    print("Plot displayed.")
    
    # Free memory
    print("Cleaning up...")
    del X_train, y_train, df_clean, train_data, test_data, df_scaled
    gc.collect()
    print("Done.")

    return model, best_rmse




######################
# FUNCTION TO LOAD SAVED MODEL AND ADD PREDICTIONS TO DATAFRAME
######################

def predict_sequences(model, df, columns_to_include, seq_length, batch_size, new_column_name, scaler=None, should_scale=False):
    import torch
    import numpy as np
    import pandas as pd
    
    # Pre-filter data and convert to contiguous array for faster access
    data_values = df[columns_to_include].values.astype(np.float32, copy=False)
    
    if should_scale and scaler is not None:
        data_values = scaler.transform(df[columns_to_include]).astype(np.float32)
        
    data_values = np.ascontiguousarray(data_values)
    
    num_sequences = len(data_values) - seq_length + 1
    if num_sequences <= 0:
        raise ValueError("Sequence length is too long for the data")
    
    # Optimize batch size based on data size
    batch_size = min(batch_size, 2048)  # Prevent excessive memory usage
    
    # Pre-allocate predictions array
    predictions = np.empty(num_sequences, dtype=np.float32)
    seq_indices = np.arange(seq_length)
    
    # Create fixed CUDA tensors
    with torch.no_grad():
        for i in range(0, num_sequences, batch_size):
            end_idx = min(i + batch_size, num_sequences)
            batch_size_actual = end_idx - i
            
            # Optimize indexing operations
            batch_sequences = np.lib.stride_tricks.as_strided(
                data_values[i:end_idx + seq_length - 1],
                shape=(batch_size_actual, seq_length, data_values.shape[1]),
                strides=(data_values.strides[0], data_values.strides[0], data_values.strides[1])
            ).astype(np.float32)
            
            # Optimize tensor transfer
            batch_tensor = torch.from_numpy(batch_sequences).cuda(non_blocking=True)
            batch_predictions = model(batch_tensor)
            predictions[i:end_idx] = batch_predictions.cpu().numpy().flatten()
            
            del batch_tensor, batch_predictions
            
            # Less frequent cleanup
            if i % (batch_size * 20) == 0:
                torch.cuda.empty_cache()

    if should_scale and scaler is not None:
        dummy_array = np.zeros((len(predictions), len(columns_to_include)), dtype=np.float32)
        dummy_df = pd.DataFrame(dummy_array, columns=columns_to_include)
        target_column_idx = columns_to_include.index(columns_to_include[-1])
        dummy_df.iloc[:, target_column_idx] = predictions
        predictions = scaler.inverse_transform(dummy_df)[:, target_column_idx].astype(np.float32)
    
    # Optimize DataFrame assignment
    result = pd.Series([None] * len(df), index=df.index, dtype=np.float32)
    result.iloc[seq_length-1:seq_length-1+len(predictions)] = predictions
    df[new_column_name] = result
    
    return df


######################
# LOAD MODEL / SCALER
######################

def load_model(model_path, input_size, num_features):
    import torch
    try:
        # Load the state dictionary
        state_dict = torch.load(model_path)
        
        # Infer hidden_size
        hidden_size = state_dict['stacks.0.blocks.0.fc1.weight'].shape[0]
        
        # Infer num_stacks and num_blocks_per_stack
        stack_indices = set()
        block_indices_per_stack = {}
        for key in state_dict.keys():
            if key.startswith('stacks.'):
                parts = key.split('.')
                stack_idx = int(parts[1])
                block_idx = int(parts[3])
                stack_indices.add(stack_idx)
                block_indices_per_stack.setdefault(stack_idx, set()).add(block_idx)
        
        num_stacks = max(stack_indices) + 1  # Indices start from 0
        num_blocks_per_stack = max(len(blocks) for blocks in block_indices_per_stack.values())
        
        # Create the model with the provided parameters
        model = NBeatsModel(
            input_size=input_size,
            num_features=num_features,
            num_stacks=num_stacks,
            num_blocks_per_stack=num_blocks_per_stack,
            hidden_size=hidden_size
        )
        
        # Load the state dictionary into the model
        model.load_state_dict(state_dict)
        
        model.eval()  # Set to evaluation mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)  # Move to device
        
        #print(f"Successfully loaded model from {model_path}")
        #print(f"Model architecture: input_size={input_size}, num_features={num_features}, hidden_size={hidden_size}, num_stacks={num_stacks}, num_blocks_per_stack={num_blocks_per_stack}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None



def load_scaler(scaler_path):
    """
    Loads a scikit-learn scaler from the specified path.
    
    Parameters:
    - scaler_path (str): Path to the saved scaler file
    
    Returns:
    - scaler: The loaded scaler object
    """
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Successfully loaded scaler from {scaler_path}")
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None


######################
# TRAIN RNN OR NBEATS
######################



# import torch
# torch.cuda.empty_cache()

# train_and_save_rnn_model(
#     model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_2_raw_15min_t2_TEST.pth',
#     scaler_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_2_raw_15min_t2_scaler_TEST.pth',
#     master_df= df_features_master, #ticker_df_adjusted,
#     columns_to_include=[
#          'close_slope_45_2_raw',
#         'close_slope_35_2_raw',
#         'close_slope_25_2_raw',
#     ],
#     label_column='close_slope_35_2_raw',  # Pass the label column here
#     seq_length=10, #10
#     hidden_dim=50, #50
#     num_layers=1, #1
#     num_epochs=100,
#     learning_rate=0.001,
#     prediction_horizon=2,
#     scale_data=False  
# )


# import torch
# torch.cuda.empty_cache()

# train_and_save_nbeats_model(
#     model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_t5.pth',
#     scaler_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_t5_scaler.pth',
#     master_df= df_features_master, 
#     columns_to_include=[

#         # 'close_CCI_short_raw_hma_7',
#         # 'close_CCI_short_raw_hma_10',
#         # 'close_CCI_short_raw_hma_14',
#         # 'close_CCI_short_raw_hma_18',
#         # 'close_CCI_short_raw_hma_22',
        
#         # 'close_slope_100_2_raw',
#         # 'close_slope_60_2_raw', 
#         #  'close_slope_45_2_raw',
#         'close_slope_35_2_raw',
#         'close_slope_25_2_raw',
#         'close_slope_15_2_raw',
#         'close_slope_25_2_raw_positive',
#         'close_2nd_deriv_25_5_positive',
#         # 'close_slope_10_2_raw',
#        #  'close_slope_5_2_raw',
#        #  'close_slope_4_2_raw',
#        # 'close_2nd_deriv_25_5_raw',
#        #  'close_2nd_deriv_15_5_raw', 
#        # 'close_2nd_deriv_10_5_raw'
        
#     ],
#     label_column='close_slope_25_2_raw',  # Pass the label column here
#     seq_length=10,
#     hidden_dim=100,
#     num_stacks=1, #2            
#     num_blocks_per_stack=3,    
#     num_epochs=15,
#     learning_rate=0.001,
#     prediction_horizon=5,
#     batch_size=1000,
#     scale_data=False,
#     is_binary=0
# )












