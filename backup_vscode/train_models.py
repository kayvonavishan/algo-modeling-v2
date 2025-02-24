# ---------------------------
# EXTREMELY RANDOMIZED TREES WITH BOOSTING (ERTBoost)
# ---------------------------
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.metrics import accuracy_score
import warnings

class ERTBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=50, learning_rate=0.1, subsample=0.8, print_interval=10, X_val=None, y_val=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.print_interval = print_interval
        self.X_val = X_val
        self.y_val = y_val
        self.models = []
        self.model_weights = []
        self.feature_importances_ = None
        self.accuracies = []
        self.classes_ = None
        
    def _ensure_all_classes_present(self, y):
        """Ensure all classes are present in the subsample"""
        unique_classes = np.unique(y)
        if len(unique_classes) != len(self.classes_):
            # Add at least one instance of missing classes
            missing_classes = set(self.classes_) - set(unique_classes)
            additional_samples = np.array(list(missing_classes))
            y = np.concatenate([y, additional_samples])
            return False
        return True
        
    def _boost_proba(self, X):
        """Get weighted predictions from all models with proper class alignment"""
        n_samples = X.shape[0]
        all_preds = np.zeros((n_samples, len(self.classes_)))
        
        try:
            for model, weight in zip(self.models, self.model_weights):
                # Get predictions and align them with global classes
                model_proba = model.predict_proba(X)
                model_classes = model.classes_
                
                # Create aligned probabilities
                aligned_proba = np.zeros((n_samples, len(self.classes_)))
                for idx, cls in enumerate(model_classes):
                    if cls in self.classes_:
                        global_idx = np.where(self.classes_ == cls)[0][0]
                        aligned_proba[:, global_idx] = model_proba[:, idx]
                
                all_preds += weight * aligned_proba
                
        except Exception as e:
            warnings.warn(f"Error in _boost_proba: {str(e)}. Returning current predictions.")
            
        return all_preds

    def fit(self, X, y, sample_weight=None):
        """Fit the ERTBoost model with robust class handling"""
        try:
            # Store global classes
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            self.models = []
            self.model_weights = []
            
            for i in range(1, self.n_estimators + 1):
                try:
                    # Subsample the data
                    n_samples = int(X.shape[0] * self.subsample)
                    indices = np.random.choice(X.shape[0], n_samples, replace=False)
                    X_subset = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
                    y_subset = y[indices]
                    
                    # Ensure all classes are present in subset
                    if not self._ensure_all_classes_present(y_subset):
                        # If classes were added, adjust X_subset accordingly
                        if hasattr(X, 'iloc'):
                            X_subset = X.iloc[np.random.choice(X.shape[0], len(y_subset))]
                        else:
                            X_subset = X[np.random.choice(X.shape[0], len(y_subset))]
                    
                    if sample_weight is not None:
                        subset_weights = sample_weight[indices]
                    else:
                        subset_weights = None

                    # Train Extremely Randomized Trees
                    ert = ExtraTreesClassifier(
                        n_estimators=10,
                        max_features='sqrt',
                        bootstrap=True,
                        n_jobs=-1
                    )
                    ert.fit(X_subset, y_subset, sample_weight=subset_weights)
                    
                    # Verify model classes match global classes
                    if not np.array_equal(np.sort(ert.classes_), np.sort(self.classes_)):
                        warnings.warn(f"Model {i} has different classes. Skipping this iteration.")
                        continue

                    self.models.append(ert)
                    self.model_weights.append(self.learning_rate)

                    # Update feature importances
                    if self.feature_importances_ is None:
                        self.feature_importances_ = ert.feature_importances_
                    else:
                        self.feature_importances_ += ert.feature_importances_

                    # Compute validation accuracy if requested
                    if self.print_interval > 0 and i % self.print_interval == 0:
                        if self.X_val is not None and self.y_val is not None:
                            try:
                                y_pred = self.predict(self.X_val)
                                acc = accuracy_score(self.y_val, y_pred)
                                self.accuracies.append((i, acc))
                            except Exception as e:
                                warnings.warn(f"Error computing validation accuracy: {str(e)}")
                                
                except Exception as e:
                    warnings.warn(f"Error in boosting iteration {i}: {str(e)}")
                    continue
                    
            # Normalize feature importances
            if self.feature_importances_ is not None and self.models:
                self.feature_importances_ /= len(self.models)
                
            if not self.models:
                raise ValueError("No valid models were trained.")
                
        except Exception as e:
            raise ValueError(f"Error during model fitting: {str(e)}")
            
        return self

    def predict_proba(self, X):
        """Predict class probabilities with proper error handling"""
        if not self.models:
            raise ValueError("Model not fitted yet.")
            
        try:
            proba = self._boost_proba(X)
            weight_sum = sum(self.model_weights)
            if weight_sum > 0:
                proba /= weight_sum
            return proba
            
        except Exception as e:
            raise ValueError(f"Error in predict_proba: {str(e)}")

    def predict(self, X):
        """Predict classes with proper error handling"""
        try:
            return np.argmax(self.predict_proba(X), axis=1)
        except Exception as e:
            raise ValueError(f"Error in predict: {str(e)}")



# ---------------------------
# ROTATION FOREST
# ---------------------------
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
import numpy as np
import pandas as pd

class RotationForestClassifier:
    def __init__(self, n_estimators=10, k_subsets=3):
        self.n_estimators = n_estimators
        self.k_subsets = k_subsets
        self.estimators = []
        self.pcas = []
        self.feature_subsets = []
        self.classes_ = None  # Initialize classes_

    def fit(self, X, y, sample_weight=None):
        """
        Fit the Rotation Forest model.
        """
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X

        self.classes_ = np.unique(y)  # Store the unique classes
        n_features = X_np.shape[1]
        subset_size = n_features // self.k_subsets

        for i in range(self.n_estimators):
            # Randomly split features into K subsets
            features_idx = np.random.permutation(n_features)
            current_subsets = [
                features_idx[j:j + subset_size] 
                for j in range(0, len(features_idx), subset_size)
            ]
            self.feature_subsets.append(current_subsets)
            
            # Apply PCA to each subset
            current_pcas = []
            transformed_data = []
            
            for subset in current_subsets:
                pca = PCA()
                transformed_subset = pca.fit_transform(X_np[:, subset])
                transformed_data.append(transformed_subset)
                current_pcas.append(pca)
            
            # Concatenate all transformed subsets
            X_transformed = np.hstack(transformed_data)
            
            # Train a Random Forest classifier on transformed data
            tree = RandomForestClassifier(n_estimators=1, max_features='sqrt', random_state=42)
            if sample_weight is not None:
                tree.fit(X_transformed, y, sample_weight=sample_weight)
            else:
                tree.fit(X_transformed, y)
            
            self.estimators.append(tree)
            self.pcas.append(current_pcas)
            
            if (i + 1) % 10 == 0:
                print(f"Trained {i + 1} Rotation Forest estimators...")
        
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
        
        predictions = []
        
        for i in range(self.n_estimators):
            transformed_data = []
            
            # Transform each subset using stored PCAs
            for subset_idx, subset in enumerate(self.feature_subsets[i]):
                transformed_subset = self.pcas[i][subset_idx].transform(X_np[:, subset])
                transformed_data.append(transformed_subset)
            
            # Concatenate transformed data and get predictions
            X_transformed = np.hstack(transformed_data)
            pred = self.estimators[i].predict(X_transformed)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions)
        final_predictions = mode(predictions, axis=0)[0].ravel()
        return final_predictions

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        """
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
        
        probas = []
        
        for i in range(self.n_estimators):
            transformed_data = []
            
            # Transform each subset using stored PCAs
            for subset_idx, subset in enumerate(self.feature_subsets[i]):
                transformed_subset = self.pcas[i][subset_idx].transform(X_np[:, subset])
                transformed_data.append(transformed_subset)
            
            # Concatenate transformed data and get probability predictions
            X_transformed = np.hstack(transformed_data)
            prob = self.estimators[i].predict_proba(X_transformed)
            probas.append(prob)
        
        # Average probabilities
        mean_proba = np.mean(probas, axis=0)
        
        # Ensure the order of classes matches self.classes_
        if self.classes_ is not None:
            proba_ordered = np.zeros((mean_proba.shape[0], len(self.classes_)))
            for idx, cls in enumerate(self.classes_):
                cls_index = np.where(self.estimators[0].classes_ == cls)[0][0]
                proba_ordered[:, idx] = mean_proba[:, cls_index]
            return proba_ordered
        else:
            return mean_proba

def train_and_save_models(
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    sample_weights,
    main_model_path,
    use_ensemble=True,
    early_stopping_enabled=True,
    early_stopping_rounds=10,
    num_epochs=100,
    num_classes=None,
    device=None
):
    """
    Train and save multiple machine learning models including XGBoost, CatBoost, 
    Random Forest, LightGBM, Neural Network, Extra Trees, and ERTBoost.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test labels
    sample_weights : array-like
        Sample weights for training
    main_model_path : str
        Base path for saving models
    use_ensemble : bool, default=True
        Whether to use ensemble of models
    early_stopping_enabled : bool, default=True
        Whether to use early stopping during training
    early_stopping_rounds : int, default=10
        Number of rounds for early stopping
    num_epochs : int, default=100
        Number of training epochs
    num_classes : int, optional
        Number of classes for classification
    device : str, optional
        Device to use for PyTorch training ('cuda' or 'cpu')
    
    Returns:
    --------
    dict
        Dictionary containing paths to all trained models and their test accuracies
    """
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import joblib
    from sklearn.metrics import accuracy_score, classification_report
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.utils.class_weight import compute_class_weight
    import random
    import seaborn as sns
    import matplotlib.pyplot as plt
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
    
    # Set device for PyTorch
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dictionary to store model accuracies
    model_accuracies = {}
    
    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # XGBoost parameters
    xgb_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': num_classes,
        'max_depth': 5,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1,
        'alpha': 0,
        'min_child_weight': 1,
    }

    # ---------------------------
    # XGBOOST
    # ---------------------------

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
    
    if early_stopping_enabled:
        bst_xgb = xgb.train(
            xgb_params, 
            dtrain, 
            num_boost_round=num_epochs, 
            evals=[(dtrain, 'train'), (dtest, 'eval')], 
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
    else:
        bst_xgb = xgb.train(
            xgb_params, 
            dtrain, 
            num_boost_round=num_epochs, 
            evals=[(dtrain, 'train'), (dtest, 'eval')], 
            verbose_eval=False
        )

    # Get XGBoost test accuracy
    y_test_pred = bst_xgb.predict(dtest)
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)
    xgb_test_accuracy = accuracy_score(y_test, y_test_pred_labels)
    model_accuracies['xgboost'] = xgb_test_accuracy

    # Save XGBoost model
    xgb_model_path = main_model_path.replace('.pkl', '_xgboost.pkl')
    model_info_xgb = {
        'model_type': 'xgboost',
        'model': bst_xgb,
        'params': xgb_params,
        'test_accuracy': xgb_test_accuracy
    }
    joblib.dump(model_info_xgb, xgb_model_path)
    print(f"XGBoost Test Accuracy: {xgb_test_accuracy:.4f}")

    # ---------------------------
    # CATBOOST
    # ---------------------------

    # CatBoost parameters
    cat_params = {
        'iterations': num_epochs,
        'depth': 6,
        'learning_rate': 0.03,
        'rsm': 0.8,
        'l2_leaf_reg': 3.0,
        'min_data_in_leaf': 1,
        'eval_metric': 'MultiClass',
        'loss_function': 'MultiClass'
    }
            
    # Initialize and train CatBoostClassifier
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=(X_test, y_test),
        verbose=False
    )
    
    # Get and print CatBoost test accuracy first
    cat_test_pred = cat_model.predict(X_test)
    cat_test_accuracy = accuracy_score(y_test, cat_test_pred)
    print(f"CatBoost Test Accuracy: {cat_test_accuracy:.4f}")
    
    # Then save CatBoost model with proper naming
    cat_model_path = main_model_path.replace('.pkl', '_catboost.pkl')
    print(f"Saving CatBoost model to: {cat_model_path}")  # Debug print
    model_info_cat = {
        'model_type': 'catboost',
        'model': cat_model,
        'params': cat_params,
        'test_accuracy': cat_test_accuracy  # Now cat_test_accuracy is defined
    }
    joblib.dump(model_info_cat, cat_model_path)


    
    # ---------------------------
    # RANDOM FOREST
    # ---------------------------
    
    from sklearn.ensemble import RandomForestClassifier
    
    # Random Forest parameters (defaults)
    rf_params = {
        'n_estimators': 200,        # Reduce from 300
        'max_depth': 15,            # Keep this limit
        'min_samples_split': 4,     # Slightly reduce
        'min_samples_leaf': 2,      # Slightly reduce
        'max_features': 'sqrt',     # Go back to sqrt if log2 is too slow
        'max_samples': 0.5,         # Reduce from 0.7
        'n_jobs': -1,              # Keep parallel processing
        'random_state': 42
    }
    
    # Initialize and train Random Forest
    rf_model = RandomForestClassifier(**rf_params)
    #rf_model.fit(X_train, y_train, sample_weight=sample_weights)

    # Train incrementally and check accuracy
    check_interval = 10  # Check accuracy every 10 trees
    for n_trees in range(check_interval, rf_params['n_estimators'] + 1, check_interval):
        rf_model.set_params(n_estimators=n_trees)
        rf_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Get accuracy
        rf_test_pred = rf_model.predict(X_test)
        rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
        #print(f"Trees: {n_trees}, Test Accuracy: {rf_test_accuracy:.4f}")
    
    # Save Random Forest model
    rf_model_path = main_model_path.replace('.pkl', '_randomforest.pkl')
    print(f"Saving Random Forest model to: {rf_model_path}")
    model_info_rf = {
        'model_type': 'randomforest',
        'model': rf_model,
        'params': rf_params,
        'test_accuracy': rf_test_accuracy  # Add test accuracy
    }
    joblib.dump(model_info_rf, rf_model_path)
    
    # Get and print Random Forest test accuracy
    rf_test_pred = rf_model.predict(X_test)
    rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
    print(f"Random Forest Test Accuracy: {rf_test_accuracy:.4f}")
    
    # ---------------------------
    # LIGHT GBM
    # ---------------------------
    
    import lightgbm as lgb
    
    # LightGBM parameters (using defaults)
    lgb_params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',          # You can experiment with 'dart' or 'goss'
        'num_leaves': 63,                 # Increased from 31
        'learning_rate': 0.05,            # Lowered from 0.1
        'feature_fraction': 0.8,          # Reduced from 1.0
        'bagging_fraction': 0.8,          # New parameter
        'bagging_freq': 5,                # New parameter
        'max_depth': 15,                  # New parameter
        'min_data_in_leaf': 20,           # Increased from default
        'lambda_l1': 0.1,                 # New regularization
        'lambda_l2': 0.1,                 # New regularization
        'n_jobs': -1,
        'verbose': -1
    }


    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Set up callbacks
    callbacks = []
    if early_stopping_enabled:
        callbacks.append(lgb.early_stopping(early_stopping_rounds))
    
    # Train LightGBM model
    lgb_model = lgb.train(
        params=lgb_params,
        train_set=train_data,
        num_boost_round=250,           # Specify iterations here
        valid_sets=[valid_data],
        valid_names=['valid'],
        callbacks=callbacks
    )
            
    # Get and print LightGBM test accuracy first
    lgb_test_pred = np.argmax(lgb_model.predict(X_test), axis=1)
    lgb_test_accuracy = accuracy_score(y_test, lgb_test_pred)
    print(f"LightGBM Test Accuracy: {lgb_test_accuracy:.4f}")
    
    # Then save LightGBM model
    lgb_model_path = main_model_path.replace('.pkl', '_lightgbm.pkl')
    print(f"Saving LightGBM model to: {lgb_model_path}")
    model_info_lgb = {
        'model_type': 'lightgbm',
        'model': lgb_model,
        'params': lgb_params,
        'test_accuracy': lgb_test_accuracy
    }
    joblib.dump(model_info_lgb, lgb_model_path)
            


    # ---------------------------
    # NEURAL NETWORK
    # ---------------------------
    
    # ---------------------------
    # 1. Setup Reproducibility
    # ---------------------------
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # ---------------------------
    # 3. Calculate Class Weights
    # ---------------------------
    #print("Calculating class weights to address class imbalance...")
    
    # Ensure y_train is a NumPy array for compute_class_weight
    y_train_np = y_train.to_numpy()
    
    # Compute class weights using sklearn's compute_class_weight
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_np),
        y=y_train_np
    )
    
    # Create a dictionary mapping class indices to weights
    classes = np.unique(y_train_np)
    class_weights = {cls: weight for cls, weight in zip(classes, class_weights_array)}
    #print(f"Computed class weights: {class_weights}")
    
    # ---------------------------
    # 4. Convert Data to PyTorch Tensors
    # ---------------------------
    #print("Converting data to PyTorch tensors...")
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    batch_size_nn = 128  # Reasonable batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size_nn, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_nn, shuffle=False, drop_last=True)
    
    # ---------------------------
    # 5. Define the Neural Network Architecture
    # ---------------------------
    # Also modify the training loop to handle potential small batches
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_sizes, num_classes, dropout_p=0.3):
            super(NeuralNet, self).__init__()
            layers = []
            previous_size = input_size
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(previous_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_p))
                previous_size = hidden_size
            layers.append(nn.Linear(previous_size, num_classes))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            if x.size(0) == 1:  # If batch size is 1
                # Switch BatchNorm to eval mode temporarily
                for module in self.modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.eval()
                output = self.network(x)
                # Switch BatchNorm back to training mode
                for module in self.modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.train()
                return output
            return self.network(x)
    
    # ---------------------------
    # 6. Define Loss Function, Optimizer, and Scheduler with Class Weights
    # ---------------------------
    #print("Setting up loss function, optimizer, and learning rate scheduler with class weights...")
    
    # Define device before initializing the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"Using device: {device}")
    
    # Initialize the model with a simpler architecture
    input_size = X_train.shape[1]  # Number of features
    hidden_sizes = [128, 64]        # Reduced number of layers and nodes
    num_classes = len(classes)      # Number of classes
    dropout_p = 0.3                 # Updated dropout probability for regularization
    
    nn_model = NeuralNet(input_size, hidden_sizes, num_classes, dropout_p).to(device)
    #print(f"Neural Network architecture:\n{nn_model}")
    
    # Create a sorted list of class weights based on class indices
    sorted_class_weights = [class_weights.get(i, 1.0) for i in range(num_classes)]
    assert len(sorted_class_weights) == num_classes, "Mismatch between number of classes and class weights."
    
    # Convert class weights to a PyTorch tensor
    class_weights_tensor = torch.tensor(sorted_class_weights, dtype=torch.float32).to(device)
    #print(f"Sorted Class Weights: {sorted_class_weights}")
    
    # Define the loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    #print("Initialized CrossEntropyLoss with class weights.")
    
    # Define the optimizer
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
    #print("Initialized Adam optimizer.")
    
    # Define the learning rate scheduler with more gradual reduction
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=4,        # Wait longer before reducing
        factor=0.5,        # More gradual reduction (was 0.3)
        min_lr=1e-6       # Don't let lr go below this
    )
    #print("Initialized ReduceLROnPlateau scheduler with more gradual reduction")
    
    # ---------------------------
    # 7. Training Loop with Early Stopping and Gradient Clipping
    # ---------------------------
    epochs = 20  # Reasonable number of epochs
    best_test_accuracy = 0.0
    best_model_state = None
    patience = 5  # Early stopping patience
    trigger_times = 0
    
    #print("Starting training loop...")
    
    for epoch in range(1, epochs + 1):
        nn_model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Skip batches that are too small (optional additional safety check)
            if batch_X.size(0) <= 1:
                continue
                
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = nn_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Evaluation on Test Set
        nn_model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = nn_model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        test_accuracy = correct / total
        #print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Classification Report for detailed metrics
        report = classification_report(all_targets, all_preds, digits=4)
        #print(f"Classification Report:\n{report}")
        
        # Step the scheduler based on test accuracy
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_accuracy)
        new_lr = optimizer.param_groups[0]['lr']
        #if new_lr != old_lr:
            #print(f'Learning rate decreased from {old_lr:.6f} to {new_lr:.6f}')
        
        # Check for improvement
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model_state = nn_model.state_dict()
            trigger_times = 0
            #print(f"New best accuracy: {best_test_accuracy:.4f} at epoch {epoch}")
        else:
            trigger_times += 1
            #print(f"No improvement in accuracy for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                #print("Early stopping triggered.")
                break
    
    # Load the best model state
    if best_model_state is not None:
        nn_model.load_state_dict(best_model_state)
        print(f"Loaded best model state with Test Accuracy: {best_test_accuracy:.4f}")
    else:
        print("No improvement observed during training.")
        
    # ---------------------------
    # 8. Save the PyTorch Model and Info
    # ---------------------------
    nn_model_path = main_model_path.replace('.pkl', '_pytorch_nn.pkl')
    nn_model_state_path = main_model_path.replace('.pkl', '_pytorch_nn.pth')
    
    # Save model state
    torch.save(nn_model.state_dict(), nn_model_state_path)
    
    # Save model info including test accuracy and architecture details
    nn_model_info = {
        'model_type': 'pytorch_nn',
        'model_state_path': nn_model_state_path,  # Path to the state dict
        'test_accuracy': best_test_accuracy,
        'params': {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'num_classes': num_classes,
            'dropout_p': dropout_p
        },
        'architecture': str(nn_model)  # Save architecture as string for reference
    }
    
    # Save the model info
    joblib.dump(nn_model_info, nn_model_path)
    print(f"Saved PyTorch Neural Network model state to: {nn_model_state_path}")
    print(f"Saved PyTorch Neural Network info to: {nn_model_path}")

    # ---------------------------
    # 9. Final Test Accuracy
    # ---------------------------
    print(f"Final PyTorch Neural Network Test Accuracy: {best_test_accuracy:.4f}")



    # ---------------------------
    # EXTRA TREES
    # ---------------------------
    from sklearn.ensemble import ExtraTreesClassifier

    # Train Extra Trees Model
    #print("Training Extra Trees Model...")
    
    # Initialize Extra Trees with default hyperparameters
    extra_trees_params = {
        'n_estimators': 200,        # Default number of trees
        'criterion': 'gini',        # Default split criterion
        'max_depth': None,          # No maximum depth
        'min_samples_split': 2,     # Minimum samples to split a node
        'min_samples_leaf': 1,      # Minimum samples at a leaf node
        'max_features': 'sqrt',     # Number of features to consider at each split
        'bootstrap': False,         # Whether bootstrap samples are used when building trees
        'n_jobs': -1,               # Use all available cores
        'random_state': 42          # For reproducibility
    }
    
    # Initialize the Extra Trees Classifier
    extra_trees_model = ExtraTreesClassifier(**extra_trees_params)
    
    # Train the Extra Trees model
    extra_trees_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Get and print Extra Trees test accuracy
    et_test_pred = extra_trees_model.predict(X_test)
    et_test_accuracy = accuracy_score(y_test, et_test_pred)
    print(f"Extra Trees Test Accuracy: {et_test_accuracy:.4f}")

    # Save Extra Trees model
    extra_trees_model_path = main_model_path.replace('.pkl', '_extratrees.pkl')
    print(f"Saving Extra Trees model to: {extra_trees_model_path}")  # Debug print
    model_info_et = {
        'model_type': 'extratrees',
        'model': extra_trees_model,
        'params': extra_trees_params,
        'test_accuracy': et_test_accuracy
    }
    joblib.dump(model_info_et, extra_trees_model_path)
    
            
    # ---------------------------
    # EXTREMELY RANDOMIZED TREES WITH BOOSTING (ERTBoost)
    # ---------------------------

    # Initialize and train ERTBoost with validation data and print_interval=10
    ertboost_params = {
        'n_estimators': 50,        
        'learning_rate': 0.1,      
        'subsample': 0.8,
        'print_interval': 10,       # Print accuracy every 10 boosting rounds
        'X_val': X_test,            # Validation features
        'y_val': y_test,            # Validation labels
    }
    
    ertboost_model = ERTBoost(**ertboost_params)
    #print("Fitting ERTBoost model...")
    ertboost_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Get and print ERTBoost test accuracy
    ert_test_pred = ertboost_model.predict(X_test)
    ert_test_accuracy = accuracy_score(y_test, ert_test_pred)
    print(f"ERTBoost Final Test Accuracy: {ert_test_accuracy:.4f}")
    
    # Save ERTBoost model
    ertboost_model_path = main_model_path.replace('.pkl', '_ertboost.pkl')
    print(f"Saving ERTBoost model to: {ertboost_model_path}")
    model_info_ertboost = {
        'model_type': 'ertboost',
        'model': ertboost_model,
        'params': ertboost_params,
        'test_accuracy': ert_test_accuracy
    }
    joblib.dump(model_info_ertboost, ertboost_model_path)

    
    
    # ---------------------------
    # 7. Update Ensemble Info to Include All Models
    # ---------------------------
    
    # Update final ensemble info to include all models
    print("Saving ensemble information...")
    ensemble_info = {
        'use_ensemble': use_ensemble,
        'xgboost_path': main_model_path.replace('.pkl', '_xgboost.pkl'),
        'catboost_path': main_model_path.replace('.pkl', '_catboost.pkl'),
        'randomforest_path': main_model_path.replace('.pkl', '_randomforest.pkl'),
        'lightgbm_path': main_model_path.replace('.pkl', '_lightgbm.pkl'),
        'extratrees_path': extra_trees_model_path,
        'pytorch_nn_path': nn_model_path,
        'ertboost_path': ertboost_model_path,
    }
    print(f"Saving ensemble info to: {main_model_path}")
    joblib.dump(ensemble_info, main_model_path)
    
    # Create list of models with their information
    models_info = [
        {'model_name': 'XGBoost', 'model': bst_xgb, 'is_xgboost': True},
        {'model_name': 'CatBoost', 'model': cat_model},
        {'model_name': 'RandomForest', 'model': rf_model},
        {'model_name': 'LightGBM', 'model': lgb_model, 'is_lightgbm': True},
        {'model_name': 'ExtraTrees', 'model': extra_trees_model},
        {'model_name': 'PyTorch NN', 'model': nn_model, 'is_pytorch_nn': True},
        {'model_name': 'ERTBoost', 'model': ertboost_model}
    ]
    
    return ensemble_info, models_info, device





# ###############
# OLD MODEL TRAINING CODE
# ################


        # ###############
        # # TRAIN MODELS
        # ################

        # import pandas as pd
        # import numpy as np
        # import xgboost as xgb
        # import joblib
        # from sklearn.metrics import accuracy_score, classification_report
        # from pytorch_tabnet.tab_model import TabNetClassifier
        # import torch
        # from torch import nn, optim
        # from torch.utils.data import DataLoader, TensorDataset
        # import lightgbm as lgb
        # from catboost import CatBoostClassifier
        # from sklearn.ensemble import RandomForestClassifier
        # from sklearn.utils.class_weight import compute_class_weight
        # import random
        # import seaborn as sns
        # import matplotlib.pyplot as plt
            
        # # Convert the datasets to DMatrix for XGBoost
        # dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights, enable_categorical=True)
        # dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        # # XGBoost parameters
        # params = {
        #     'objective': 'multi:softprob',
        #     'eval_metric': 'mlogloss',
        #     'num_class': num_classes,
        #     'max_depth': max_depth,
        #     'learning_rate': learning_rate,
        #     'subsample': subsample,
        #     'colsample_bytree': colsample_bytree,
        #     'lambda': lambda_p,
        #     'alpha': alpha,
        #     'min_child_weight': min_child_weight,
        # }

        # # Train XGBoost model
        # if early_stopping_enabled:
        #     bst_xgb = xgb.train(
        #         params, 
        #         dtrain, 
        #         num_boost_round=num_epochs, 
        #         evals=[(dtrain, 'train'), (dtest, 'eval')], 
        #         early_stopping_rounds=early_stopping_rounds,
        #         verbose_eval=False
        #     )
        # else:
        #     bst_xgb = xgb.train(
        #         params, 
        #         dtrain, 
        #         num_boost_round=num_epochs, 
        #         evals=[(dtrain, 'train'), (dtest, 'eval')], 
        #         verbose_eval=False
        #     )

        # # Get test accuracy
        # y_test_pred = bst_xgb.predict(dtest)
        # y_test_pred_labels = np.argmax(y_test_pred, axis=1)
        # test_accuracy = accuracy_score(y_test, y_test_pred_labels)

        # # Save XGBoost model
        # model_info_xgb = {
        #     'model_type': 'xgboost',
        #     'model': bst_xgb,
        #     'params': params,
        #     'test_accuracy': test_accuracy
        # }

        # # Save model
        # xgb_model_path = main_model_path.replace('.pkl', '_xgboost.pkl')
        # print(f"Saving XGBoost model to: {xgb_model_path}")
        # joblib.dump(model_info_xgb, xgb_model_path)

        # print(f"XGBoost Test Accuracy: {test_accuracy:.4f}")
                    
        

        # # CatBoost parameters
        # cat_params = {
        #     'iterations': num_epochs,
        #     'depth': 6, #max_depth,
        #     'learning_rate': 0.03, #learning_rate,
        #     'rsm': 0.8,              # Default value instead of colsample_bytree
        #     'l2_leaf_reg': 3.0,      # Default value instead of lambda_p
        #     'min_data_in_leaf': 1,   # Default value instead of min_child_weight
        #     'eval_metric': 'MultiClass',
        #     'loss_function': 'MultiClass'
        # }
                
        # # Initialize and train CatBoostClassifier
        # cat_model = CatBoostClassifier(**cat_params)
        # cat_model.fit(
        #     X_train,
        #     y_train,
        #     sample_weight=sample_weights,
        #     eval_set=(X_test, y_test),
        #     verbose=False
        # )
        
        # # Get and print CatBoost test accuracy first
        # cat_test_pred = cat_model.predict(X_test)
        # cat_test_accuracy = accuracy_score(y_test, cat_test_pred)
        # print(f"CatBoost Test Accuracy: {cat_test_accuracy:.4f}")
        
        # # Then save CatBoost model with proper naming
        # cat_model_path = main_model_path.replace('.pkl', '_catboost.pkl')
        # print(f"Saving CatBoost model to: {cat_model_path}")  # Debug print
        # model_info_cat = {
        #     'model_type': 'catboost',
        #     'model': cat_model,
        #     'params': cat_params,
        #     'test_accuracy': cat_test_accuracy  # Now cat_test_accuracy is defined
        # }
        # joblib.dump(model_info_cat, cat_model_path)

        
        
        # # Train Random Forest Model
        # #print("Training Random Forest Model...")
        
        # from sklearn.ensemble import RandomForestClassifier
        
        # # Random Forest parameters (defaults)
        # rf_params = {
        #     'n_estimators': 200,        # Reduce from 300
        #     'max_depth': 15,            # Keep this limit
        #     'min_samples_split': 4,     # Slightly reduce
        #     'min_samples_leaf': 2,      # Slightly reduce
        #     'max_features': 'sqrt',     # Go back to sqrt if log2 is too slow
        #     'max_samples': 0.5,         # Reduce from 0.7
        #     'n_jobs': -1,              # Keep parallel processing
        #     'random_state': 42
        # }
        
        # # Initialize and train Random Forest
        # rf_model = RandomForestClassifier(**rf_params)
        # #rf_model.fit(X_train, y_train, sample_weight=sample_weights)

        # # Train incrementally and check accuracy
        # check_interval = 10  # Check accuracy every 10 trees
        # for n_trees in range(check_interval, rf_params['n_estimators'] + 1, check_interval):
        #     rf_model.set_params(n_estimators=n_trees)
        #     rf_model.fit(X_train, y_train, sample_weight=sample_weights)
            
        #     # Get accuracy
        #     rf_test_pred = rf_model.predict(X_test)
        #     rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
        #     #print(f"Trees: {n_trees}, Test Accuracy: {rf_test_accuracy:.4f}")
        
        # # Save Random Forest model
        # rf_model_path = main_model_path.replace('.pkl', '_randomforest.pkl')
        # print(f"Saving Random Forest model to: {rf_model_path}")
        # model_info_rf = {
        #     'model_type': 'randomforest',
        #     'model': rf_model,
        #     'params': rf_params,
        #     'test_accuracy': rf_test_accuracy  # Add test accuracy
        # }
        # joblib.dump(model_info_rf, rf_model_path)
        
        # # Get and print Random Forest test accuracy
        # rf_test_pred = rf_model.predict(X_test)
        # rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
        # print(f"Random Forest Test Accuracy: {rf_test_accuracy:.4f}")
        
        # # Train LightGBM Model
        # #print("Training LightGBM Model...")
        
        # import lightgbm as lgb
        
        # # LightGBM parameters (using defaults)
        # lgb_params = {
        #     'objective': 'multiclass',
        #     'num_class': num_classes,
        #     'metric': 'multi_logloss',
        #     'boosting_type': 'gbdt',          # You can experiment with 'dart' or 'goss'
        #     'num_leaves': 63,                 # Increased from 31
        #     'learning_rate': 0.05,            # Lowered from 0.1
        #     'feature_fraction': 0.8,          # Reduced from 1.0
        #     'bagging_fraction': 0.8,          # New parameter
        #     'bagging_freq': 5,                # New parameter
        #     'max_depth': 15,                  # New parameter
        #     'min_data_in_leaf': 20,           # Increased from default
        #     'lambda_l1': 0.1,                 # New regularization
        #     'lambda_l2': 0.1,                 # New regularization
        #     'n_jobs': -1,
        #     'verbose': -1
        # }


        
        # # Create LightGBM datasets
        # train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        # valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # # Set up callbacks
        # callbacks = []
        # if early_stopping_enabled:
        #     callbacks.append(lgb.early_stopping(early_stopping_rounds))
        
        # # Train LightGBM model
        # lgb_model = lgb.train(
        #     params=lgb_params,
        #     train_set=train_data,
        #     num_boost_round=250,           # Specify iterations here
        #     valid_sets=[valid_data],
        #     valid_names=['valid'],
        #     callbacks=callbacks
        # )
                
        # # Get and print LightGBM test accuracy first
        # lgb_test_pred = np.argmax(lgb_model.predict(X_test), axis=1)
        # lgb_test_accuracy = accuracy_score(y_test, lgb_test_pred)
        # print(f"LightGBM Test Accuracy: {lgb_test_accuracy:.4f}")
        
        # # Then save LightGBM model
        # lgb_model_path = main_model_path.replace('.pkl', '_lightgbm.pkl')
        # print(f"Saving LightGBM model to: {lgb_model_path}")
        # model_info_lgb = {
        #     'model_type': 'lightgbm',
        #     'model': lgb_model,
        #     'params': lgb_params,
        #     'test_accuracy': lgb_test_accuracy
        # }
        # joblib.dump(model_info_lgb, lgb_model_path)
                


        

        
        # # ---------------------------
        # # 1. Setup Reproducibility
        # # ---------------------------
        # seed = 42
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # random.seed(seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(seed)
        
        # # ---------------------------
        # # 3. Calculate Class Weights
        # # ---------------------------
        # #print("Calculating class weights to address class imbalance...")
        
        # # Ensure y_train is a NumPy array for compute_class_weight
        # y_train_np = y_train.to_numpy()
        
        # # Compute class weights using sklearn's compute_class_weight
        # class_weights_array = compute_class_weight(
        #     class_weight='balanced',
        #     classes=np.unique(y_train_np),
        #     y=y_train_np
        # )
        
        # # Create a dictionary mapping class indices to weights
        # classes = np.unique(y_train_np)
        # class_weights = {cls: weight for cls, weight in zip(classes, class_weights_array)}
        # #print(f"Computed class weights: {class_weights}")
        
        # # ---------------------------
        # # 4. Convert Data to PyTorch Tensors
        # # ---------------------------
        # #print("Converting data to PyTorch tensors...")
        
        # # Convert data to PyTorch tensors
        # X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        # y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        # X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        # y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
        
        # # Create TensorDatasets
        # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # # Create DataLoaders
        # batch_size_nn = 128  # Reasonable batch size
        # train_loader = DataLoader(train_dataset, batch_size=batch_size_nn, shuffle=True, drop_last=True)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size_nn, shuffle=False, drop_last=True)
        
        # # ---------------------------
        # # 5. Define the Neural Network Architecture
        # # ---------------------------
        # # Also modify the training loop to handle potential small batches
        # class NeuralNet(nn.Module):
        #     def __init__(self, input_size, hidden_sizes, num_classes, dropout_p=0.3):
        #         super(NeuralNet, self).__init__()
        #         layers = []
        #         previous_size = input_size
        #         for hidden_size in hidden_sizes:
        #             layers.append(nn.Linear(previous_size, hidden_size))
        #             layers.append(nn.BatchNorm1d(hidden_size))
        #             layers.append(nn.ReLU())
        #             layers.append(nn.Dropout(dropout_p))
        #             previous_size = hidden_size
        #         layers.append(nn.Linear(previous_size, num_classes))
        #         self.network = nn.Sequential(*layers)
            
        #     def forward(self, x):
        #         if x.size(0) == 1:  # If batch size is 1
        #             # Switch BatchNorm to eval mode temporarily
        #             for module in self.modules():
        #                 if isinstance(module, nn.BatchNorm1d):
        #                     module.eval()
        #             output = self.network(x)
        #             # Switch BatchNorm back to training mode
        #             for module in self.modules():
        #                 if isinstance(module, nn.BatchNorm1d):
        #                     module.train()
        #             return output
        #         return self.network(x)
        
        # # ---------------------------
        # # 6. Define Loss Function, Optimizer, and Scheduler with Class Weights
        # # ---------------------------
        # #print("Setting up loss function, optimizer, and learning rate scheduler with class weights...")
        
        # # Define device before initializing the model
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # #print(f"Using device: {device}")
        
        # # Initialize the model with a simpler architecture
        # input_size = X_train.shape[1]  # Number of features
        # hidden_sizes = [128, 64]        # Reduced number of layers and nodes
        # num_classes = len(classes)      # Number of classes
        # dropout_p = 0.3                 # Updated dropout probability for regularization
        
        # nn_model = NeuralNet(input_size, hidden_sizes, num_classes, dropout_p).to(device)
        # #print(f"Neural Network architecture:\n{nn_model}")
        
        # # Create a sorted list of class weights based on class indices
        # sorted_class_weights = [class_weights.get(i, 1.0) for i in range(num_classes)]
        # assert len(sorted_class_weights) == num_classes, "Mismatch between number of classes and class weights."
        
        # # Convert class weights to a PyTorch tensor
        # class_weights_tensor = torch.tensor(sorted_class_weights, dtype=torch.float32).to(device)
        # #print(f"Sorted Class Weights: {sorted_class_weights}")
        
        # # Define the loss function with class weights
        # criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        # #print("Initialized CrossEntropyLoss with class weights.")
        
        # # Define the optimizer
        # optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
        # #print("Initialized Adam optimizer.")
        
        # # Define the learning rate scheduler with more gradual reduction
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='max',
        #     patience=4,        # Wait longer before reducing
        #     factor=0.5,        # More gradual reduction (was 0.3)
        #     min_lr=1e-6       # Don't let lr go below this
        # )
        # #print("Initialized ReduceLROnPlateau scheduler with more gradual reduction")
        
        # # ---------------------------
        # # 7. Training Loop with Early Stopping and Gradient Clipping
        # # ---------------------------
        # epochs = 20  # Reasonable number of epochs
        # best_test_accuracy = 0.0
        # best_model_state = None
        # patience = 5  # Early stopping patience
        # trigger_times = 0
        
        # #print("Starting training loop...")
        
        # for epoch in range(1, epochs + 1):
        #     nn_model.train()
        #     running_loss = 0.0
        #     for batch_X, batch_y in train_loader:
        #         # Skip batches that are too small (optional additional safety check)
        #         if batch_X.size(0) <= 1:
        #             continue
                    
        #         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        #         optimizer.zero_grad()
        #         outputs = nn_model(batch_X)
        #         loss = criterion(outputs, batch_y)
        #         loss.backward()
        #         nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        #         optimizer.step()
        #         running_loss += loss.item() * batch_X.size(0)
            
        #     epoch_loss = running_loss / len(train_loader.dataset)
            
        #     # Evaluation on Test Set
        #     nn_model.eval()
        #     correct = 0
        #     total = 0
        #     all_preds = []
        #     all_targets = []
        #     with torch.no_grad():
        #         for batch_X, batch_y in test_loader:
        #             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        #             outputs = nn_model(batch_X)
        #             _, predicted = torch.max(outputs.data, 1)
        #             total += batch_y.size(0)
        #             correct += (predicted == batch_y).sum().item()
        #             all_preds.extend(predicted.cpu().numpy())
        #             all_targets.extend(batch_y.cpu().numpy())
        #     test_accuracy = correct / total
        #     #print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            
        #     # Classification Report for detailed metrics
        #     report = classification_report(all_targets, all_preds, digits=4)
        #     #print(f"Classification Report:\n{report}")
            
        #     # Step the scheduler based on test accuracy
        #     old_lr = optimizer.param_groups[0]['lr']
        #     scheduler.step(test_accuracy)
        #     new_lr = optimizer.param_groups[0]['lr']
        #     #if new_lr != old_lr:
        #         #print(f'Learning rate decreased from {old_lr:.6f} to {new_lr:.6f}')
            
        #     # Check for improvement
        #     if test_accuracy > best_test_accuracy:
        #         best_test_accuracy = test_accuracy
        #         best_model_state = nn_model.state_dict()
        #         trigger_times = 0
        #         #print(f"New best accuracy: {best_test_accuracy:.4f} at epoch {epoch}")
        #     else:
        #         trigger_times += 1
        #         #print(f"No improvement in accuracy for {trigger_times} epoch(s).")
        #         if trigger_times >= patience:
        #             #print("Early stopping triggered.")
        #             break
        
        # # Load the best model state
        # if best_model_state is not None:
        #     nn_model.load_state_dict(best_model_state)
        #     print(f"Loaded best model state with Test Accuracy: {best_test_accuracy:.4f}")
        # else:
        #     print("No improvement observed during training.")
            
        # # ---------------------------
        # # 8. Save the PyTorch Model and Info
        # # ---------------------------
        # nn_model_path = main_model_path.replace('.pkl', '_pytorch_nn.pkl')
        # nn_model_state_path = main_model_path.replace('.pkl', '_pytorch_nn.pth')
        
        # # Save model state
        # torch.save(nn_model.state_dict(), nn_model_state_path)
        
        # # Save model info including test accuracy and architecture details
        # nn_model_info = {
        #     'model_type': 'pytorch_nn',
        #     'model_state_path': nn_model_state_path,  # Path to the state dict
        #     'test_accuracy': best_test_accuracy,
        #     'params': {
        #         'input_size': input_size,
        #         'hidden_sizes': hidden_sizes,
        #         'num_classes': num_classes,
        #         'dropout_p': dropout_p
        #     },
        #     'architecture': str(nn_model)  # Save architecture as string for reference
        # }
        
        # # Save the model info
        # joblib.dump(nn_model_info, nn_model_path)
        # print(f"Saved PyTorch Neural Network model state to: {nn_model_state_path}")
        # print(f"Saved PyTorch Neural Network info to: {nn_model_path}")
    
        # # ---------------------------
        # # 9. Final Test Accuracy
        # # ---------------------------
        # print(f"Final PyTorch Neural Network Test Accuracy: {best_test_accuracy:.4f}")



        # # ---------------------------
        # # EXTRA TREES
        # # ---------------------------
        # from sklearn.ensemble import ExtraTreesClassifier

        # # Train Extra Trees Model
        # #print("Training Extra Trees Model...")
        
        # # Initialize Extra Trees with default hyperparameters
        # extra_trees_params = {
        #     'n_estimators': 200,        # Default number of trees
        #     'criterion': 'gini',        # Default split criterion
        #     'max_depth': None,          # No maximum depth
        #     'min_samples_split': 2,     # Minimum samples to split a node
        #     'min_samples_leaf': 1,      # Minimum samples at a leaf node
        #     'max_features': 'sqrt',     # Number of features to consider at each split
        #     'bootstrap': False,         # Whether bootstrap samples are used when building trees
        #     'n_jobs': -1,               # Use all available cores
        #     'random_state': 42          # For reproducibility
        # }
        
        # # Initialize the Extra Trees Classifier
        # extra_trees_model = ExtraTreesClassifier(**extra_trees_params)
        
        # # Train the Extra Trees model
        # extra_trees_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # # Get and print Extra Trees test accuracy
        # et_test_pred = extra_trees_model.predict(X_test)
        # et_test_accuracy = accuracy_score(y_test, et_test_pred)
        # print(f"Extra Trees Test Accuracy: {et_test_accuracy:.4f}")

        # # Save Extra Trees model
        # extra_trees_model_path = main_model_path.replace('.pkl', '_extratrees.pkl')
        # print(f"Saving Extra Trees model to: {extra_trees_model_path}")  # Debug print
        # model_info_et = {
        #     'model_type': 'extratrees',
        #     'model': extra_trees_model,
        #     'params': extra_trees_params,
        #     'test_accuracy': et_test_accuracy
        # }
        # joblib.dump(model_info_et, extra_trees_model_path)
      
                
        # # ---------------------------
        # # EXTREMELY RANDOMIZED TREES WITH BOOSTING (ERTBoost)
        # # ---------------------------
    
        # # Initialize and train ERTBoost with validation data and print_interval=10
        # ertboost_params = {
        #     'n_estimators': 50,        
        #     'learning_rate': 0.1,      
        #     'subsample': 0.8,
        #     'print_interval': 10,       # Print accuracy every 10 boosting rounds
        #     'X_val': X_test,            # Validation features
        #     'y_val': y_test,            # Validation labels
        # }
        
        # ertboost_model = ERTBoost(**ertboost_params)
        # #print("Fitting ERTBoost model...")
        # ertboost_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # # Get and print ERTBoost test accuracy
        # ert_test_pred = ertboost_model.predict(X_test)
        # ert_test_accuracy = accuracy_score(y_test, ert_test_pred)
        # print(f"ERTBoost Final Test Accuracy: {ert_test_accuracy:.4f}")
        
        # # Save ERTBoost model
        # ertboost_model_path = main_model_path.replace('.pkl', '_ertboost.pkl')
        # print(f"Saving ERTBoost model to: {ertboost_model_path}")
        # model_info_ertboost = {
        #     'model_type': 'ertboost',
        #     'model': ertboost_model,
        #     'params': ertboost_params,
        #     'test_accuracy': ert_test_accuracy
        # }
        # joblib.dump(model_info_ertboost, ertboost_model_path)

        
        
        # # ---------------------------
        # # 7. Update Ensemble Info to Include All Models
        # # ---------------------------
        
        
        # # Update final ensemble info to include Extra Trees
        # print("Saving ensemble information...")
        # ensemble_info = {
        #     'use_ensemble': use_ensemble,
        #     'xgboost_path': main_model_path.replace('.pkl', '_xgboost.pkl'),
        #     'catboost_path': main_model_path.replace('.pkl', '_catboost.pkl'),
        #     'randomforest_path': main_model_path.replace('.pkl', '_randomforest.pkl'),
        #     'lightgbm_path': main_model_path.replace('.pkl', '_lightgbm.pkl'),
        #     'extratrees_path': extra_trees_model_path,
        #     'pytorch_nn_path': nn_model_path,
        #     'ertboost_path': ertboost_model_path,
        # }
        # print(f"Saving ensemble info to: {main_model_path}")
        # joblib.dump(ensemble_info, main_model_path)
    

