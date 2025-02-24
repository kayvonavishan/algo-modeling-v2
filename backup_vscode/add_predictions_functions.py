import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# Make predictions in batches 
# ==========================
def process_predictions_in_batches(df, batch_size=100000, model_path='my_model_info.pkl', debug=False, include_confusion_matrix=True, time_period=None,use_ensemble=True):
    """
    Process predictions in batches using the already scaled features.
    Handles both single model and ensemble predictions.
    """
    import joblib
    import xgboost as xgb
    import numpy as np
    import pandas as pd
    import gc
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from tqdm import tqdm

    # assertion
    assert df.index.is_monotonic_increasing, "DataFrame index must be monotonically increasing in time. Please sort your data by timestamp before prediction."
    
    # Define columns that should be dropped before prediction
    columns_to_drop = ['timestamp', 'close', 'high', 'low', 'close_raw', 'high_raw', 'low_raw', 'label', 'symbol']

    # Store labels separately before dropping any columns
    labels = df['label'].values
    
    # Create a copy of the input dataframe and drop specified columns
    df_features_prepared = df.copy()
    df_features_prepared = df_features_prepared.drop(columns=columns_to_drop, errors='ignore')

    # ADD THESE CHECKS HERE
    assert len(df) == len(df_features_prepared), "Length mismatch between original and prepared DataFrames"
    assert (df.index == df_features_prepared.index).all(), "Index mismatch between original and prepared DataFrames"
    
    # Load the ensemble/model info
    ensemble_info = joblib.load(model_path)

    # Check if we're using ensemble
    if use_ensemble:

        # ==========================
        # Load Models 
        # ==========================
        
        # Print model paths including LightGBM
        # print(f"Loading XGBoost model from: {ensemble_info['xgboost_path']}")
        # print(f"Loading CatBoost model from: {ensemble_info['catboost_path']}")
        # print(f"Loading Random Forest model from: {ensemble_info['randomforest_path']}")
        # print(f"Loading Extra Trees model from: {ensemble_info['extratrees_path']}")
        # print(f"Loading LightGBM model from: {ensemble_info['lightgbm_path']}")

        # Load all models including LightGBM
        model_info_xgb = joblib.load(ensemble_info['xgboost_path'])
        model_info_cat = joblib.load(ensemble_info['catboost_path'])
        model_info_rf = joblib.load(ensemble_info['randomforest_path'])
        model_info_et = joblib.load(ensemble_info['extratrees_path'])
        model_info_lgb = joblib.load(ensemble_info['lightgbm_path'])

        bst_xgb = model_info_xgb['model']
        bst_cat = model_info_cat['model']
        bst_rf = model_info_rf['model']
        bst_et = model_info_et['model']
        bst_lgb = model_info_lgb['model']

        # ==========================
        # Data / Feature Prep
        # ==========================
        
        # Get Random Forest feature names
        if 'feature_names' in model_info_rf:
            rf_feature_names = model_info_rf['feature_names']
        else:
            # If feature names not stored, try to get them from the model
            try:
                rf_feature_names = bst_rf.feature_names_in_.tolist()
            except AttributeError:
                print("Warning: Could not find feature names in Random Forest model")
                rf_feature_names = df_features_prepared.columns.tolist()

        # Verify and align features for Random Forest
        available_features = set(df_features_prepared.columns)
        rf_features_set = set(rf_feature_names)
        
        # Check for missing required features
        missing_rf_features = rf_features_set - available_features
        if missing_rf_features:
            raise ValueError(f"Missing required features for Random Forest: {missing_rf_features}")
        
        # Check for extra features that weren't in training
        extra_features = available_features - rf_features_set
        if extra_features:
            print(f"Warning: Extra features found that weren't in training: {extra_features}")
            # Only keep the features that were used in training
            df_features_prepared = df_features_prepared[rf_feature_names]

        # Get Extra Trees feature names
        if 'feature_names' in model_info_et:
            et_feature_names = model_info_et['feature_names']
        else:
            try:
                et_feature_names = bst_et.feature_names_in_.tolist()
            except AttributeError:
                print("Warning: Could not find feature names in Extra Trees model")
                et_feature_names = df_features_prepared.columns.tolist()
        
        # Verify and align features for Extra Trees
        available_features = set(df_features_prepared.columns)
        et_features_set = set(et_feature_names)
        
        # Check for missing required features
        missing_et_features = et_features_set - available_features
        if missing_et_features:
            raise ValueError(f"Missing required features for Extra Trees: {missing_et_features}")
        
        # Check for extra features that weren't in training
        extra_features = available_features - et_features_set
        if extra_features:
            print(f"Warning: Extra features found that weren't in Extra Trees training: {extra_features}")

        # print("XGBoost model keys:", model_info_xgb.keys())
        # print("CatBoost model keys:", model_info_cat.keys())
        # print("Random Forest model keys:", model_info_rf.keys())

        # Verify XGBoost features match training data
        required_features_xgb = set(bst_xgb.feature_names)
        available_features = set(df_features_prepared.columns)
        missing_features_xgb = required_features_xgb - available_features
        if missing_features_xgb:
            raise ValueError(f"Missing features for XGBoost: {missing_features_xgb}")

        # Verify CatBoost features match training data
        required_features_cat = set(bst_cat.feature_names_)
        missing_features_cat = required_features_cat - available_features
        if missing_features_cat:
            raise ValueError(f"Missing features for CatBoost: {missing_features_cat}")

        # Get number of classes from XGBoost (assuming all models have the same number of classes)
        X_sample_xgb = df_features_prepared.iloc[:1][bst_xgb.feature_names]
        dbatch_sample = xgb.DMatrix(X_sample_xgb, feature_names=bst_xgb.feature_names, enable_categorical=True)
        sample_pred = bst_xgb.predict(dbatch_sample)
        if isinstance(sample_pred, list):
            sample_pred = np.array(sample_pred)
        n_classes = sample_pred.shape[1]

        # ==========================
        # Initialize arrays for all models
        # ==========================
        
        n_samples = len(df_features_prepared)
        pred_probs_xgb = np.zeros((n_samples, n_classes))
        pred_labels_xgb = np.zeros(n_samples, dtype=int)
        pred_probs_cat = np.zeros((n_samples, n_classes))
        pred_labels_cat = np.zeros(n_samples, dtype=int)
        pred_probs_rf = np.zeros((n_samples, n_classes))
        pred_labels_rf = np.zeros(n_samples, dtype=int)
        pred_probs_et = np.zeros((n_samples, n_classes))
        pred_labels_et = np.zeros(n_samples, dtype=int)
        pred_probs_lgb = np.zeros((n_samples, n_classes))
        pred_labels_lgb = np.zeros(n_samples, dtype=int)

        # ==========================
        # Process XGBoost predictions in batches
        # ==========================
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc="Processing XGBoost batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            # Extract batch using only required features
            X_batch = df_features_prepared.iloc[start_idx:end_idx][bst_xgb.feature_names]
            
            # Create DMatrix with same settings as training
            dbatch = xgb.DMatrix(X_batch, feature_names=bst_xgb.feature_names, enable_categorical=True)
            
            # Get predictions
            batch_pred_probs = bst_xgb.predict(dbatch)
            
            # Handle different prediction outputs
            if isinstance(batch_pred_probs, list):
                batch_pred_probs = np.array(batch_pred_probs)
            
            # Store predictions
            pred_probs_xgb[start_idx:end_idx] = batch_pred_probs
            pred_labels_xgb[start_idx:end_idx] = np.argmax(batch_pred_probs, axis=1)
            
            # Clean up
            del X_batch, dbatch, batch_pred_probs
            gc.collect()

        # ==========================
        # Process catboost predictions in batches
        # ==========================
        for i in tqdm(range(n_batches), desc="Processing CatBoost batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            # Extract batch using only required features
            X_batch = df_features_prepared.iloc[start_idx:end_idx][bst_cat.feature_names_]
            
            # Get predictions
            batch_pred_probs = bst_cat.predict_proba(X_batch)
            
            # Store predictions
            pred_probs_cat[start_idx:end_idx] = batch_pred_probs
            pred_labels_cat[start_idx:end_idx] = np.argmax(batch_pred_probs, axis=1)
            
            # Clean up
            del X_batch, batch_pred_probs
            gc.collect()

        # ==========================
        # Process Random Forest predictions in batches
        # ==========================
        for i in tqdm(range(n_batches), desc="Processing Random Forest batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            # Extract batch using ONLY the stored feature names from training
            X_batch = df_features_prepared.iloc[start_idx:end_idx]
            # Filter to only include columns that were present during training
            X_batch = X_batch[rf_feature_names]
            
            # Get predictions
            batch_pred_probs = bst_rf.predict_proba(X_batch)
            
            # Store predictions
            pred_probs_rf[start_idx:end_idx] = batch_pred_probs
            pred_labels_rf[start_idx:end_idx] = np.argmax(batch_pred_probs, axis=1)
            
            # Clean up
            del X_batch, batch_pred_probs
            gc.collect()

        # ==========================
        # Process Extra Trees predictions in batches
        # ==========================
        for i in tqdm(range(n_batches), desc="Processing Extra Trees batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            # Extract batch using stored feature names
            X_batch = df_features_prepared.iloc[start_idx:end_idx][et_feature_names]
            
            # Get predictions
            batch_pred_probs = bst_et.predict_proba(X_batch)
            
            # Store predictions
            pred_probs_et[start_idx:end_idx] = batch_pred_probs
            pred_labels_et[start_idx:end_idx] = np.argmax(batch_pred_probs, axis=1)
            
            # Clean up
            del X_batch, batch_pred_probs
            gc.collect()

        # ==========================
        # Process LightGBM predictions in batches
        # ==========================
        for i in tqdm(range(n_batches), desc="Processing LightGBM batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            # Extract batch using prepared features
            X_batch = df_features_prepared.iloc[start_idx:end_idx]
            
            # Get predictions
            batch_pred_probs = bst_lgb.predict(X_batch)
            
            # Store predictions
            pred_probs_lgb[start_idx:end_idx] = batch_pred_probs
            pred_labels_lgb[start_idx:end_idx] = np.argmax(batch_pred_probs, axis=1)
            
            # Clean up
            del X_batch, batch_pred_probs
            gc.collect()


        # ==========================
        # Ensemble Predictions 
        # ==========================
                    
        # Load test accuracies from model info including LightGBM
        xgb_accuracy = model_info_xgb['test_accuracy']
        cat_accuracy = model_info_cat['test_accuracy']
        rf_accuracy = model_info_rf['test_accuracy']
        et_accuracy = model_info_et['test_accuracy']
        lgb_accuracy = model_info_lgb['test_accuracy']
        
        # Calculate normalized weights including LightGBM
        total_accuracy = xgb_accuracy + cat_accuracy + rf_accuracy + et_accuracy + lgb_accuracy
        xgb_weight = xgb_accuracy / total_accuracy
        cat_weight = cat_accuracy / total_accuracy
        rf_weight = rf_accuracy / total_accuracy
        et_weight = et_accuracy / total_accuracy
        lgb_weight = lgb_accuracy / total_accuracy
        
        # Debug weights if needed
        if debug:
            print("\nModel Weights based on Test Accuracy:")
            print(f"XGBoost Weight: {xgb_weight:.4f} (Accuracy: {xgb_accuracy:.4f})")
            print(f"CatBoost Weight: {cat_weight:.4f} (Accuracy: {cat_accuracy:.4f})")
            print(f"Random Forest Weight: {rf_weight:.4f} (Accuracy: {rf_accuracy:.4f})")
            print(f"Extra Trees Weight: {et_weight:.4f} (Accuracy: {et_accuracy:.4f})")
            print(f"LightGBM Weight: {lgb_weight:.4f} (Accuracy: {lgb_accuracy:.4f})")
            print("-" * 50)
        
        # Modify ensemble predictions calculation to use weighted average
        cm = None
        fig_cm = None
        pred_probs_ensemble = (
            pred_probs_xgb * xgb_weight + 
            pred_probs_cat * cat_weight + 
            pred_probs_rf * rf_weight + 
            pred_probs_et * et_weight +
            pred_probs_lgb * lgb_weight
        )
        pred_labels_ensemble = np.argmax(pred_probs_ensemble, axis=1)

        # Add our new confusion matrix code right here, before the debug prints
        # 2. Build the confusion matrix figure (optional)
        from sklearn.metrics import confusion_matrix, accuracy_score
        if include_confusion_matrix:

            # 1) Check the percentage of invalid labels
            invalid_mask = np.isin(labels, [-100, -999])
            pct_invalid = 100.0 * np.sum(invalid_mask) / len(labels)
            print(f"[DEBUG] Percentage of -100/-999 labels: {pct_invalid:.2f}%")

            # If you want to enforce no more than 5%:
            if pct_invalid > 10.0:
                raise ValueError(
                    f"Too many invalid labels (-100/-999): {pct_invalid:.2f}%. "
                    f"Exceeded the 5% threshold."
                )
            
            # 2) Create a valid_mask excluding -100/-999
            valid_mask = ~invalid_mask
            y_true = labels[valid_mask]
            y_pred = pred_labels_ensemble[valid_mask]

            cm = confusion_matrix(y_true, y_pred)

            # 3) Compute accuracy on filtered data
            ensemble_accuracy = accuracy_score(y_true, y_pred)

            # ---------------------------
            # ADD DEBUG PRINTS HERE
            # ---------------------------
            print("\n[DEBUG] Ensemble Confusion Matrix:")
            print("[DEBUG] Confusion Matrix shape:", cm.shape)
            print("[DEBUG] Unique actual labels (labels):", np.unique(labels))
            print("[DEBUG] Unique predicted labels (pred_labels_ensemble):", np.unique(pred_labels_ensemble))

            # Optional: check for -100 or -999
            if -100 in labels or -999 in labels:
                print("[DEBUG] WARNING: Found -100 or -999 in the actual labels!")


            fig_cm, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=np.unique(y_pred),  # or range(cm.shape[1])
                yticklabels=np.unique(y_true),  # or range(cm.shape[0])
                ax=ax
            )
            time_period_str = f" - {time_period}" if time_period else ""
            ax.set_title(f"Ensemble Confusion Matrix (Acc={ensemble_accuracy*100:.2f}%){time_period_str}")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            plt.tight_layout()
        
        # Debug prints for first 5 rows
        if debug:
            print("\nPrediction Comparison for First 5 Rows:")
            print("-" * 50)
            print(f"Model Test Accuracies:")
            print(f"XGBoost: {xgb_accuracy:.4f}")
            print(f"CatBoost: {cat_accuracy:.4f}")
            print(f"Random Forest: {rf_accuracy:.4f}")
            print(f"Extra Trees: {et_accuracy:.4f}")
            print(f"LightGBM: {lgb_accuracy:.4f}")
            print("-" * 50)
            
            for row in range(min(5, n_samples)):
                actual_label = df_features_prepared.iloc[row]['label']
                print(f"\nRow {row + 1} (Actual Label: {actual_label}):")
                
                # XGBoost - sort predictions by probability
                xgb_probs = {f"class_{i}": prob for i, prob in enumerate(pred_probs_xgb[row])}
                sorted_xgb = dict(sorted(xgb_probs.items(), key=lambda x: x[1], reverse=True))
                print(f"XGBoost predictions (weight: {xgb_weight:.4f}, accuracy: {xgb_accuracy:.4f}):", 
                    {k: f"{v:.4f}" for k, v in sorted_xgb.items()})
                
                # CatBoost - sort predictions by probability
                cat_probs = {f"class_{i}": prob for i, prob in enumerate(pred_probs_cat[row])}
                sorted_cat = dict(sorted(cat_probs.items(), key=lambda x: x[1], reverse=True))
                print(f"CatBoost predictions (weight: {cat_weight:.4f}, accuracy: {cat_accuracy:.4f}):", 
                    {k: f"{v:.4f}" for k, v in sorted_cat.items()})
                
                # Random Forest - sort predictions by probability
                rf_probs = {f"class_{i}": prob for i, prob in enumerate(pred_probs_rf[row])}
                sorted_rf = dict(sorted(rf_probs.items(), key=lambda x: x[1], reverse=True))
                print(f"Random Forest predictions (weight: {rf_weight:.4f}, accuracy: {rf_accuracy:.4f}):", 
                    {k: f"{v:.4f}" for k, v in sorted_rf.items()})
                
                # Extra Trees - sort predictions by probability
                et_probs = {f"class_{i}": prob for i, prob in enumerate(pred_probs_et[row])}
                sorted_et = dict(sorted(et_probs.items(), key=lambda x: x[1], reverse=True))
                print(f"Extra Trees predictions (weight: {et_weight:.4f}, accuracy: {et_accuracy:.4f}):", 
                    {k: f"{v:.4f}" for k, v in sorted_et.items()})
                
                # LightGBM - sort predictions by probability
                lgb_probs = {f"class_{i}": prob for i, prob in enumerate(pred_probs_lgb[row])}
                sorted_lgb = dict(sorted(lgb_probs.items(), key=lambda x: x[1], reverse=True))
                print(f"LightGBM predictions (weight: {lgb_weight:.4f}, accuracy: {lgb_accuracy:.4f}):", 
                    {k: f"{v:.4f}" for k, v in sorted_lgb.items()})
                
                # Show weighted calculation for each class
                print("\nWeighted average calculation for each class:")
                for class_idx in range(len(pred_probs_xgb[row])):
                    weighted_calc = (
                        f"Class_{class_idx} = "
                        f"({pred_probs_xgb[row][class_idx]:.4f} × {xgb_weight:.4f}) + "
                        f"({pred_probs_cat[row][class_idx]:.4f} × {cat_weight:.4f}) + "
                        f"({pred_probs_rf[row][class_idx]:.4f} × {rf_weight:.4f}) + "
                        f"({pred_probs_et[row][class_idx]:.4f} × {et_weight:.4f}) + "
                        f"({pred_probs_lgb[row][class_idx]:.4f} × {lgb_weight:.4f}) = "
                        f"{pred_probs_ensemble[row][class_idx]:.4f}"
                    )
                    print(weighted_calc)
                
                # Ensemble - sort predictions by probability
                ensemble_probs = {f"class_{i}": prob for i, prob in enumerate(pred_probs_ensemble[row])}
                sorted_ensemble = dict(sorted(ensemble_probs.items(), key=lambda x: x[1], reverse=True))
                print("\nWeighted Ensemble predictions:", {k: f"{v:.4f}" for k, v in sorted_ensemble.items()})
                
                print(f"Final predictions - XGB: {pred_labels_xgb[row]}, CatBoost: {pred_labels_cat[row]}, "
                    f"RF: {pred_labels_rf[row]}, ET: {pred_labels_et[row]}, "
                    f"LGB: {pred_labels_lgb[row]}, Ensemble: {pred_labels_ensemble[row]}")
            print("-" * 50)
        
        # Add predictions to original dataframe to preserve all columns
        df_predictions = df.copy()
        
        # Add model weights to the dataframe for reference
        df_predictions['xgb_weight'] = xgb_weight
        df_predictions['catboost_weight'] = cat_weight
        df_predictions['rf_weight'] = rf_weight
        df_predictions['et_weight'] = et_weight
        df_predictions['lgb_weight'] = lgb_weight
        
        # Add XGBoost predictions
        for i in range(n_classes):
            df_predictions[f'xgb_prediction_raw_class_{i}'] = pred_probs_xgb[:, i]
        df_predictions['xgb_predicted_label'] = pred_labels_xgb
        
        # Add CatBoost predictions
        for i in range(n_classes):
            df_predictions[f'catboost_prediction_raw_class_{i}'] = pred_probs_cat[:, i]
        df_predictions['catboost_predicted_label'] = pred_labels_cat
        
        # Add Random Forest predictions
        for i in range(n_classes):
            df_predictions[f'rf_prediction_raw_class_{i}'] = pred_probs_rf[:, i]
        df_predictions['rf_predicted_label'] = pred_labels_rf
        
        # Add Extra Trees predictions
        for i in range(n_classes):
            df_predictions[f'et_prediction_raw_class_{i}'] = pred_probs_et[:, i]
        df_predictions['et_predicted_label'] = pred_labels_et
        
        # Add LightGBM predictions
        for i in range(n_classes):
            df_predictions[f'lgb_prediction_raw_class_{i}'] = pred_probs_lgb[:, i]
        df_predictions['lgb_predicted_label'] = pred_labels_lgb
        
        # Add final weighted ensemble predictions
        for i in range(n_classes):
            df_predictions[f'prediction_raw_class_{i}'] = pred_probs_ensemble[:, i]
        df_predictions['predicted_label'] = pred_labels_ensemble

    else:

        # First load all models
        model_info_xgb = joblib.load(ensemble_info['xgboost_path'])
        model_info_cat = joblib.load(ensemble_info['catboost_path'])
        model_info_rf = joblib.load(ensemble_info['randomforest_path'])
        model_info_et = joblib.load(ensemble_info['extratrees_path'])
        model_info_lgb = joblib.load(ensemble_info['lightgbm_path'])

        # Get test accuracies from all models
        xgb_accuracy = model_info_xgb['test_accuracy']
        cat_accuracy = model_info_cat['test_accuracy']
        rf_accuracy = model_info_rf['test_accuracy']
        et_accuracy = model_info_et['test_accuracy']
        lgb_accuracy = model_info_lgb['test_accuracy']
        
        # Extract models
        bst_xgb = model_info_xgb['model']
        bst_cat = model_info_cat['model']
        bst_rf = model_info_rf['model']
        bst_et = model_info_et['model']
        bst_lgb = model_info_lgb['model']
        
        # Create a dictionary mapping accuracies to models and their info
        model_accuracies = {
            xgb_accuracy: ('xgboost', model_info_xgb, bst_xgb),
            cat_accuracy: ('catboost', model_info_cat, bst_cat),
            rf_accuracy: ('randomforest', model_info_rf, bst_rf),
            et_accuracy: ('extratrees', model_info_et, bst_et),
            lgb_accuracy: ('lightgbm', model_info_lgb, bst_lgb)
        }
        
        # Find the best performing model
        best_accuracy = max(model_accuracies.keys())
        best_model_type, best_model_info, best_model = model_accuracies[best_accuracy]
        
        if debug:
            print(f"\nSelected best performing model: {best_model_type}")
            print(f"Test accuracy: {best_accuracy:.4f}")
            print("-" * 50)
        
        # Handle different model types for prediction
        if best_model_type == 'xgboost':
            # Verify features match training data
            required_features = set(best_model.feature_names)
            available_features = set(df_features_prepared.columns)
            missing_features = required_features - available_features
            if missing_features:
                raise ValueError(f"Missing features for XGBoost: {missing_features}")
            
            # Get number of classes from sample prediction
            X_sample = df_features_prepared.iloc[:1][best_model.feature_names]
            dbatch_sample = xgb.DMatrix(X_sample, feature_names=best_model.feature_names, enable_categorical=True)
            sample_pred = best_model.predict(dbatch_sample)
            if isinstance(sample_pred, list):
                sample_pred = np.array(sample_pred)
            n_classes = sample_pred.shape[1]
            
            # Initialize arrays
            n_samples = len(df_features_prepared)
            pred_probs = np.zeros((n_samples, n_classes))
            pred_labels = np.zeros(n_samples, dtype=int)
            
            # Process in batches
            n_batches = (n_samples + batch_size - 1) // batch_size
            for i in tqdm(range(n_batches), desc=f"Processing {best_model_type} batches"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = df_features_prepared.iloc[start_idx:end_idx][best_model.feature_names]
                dbatch = xgb.DMatrix(X_batch, feature_names=best_model.feature_names, enable_categorical=True)
                batch_pred_probs = best_model.predict(dbatch)
                
                if isinstance(batch_pred_probs, list):
                    batch_pred_probs = np.array(batch_pred_probs)
                
                pred_probs[start_idx:end_idx] = batch_pred_probs
                pred_labels[start_idx:end_idx] = np.argmax(batch_pred_probs, axis=1)
                
                del X_batch, dbatch, batch_pred_probs
                gc.collect()
                
        elif best_model_type == 'catboost':
            # Verify features match training data
            required_features = set(best_model.feature_names_)
            available_features = set(df_features_prepared.columns)
            missing_features = required_features - available_features
            if missing_features:
                raise ValueError(f"Missing features for CatBoost: {missing_features}")
            
            # Get number of classes
            X_sample = df_features_prepared.iloc[:1][best_model.feature_names_]
            sample_pred = best_model.predict_proba(X_sample)
            n_classes = sample_pred.shape[1]
            
            # Initialize arrays
            n_samples = len(df_features_prepared)
            pred_probs = np.zeros((n_samples, n_classes))
            pred_labels = np.zeros(n_samples, dtype=int)
            
            # Process in batches
            n_batches = (n_samples + batch_size - 1) // batch_size
            for i in tqdm(range(n_batches), desc=f"Processing {best_model_type} batches"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = df_features_prepared.iloc[start_idx:end_idx][best_model.feature_names_]
                batch_pred_probs = best_model.predict_proba(X_batch)
                
                pred_probs[start_idx:end_idx] = batch_pred_probs
                pred_labels[start_idx:end_idx] = np.argmax(batch_pred_probs, axis=1)
                
                del X_batch, batch_pred_probs
                gc.collect()
                
        elif best_model_type in ['randomforest', 'extratrees']:
            # Get feature names
            if 'feature_names' in best_model_info:
                feature_names = best_model_info['feature_names']
            else:
                try:
                    feature_names = best_model.feature_names_in_.tolist()
                except AttributeError:
                    print(f"Warning: Could not find feature names in {best_model_type} model")
                    feature_names = df_features_prepared.columns.tolist()
            
            # Verify features
            available_features = set(df_features_prepared.columns)
            features_set = set(feature_names)
            missing_features = features_set - available_features
            if missing_features:
                raise ValueError(f"Missing features for {best_model_type}: {missing_features}")
            
            # Get number of classes
            X_sample = df_features_prepared.iloc[:1][feature_names]
            sample_pred = best_model.predict_proba(X_sample)
            n_classes = sample_pred.shape[1]
            
            # Initialize arrays
            n_samples = len(df_features_prepared)
            pred_probs = np.zeros((n_samples, n_classes))
            pred_labels = np.zeros(n_samples, dtype=int)
            
            # Process in batches
            n_batches = (n_samples + batch_size - 1) // batch_size
            for i in tqdm(range(n_batches), desc=f"Processing {best_model_type} batches"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = df_features_prepared.iloc[start_idx:end_idx][feature_names]
                batch_pred_probs = best_model.predict_proba(X_batch)
                
                pred_probs[start_idx:end_idx] = batch_pred_probs
                pred_labels[start_idx:end_idx] = np.argmax(batch_pred_probs, axis=1)
                
                del X_batch, batch_pred_probs
                gc.collect()
                
        elif best_model_type == 'lightgbm':
            # Get number of classes
            X_sample = df_features_prepared.iloc[:1]
            sample_pred = best_model.predict(X_sample)
            n_classes = sample_pred.shape[1]
            
            # Initialize arrays
            n_samples = len(df_features_prepared)
            pred_probs = np.zeros((n_samples, n_classes))
            pred_labels = np.zeros(n_samples, dtype=int)
            
            # Process in batches
            n_batches = (n_samples + batch_size - 1) // batch_size
            for i in tqdm(range(n_batches), desc=f"Processing {best_model_type} batches"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = df_features_prepared.iloc[start_idx:end_idx]
                batch_pred_probs = best_model.predict(X_batch)
                
                pred_probs[start_idx:end_idx] = batch_pred_probs
                pred_labels[start_idx:end_idx] = np.argmax(batch_pred_probs, axis=1)
                
                del X_batch, batch_pred_probs
                gc.collect()
        
        else:
            raise ValueError(f"Unsupported model type: {best_model_type}")
        
        # Add predictions to original dataframe
        df_predictions = df.copy()
        
        # Add model info
        df_predictions['best_model_type'] = best_model_type
        df_predictions['best_model_accuracy'] = best_accuracy
        
        # Add predictions
        for i in range(n_classes):
            df_predictions[f'prediction_raw_class_{i}'] = pred_probs[:, i]
        df_predictions['predicted_label'] = pred_labels

    return df_predictions, fig_cm
