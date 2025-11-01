import json
import os
import pickle
import random
import time
from datetime import datetime
from itertools import combinations
from math import comb
from typing import List, Set

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import validate_data


# As described in: https://scikit-learn.org/stable/developers/develop.html
# Also: https://scikit-learn.org/dev/modules/generated/sklearn.utils.validation.check_is_fitted.html
class SubclassClassifier(ClassifierMixin, BaseEstimator):
    """
    A scikit-learn style classifier implementing the Kudo et al. subclass method.
    """
    
    def __init__(self, U=4, V=10, K=1, s=100, verbose=False, save_results=False, results_dir='results', random_state=None):
        """
        Initialize the SubclassClassifier.
        
        Parameters:
        -----------
        U : int, default=4
            Number of time thresholds for binning
        V : int, default=20
            Number of feature thresholds per dimension for binning
        K : int, default=1
            Minimum number of times curve must pass through rectangle
        s : int, default=500
            Number of iterations for randomized subclass method
        verbose : bool, default=True
            Whether to print progress information
        save_results : bool, default=False
            Whether to save experiment results to JSON
        results_dir : str, default='results'
            Directory to save results (created if doesn't exist)
        """
        self.U = U
        self.V = V
        self.K = K
        self.s = s
        self.verbose = verbose
        self.save_results = save_results
        self.results_dir = results_dir
        self.random_state = random_state
        
    def _log(self, message):
        """Print message if verbose is True."""
        if self.verbose:
            print(message)
    
    def _check_is_fitted(self):
        """Check if the estimator has been fitted."""
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise NotFittedError("This SubclassClassifier instance is not fitted yet. "
                               "Call 'fit' with appropriate arguments before use :)")
    def _create_speaker_config_from_labels(self, X, y):
        """
        Create speaker configuration from training labels and sort data accordingly.
        
        Parameters:
        -----------
        X : list of numpy arrays
            Training time series data
        y : array-like
            Training labels
        
        Returns:
        --------
        speaker_config : list of dict
            Generated speaker configuration
        X_sorted : list of numpy arrays
            X data sorted by labels
        """
        y = np.array(y)
        
        # Create paired data and sort by labels
        paired_data = list(zip(X, y))
        paired_data.sort(key=lambda pair: str(pair[1]))
        
        # Unpack sorted data
        X_sorted = [pair[0] for pair in paired_data]
        y_sorted = np.array([pair[1] for pair in paired_data])
        
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(y_sorted, return_counts=True)
        
        # Build speaker configuration
        speaker_config = []
        current_idx = 0
        
        for class_id, count in zip(unique_classes, class_counts):
            speaker_config.append({
                'id': class_id,
                'start_idx': current_idx,
                'num_samples': int(count)
            })
            current_idx += count
        
        self._log(f"Auto-generated speaker config from labels:")
        for speaker in speaker_config:
            self._log(f"  Class {speaker['id']}: {speaker['num_samples']} samples (indices {speaker['start_idx']}-{speaker['start_idx'] + speaker['num_samples']-1})")
        
        return speaker_config, X_sorted
        
    def fit(self, X, y=None, speaker_config=None):
        """
        Fit the SubclassClassifier.
        
        Parameters:
        -----------
        X : list of numpy arrays or array-like of shape (n_samples, n_features)
            Training time series data. Each array has shape (n_timesteps, n_features)
        y : array-like of shape (n_samples,), optional
            Target values (class labels). If provided, will auto-generate speaker_config.
        speaker_config : list of dict, optional
            Speaker configuration. If None and y is provided, will be auto-generated.
            If both are None, uses default 9 speakers with 30 samples each.
        
        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        start_time = time.time()
        
        # Create results directory if needed
        if self.save_results and not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        self._log("Starting SubclassClassifier fit...")
        
        # This monstrosity is me trying to capture all of the sklearn possible test inputs
        if hasattr(X, 'shape') and len(X.shape) == 2:
            # sklearn compatibility: use validate_data for proper validation
            X_validated = validate_data(self, X, reset=True)
            # Convert to list of individual samples for our processing
            X = [X_validated[i:i+1].flatten() for i in range(X_validated.shape[0])]
        elif isinstance(X, (list, tuple)) and len(X) > 0:
            # Check if it's a list of numpy arrays (time series format)
            if hasattr(X[0], 'shape') and len(X[0].shape) > 1:
                self.n_features_in_ = X[0].shape[1]
            elif hasattr(X[0], 'shape') and len(X[0].shape) == 1:
                self.n_features_in_ = 1  # 1D arrays are treated as single feature
            else:
                # Handle list of lists (sklearn test format: X.tolist())
                try:
                    # Convert list of lists to numpy array and validate
                    X_array = np.array(X)
                    if X_array.ndim == 2:
                        X_validated = validate_data(self, X_array, reset=True)
                        # Convert to list of individual samples for our processing
                        X = [X_validated[i:i+1].flatten() for i in range(X_validated.shape[0])]
                    else:
                        raise ValueError("X must be a 2D array or list of arrays, and cannot be empty")
                except Exception:
                    raise ValueError("Unsupported input shape")
        else:
            # Try to convert unknown input types to numpy array and validate
            try:
                X_array = np.asarray(X)
                if X_array.ndim == 2:
                    X_validated = validate_data(self, X_array, reset=True)
                    # Convert to list of individual samples for our processing
                    X = [X_validated[i:i+1].flatten() for i in range(X_validated.shape[0])]
                else:
                    raise ValueError("X must be a 2D array or list of arrays, and cannot be empty")
            except Exception:
                raise ValueError("X must be a 2D array or list of arrays, and cannot be empty")
        
        # Validate X and y have same number of samples (sklearn requirement)
        if y is not None:
            y = np.array(y)
            if len(X) != len(y):
                raise ValueError(f"X and y must have the same number of samples. "
                            f"X has {len(X)} samples, y has {len(y)} samples.")

            # Check for NaN or inf values in y (sklearn requirement)
            if np.issubdtype(y.dtype, np.number):
                if np.any(np.isnan(y)):
                    raise ValueError("Input y contains NaN values.")
                if np.any(np.isinf(y)):
                    raise ValueError("Input y contains inf values.")
            else:
                # For non-numeric types, check for None values
                if any(val is None for val in y):
                    raise ValueError("Input y contains None values.")
     
        
        # Auto-generate speaker configuration from y if provided
        if speaker_config is None:
            if y is not None:
                speaker_config, X_sorted = self._create_speaker_config_from_labels(X, y)
                X = X_sorted
            else:
                # Fallback to default configuration for the paper
                speaker_config = [
                    {'id': 1, 'start_idx': 0, 'num_samples': 30},
                    {'id': 2, 'start_idx': 30, 'num_samples': 30},
                    {'id': 3, 'start_idx': 60, 'num_samples': 30},
                    {'id': 4, 'start_idx': 90, 'num_samples': 30},
                    {'id': 5, 'start_idx': 120, 'num_samples': 30},
                    {'id': 6, 'start_idx': 150, 'num_samples': 30},
                    {'id': 7, 'start_idx': 180, 'num_samples': 30},
                    {'id': 8, 'start_idx': 210, 'num_samples': 30},
                    {'id': 9, 'start_idx': 240, 'num_samples': 30},
                ]
        
        self.classes_ = np.array([speaker['id'] for speaker in speaker_config])
        
        # Validate that speaker config doesn't exceed data size (again for sklearn tests)
        max_end_idx = max(speaker['start_idx'] + speaker['num_samples'] for speaker in speaker_config)
        if max_end_idx > len(X):
            raise ValueError(f"Speaker configuration requires {max_end_idx} samples but only {len(X)} provided")
        
        # Step 1: Create binning grid
        self._log(f"Step 1: Creating binning grid (U={self.U}, V={self.V}, K={self.K})")
        self.time_thresholds_, self.feature_thresholds_ = create_binning_grid(X, self.U, self.V)
        
        # Step 2: Convert training data to binary vectors
        self._log("Step 2: Converting training data to binary vectors...")
        binary_vectors = []
        for i, curve in enumerate(X):
            if self.verbose and i % 50 == 0:
                self._log(f"  Processing curve {i+1}/{len(X)}...")
            
            binary_vec = curve_to_binary_vector(
                curve, self.time_thresholds_, self.feature_thresholds_, self.K
            )
            binary_vectors.append(binary_vec)
        
        binary_vectors = np.array(binary_vectors)
        self._log(f"  Binary vector length: {binary_vectors.shape[1]}")
        
        # Step 3: Learn subclasses for each speaker
        self._log("Step 3: Learning subclasses for each speaker...")
        self.speaker_data_ = {}
        
        for speaker in speaker_config:
            self._log(f"  Processing Speaker {speaker['id']} (samples {speaker['start_idx']} to {speaker['start_idx'] + speaker['num_samples']-1})...")
            
            S_positive, S_negative = get_speaker_positive_negative_sets(
                binary_vectors, speaker['start_idx'], speaker['num_samples']
            )

            subclasses = randomized_subclass_method(S_positive, S_negative, s=self.s, verbose=self.verbose, random_state=self.random_state)

            self.speaker_data_[speaker['id']] = {
                'start_idx': speaker['start_idx'],
                'num_samples': speaker['num_samples'],
                'S_positive': S_positive,
                'S_negative': S_negative,
                'subclasses': subclasses
            }
            
            self._log(f"    Found {len(subclasses)} subclasses")
        
        # Set timing and fitted status
        self.fit_time_ = time.time() - start_time
        self.is_fitted_ = True
        
        self._log(f"Fit completed in {self.fit_time_:.2f} seconds")
        return self
        
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : list of numpy arrays or array-like of shape (n_samples, n_features)
            Test time series data. Each array has shape (n_timesteps, n_features)
        
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        start_time = time.time()
        
        # Check if fitted
        self._check_is_fitted()
        
        # Handle both formats: sklearn 2D arrays and time series lists
        if hasattr(X, 'shape') and len(X.shape) == 2:
            # sklearn compatibility: use validate_data for proper validation
            X_validated = validate_data(self, X, reset=False)
            # Convert to list of individual samples for our processing
            X = [X_validated[i:i+1].flatten() for i in range(X_validated.shape[0])]
        elif isinstance(X, (list, tuple)) and len(X) > 0:
            # Check if it's a list of numpy arrays (time series format)
            if hasattr(X[0], 'shape') and len(X[0].shape) > 1:
                n_features = X[0].shape[1]
            elif hasattr(X[0], 'shape') and len(X[0].shape) == 1:
                n_features = 1
            else:
                # Handle list of lists (sklearn test format: X.tolist())
                try:
                    # Convert list of lists to numpy array and validate
                    X_array = np.array(X)
                    if X_array.ndim == 2:
                        X_validated = validate_data(self, X_array, reset=False)
                        # Convert to list of individual samples for our processing
                        X = [X_validated[i:i+1].flatten() for i in range(X_validated.shape[0])]
                    else:
                        raise ValueError("X must be a 2D array or list of arrays, and cannot be empty")
                except Exception:
                    raise ValueError("Unsupported input shape")
                
            if 'n_features' in locals() and n_features != self.n_features_in_:
                raise ValueError(f"X has {n_features} features, but SubclassClassifier "
                            f"is expecting {self.n_features_in_} features as seen in fit.")
        else:
            # Try to convert unknown input types to numpy array and validate
            try:
                X_array = np.asarray(X)
                if X_array.ndim == 2:
                    X_validated = validate_data(self, X_array, reset=False)
                    # Convert to list of individual samples for our processing
                    X = [X_validated[i:i+1].flatten() for i in range(X_validated.shape[0])]
                else:
                    raise ValueError("X must be a 2D array or list of arrays, and cannot be empty")
            except Exception:
                raise ValueError("X must be a 2D array or list of arrays, and cannot be empty")

        self._log(f"Predicting {len(X)} samples...")
        
        # Convert test data to binary vectors
        test_binary_vectors = []
        for i, test_curve in enumerate(X):
            if self.verbose and i % 50 == 0:
                self._log(f"  Processing test curve {i+1}/{len(X)}...")
            
            test_binary_vec = curve_to_binary_vector(
                test_curve, self.time_thresholds_, self.feature_thresholds_, self.K
            )
            test_binary_vectors.append(test_binary_vec)
        
        # Classify each test sample
        predictions = []
        for i, test_binary_vec in enumerate(test_binary_vectors):
            if self.verbose and i % 50 == 0:
                self._log(f"  Classifying test sample {i+1}/{len(test_binary_vectors)}...")
            
            predicted_speaker, method, info = classify_test_sample(test_binary_vec, self.speaker_data_)
            predictions.append(predicted_speaker)
        
        # Store prediction time
        self.predict_time_ = time.time() - start_time
        self._log(f"Prediction completed in {self.predict_time_:.2f} seconds")
        
        return np.array(predictions)
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : list of numpy arrays
            Test time series data
        y : array-like of shape (n_samples,)
            True labels for X
        
        Returns:
        --------
        score : float
            Mean accuracy
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'U': self.U,
            'V': self.V, 
            'K': self.K,
            's': self.s,
            'verbose': self.verbose,
            'save_results': self.save_results,
            'results_dir': self.results_dir,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def save_experiment_results(self, X_test=None, y_test=None, y_pred=None, experiment_name=None):
        """
        Save detailed experiment results to JSON file.
        
        Parameters:
        -----------
        X_test : list of numpy arrays, optional
            Test data used for evaluation (only needed if y_pred not provided)
        y_test : array-like
            True test labels
        y_pred : array-like, optional
            Pre-computed predictions. If None, will compute using X_test.
        experiment_name : str, optional
            Custom name for the experiment. If None, uses timestamp.
        """
        self._check_is_fitted()
        
        # Validate inputs
        if y_pred is None and X_test is None:
            raise ValueError("Either y_pred or X_test must be provided")
        
        if y_test is None:
            raise ValueError("y_test must be provided")
        
        # Track if predictions were pre-computed BEFORE we modify y_pred
        predictions_were_precomputed = y_pred is not None
        
        # Make predictions if not provided
        if y_pred is None:
            self._log("Computing predictions...")
            y_pred = self.predict(X_test)
        else:
            self._log("Using pre-computed predictions")
            y_pred = np.array(y_pred)
        
        # Ensure y_test is numpy array for consistency
        y_test = np.array(y_test)
        
        # Validate prediction and test label shapes
        if len(y_pred) != len(y_test):
            raise ValueError(f"y_pred and y_test must have same length. "
                            f"Got y_pred: {len(y_pred)}, y_test: {len(y_test)}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)
        
        # Count subclasses per speaker
        subclass_counts = {
            speaker_id: len(data['subclasses']) 
            for speaker_id, data in self.speaker_data_.items()
        }
        
        # Prepare results dictionary
        results = {
            'experiment_name': experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'parameters': self.get_params(),
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'subclass_counts': subclass_counts,
            'fit_time_seconds': self.fit_time_,
            'predict_time_seconds': getattr(self, 'predict_time_', None),
            'timestamp': datetime.now().isoformat(),
            'n_train_samples': sum(data['num_samples'] for data in self.speaker_data_.values()),
            'n_test_samples': len(y_test),
            'predictions_precomputed': predictions_were_precomputed
        }
        
        # Ensure results directory exists
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Save to file
        filename = f"{self.results_dir}/{results['experiment_name']}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        self._log(f"Results saved to {filename}")
        return results

# Convenience function for running experiments
def run_experiment_pipeline(train_inputs, test_inputs, train_outputs, test_outputs, 
                           config, experiment_name=None, save_results=True, random_state=None,
                           use_cv=False, cv_folds=5, cv_only=False):
    """
    Run a complete experiment with given configuration.
    
    Parameters:
    -----------
    train_inputs, test_inputs, train_outputs, test_outputs : data
        Training and test data
    config : dict
        Configuration dictionary with keys: U, V, K, s
    experiment_name : str, optional
        Name for the experiment
    save_results : bool, default=True
        Whether to save results to JSON
    random_state : int, optional
        Random seed for reproducibility
    use_cv : bool, default=False
        Whether to use cross-validation for evaluation
    cv_folds : int, default=5
        Number of folds for cross-validation (only used if use_cv=True)
    cv_only : bool, default=False
        If True, only run CV on training data (for hyperparameter selection).
        If False, also evaluate on test set (for final model evaluation).
        
    Returns:
    --------
    classifier : SubclassClassifier
        Fitted classifier
    results : dict
        Experiment results (includes CV results if use_cv=True)
    """
    if use_cv:
        print(f"\n=== CROSS-VALIDATION PHASE ===")
        print(f"Configuration: {config}")
        print(f"Using {cv_folds}-fold cross-validation on training data only")
        
        # Run cross-validation on training data
        cv_results = run_stratified_kfold_cv(
            train_inputs, train_outputs, config, k=cv_folds, random_state=random_state
        )
        
        if cv_only:
            # Return only CV results (for hyperparameter selection)
            results = {
                'experiment_name': experiment_name or f"cv_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'config': config,
                'cv_results': cv_results,
                'evaluation_method': 'cv_only',
                'cv_folds': cv_folds,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save CV-only results if requested
            if save_results:
                if not os.path.exists('results'):
                    os.makedirs('results')
                filename = f"results/{results['experiment_name']}.json"
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"CV-only results saved to {filename}")
            
            print(f"CV Results: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
            return None, results  # No final classifier trained
        
        else:
            # Final evaluation phase
            print(f"\n=== FINAL EVALUATION PHASE ===")
            print(f"Training final model on full training set...")
            print(f"Will evaluate on held-out test set")
            
            # Train final model on full training set
            y_train = np.array([np.argmax(output[0]) + 1 for output in train_outputs])
            classifier = SubclassClassifier(
                U=config['U'], V=config['V'], K=config['K'], s=config['s'],
                verbose=True, save_results=False,
                random_state=random_state
            )
            classifier.fit(train_inputs, y=y_train)
            
            # Evaluate on held-out test set
            y_test = np.array([np.argmax(output[0]) + 1 for output in test_outputs])
            y_pred = classifier.predict(test_inputs)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\nFinal Results:")
            print(f"Cross-validation: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
            print(f"Test set accuracy: {test_accuracy:.4f}")
            
            # Create comprehensive results with both CV and test evaluation
            if save_results:
                # Get detailed test results using the classifier's method
                detailed_results = classifier.save_experiment_results(
                    y_test=y_test, 
                    y_pred=y_pred,
                    experiment_name=experiment_name
                )
                
                # Add CV results to the detailed results
                detailed_results['cv_results'] = cv_results
                detailed_results['evaluation_method'] = 'cv_then_test'
                detailed_results['cv_folds'] = cv_folds
                
                # Save the enhanced results
                filename = f"results/{detailed_results['experiment_name']}.json"
                with open(filename, 'w') as f:
                    json.dump(detailed_results, f, indent=4)
                print(f"Complete results saved to {filename}")
                
                results = detailed_results
            else:
                # Return basic results without saving
                results = {
                    'experiment_name': experiment_name or f"final_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'config': config,
                    'cv_results': cv_results,
                    'test_accuracy': float(test_accuracy),
                    'evaluation_method': 'cv_then_test',
                    'cv_folds': cv_folds,
                    'timestamp': datetime.now().isoformat()
                }
            
            return classifier, results
        
    else:
        # Original single train/test split
        print(f"\n=== SINGLE TRAIN/TEST EXPERIMENT ===")
        print(f"Configuration: {config}")
        
        # Extract true labels from test outputs
        y_test = np.array([np.argmax(output[0]) + 1 for output in test_outputs])
        
        # Create and fit classifier
        classifier = SubclassClassifier(
            U=config['U'], V=config['V'], K=config['K'], s=config['s'],
            verbose=True, save_results=save_results, random_state=random_state
        )
        
        classifier.fit(train_inputs)
        y_pred = classifier.predict(test_inputs)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Save detailed results if requested
        if save_results:
            results = classifier.save_experiment_results(
                y_test=y_test, 
                y_pred=y_pred, 
                experiment_name=experiment_name
            )
        else:
            results = {
                'experiment_name': experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'config': config,
                'accuracy': float(accuracy),
                'evaluation_method': 'single_split',
                'timestamp': datetime.now().isoformat()
            }
    
    return classifier, results

def run_stratified_kfold_cv(train_inputs, train_outputs, config, k=10, random_state=42):
    """
    Run stratified k-fold cross-validation maintaining speaker proportions.
    
    Parameters:
    -----------
    train_inputs : list of numpy arrays
        Training time series data
    train_outputs : list of numpy arrays
        Training outputs (one-hot encoded)
    config : dict
        Configuration dictionary with keys: U, V, K, s
    k : int, default=10
        Number of folds
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    cv_results : dict
        Cross-validation results with fold-wise accuracies and statistics
    """
    # Extract labels from train_outputs
    y_labels = np.array([np.argmax(output[0]) + 1 for output in train_outputs])
    
    print(f"Original data: {len(train_inputs)} samples")
    print(f"Speaker distribution: {np.bincount(y_labels)[1:]}")
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    fold_results = []
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_inputs, y_labels)):
        print(f"\n--- Fold {fold + 1}/{k} ---")
        
        # Split data using indices
        fold_train_inputs = [train_inputs[i] for i in train_idx]
        fold_val_inputs = [train_inputs[i] for i in val_idx]
        fold_train_labels = y_labels[train_idx]
        fold_val_labels = y_labels[val_idx]
        
        # Check distributions
        train_dist = np.bincount(fold_train_labels)[1:]
        val_dist = np.bincount(fold_val_labels)[1:]
        print(f"Training set: {len(fold_train_inputs)} samples, distribution: {train_dist}")
        print(f"Validation set: {len(fold_val_inputs)} samples, distribution: {val_dist}")
        
        # Create classifier with fold-specific random seed
        classifier = SubclassClassifier(
            U=config['U'], V=config['V'], K=config['K'], s=config['s'],
            verbose=False, random_state=random_state + fold
        )
        
        # Fit using y labels (auto-generates speaker config)
        classifier.fit(fold_train_inputs, y=fold_train_labels)
        
        # Evaluate on validation set
        accuracy = classifier.score(fold_val_inputs, fold_val_labels)
        fold_accuracies.append(accuracy)
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'train_size': len(fold_train_inputs),
            'val_size': len(fold_val_inputs),
            'train_speaker_dist': train_dist.tolist(),
            'val_speaker_dist': val_dist.tolist(),
            'fit_time': classifier.fit_time_,
            'predict_time': classifier.predict_time_
        })
        
        print(f"Fold {fold + 1} accuracy: {accuracy:.4f}")
    
    # Calculate statistics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    cv_results = {
        'config': config,
        'k_folds': k,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'fold_accuracies': fold_accuracies,
        'fold_results': fold_results,
        'random_state': random_state
    }
    
    print(f"\n=== CROSS-VALIDATION RESULTS ===")
    print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Individual fold accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    
    return cv_results

# Analysis helper functions
def load_all_results(results_dir='results'):
    """Load all experiment results from JSON files."""
    all_results = []
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist")
        return all_results
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                all_results.append(json.load(f))
    
    return all_results

######
def load_saved_data(data_folder='data'):
    """
    Load the preprocessed ape call data from pickle files.
    
    Args:
        data_folder (str): Path to the folder containing pickle files
        
    Returns:
        tuple: (train_inputs, test_inputs, train_outputs, test_outputs)
    """
    try:
        with open(f'{data_folder}/train_inputs.pkl', 'rb') as f:
            train_inputs = pickle.load(f)
        with open(f'{data_folder}/test_inputs.pkl', 'rb') as f:
            test_inputs = pickle.load(f)
        with open(f'{data_folder}/train_outputs.pkl', 'rb') as f:
            train_outputs = pickle.load(f)
        with open(f'{data_folder}/test_outputs.pkl', 'rb') as f:
            test_outputs = pickle.load(f)
        
        print(f"Data loaded successfully from {data_folder}/")
        print(f"Train samples: {len(train_inputs)}, Test samples: {len(test_inputs)}")
        print(f"Input dimensions: {train_inputs[0].shape[1]}, Output dimensions: {train_outputs[0].shape[1]}")
        
        return train_inputs, test_inputs, train_outputs, test_outputs
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None
    
def create_binning_grid(train_inputs, U=4, V=20):
    """
    Create the binning grid with U time thresholds and V feature thresholds.
    
    Args:
        train_inputs: list of numpy arrays (270 time series, each N x 12)
        U: number of equally spaced time thresholds
        V: number of equally spaced feature thresholds per dimension
    
    Returns:
        time_thresholds: array of U time threshold values
        feature_thresholds: array of shape (12, V) with threshold values for each feature
    """
    # Find maximum time length across all training samples
    max_length = max(len(sample) for sample in train_inputs)
    
    # Create equally spaced time thresholds from 0 to max_length
    time_thresholds = np.linspace(0, max_length, U)
    
    # For each of the features, find min and max values across all samples
    all_data = np.vstack(train_inputs)  # Concatenate all training data
    n_features = all_data.shape[1]  # Get the number of features (instead of hardcoding 12 :) )

    feature_thresholds = np.zeros((n_features, V))
    for dim in range(n_features):
        min_val = np.min(all_data[:, dim])
        max_val = np.max(all_data[:, dim])
        feature_thresholds[dim] = np.linspace(min_val, max_val, V)
    
    return time_thresholds, feature_thresholds

def curve_to_binary_vector(curve, time_thresholds, feature_thresholds, K=1):
    """
    Convert a single time series curve to binary vector using the binning method.
    
    Args:
        curve: numpy array of shape (N, D) representing one time series
        time_thresholds: array of U time threshold values
        feature_thresholds: array of shape (D, V) with threshold values for each feature
        K: minimum number of times curve must pass through rectangle (1 in paper)
    
    Returns:
        binary_vector: binary vector representing the curve
    """
    if curve.ndim == 1:
        # Convert 1D array to 2D with single feature | mostly for scikitlearn compatability
        curve = curve.reshape(-1, 1)
        
    U = len(time_thresholds)
    V = feature_thresholds.shape[1]
    n_dims = curve.shape[1]  # Get actual number of dimensions from the curve
    
    # Calculate number of possible rectangles per dimension
    # We choose any 2 time thresholds and any 2 feature thresholds
    n_time_rectangles = comb(U, 2)  # U choose 2
    n_feature_rectangles = comb(V, 2)  # V choose 2
    n_rectangles_per_dim = n_time_rectangles * n_feature_rectangles
    
    # Binary vector length: n_dims × n_rectangles_per_dim × K × 2
    total_length = n_dims * n_rectangles_per_dim * K * 2
    binary_vector = np.zeros(total_length, dtype=int)
    
    bit_index = 0
    
    for dim in range(n_dims):
        feature_values = curve[:, dim]
        time_steps = np.arange(len(curve))
        
        # Generate all possible time intervals (choose 2 from U thresholds)
        for t_pair in combinations(range(U), 2):
            t_start = time_thresholds[t_pair[0]]
            t_end = time_thresholds[t_pair[1]]
            
            # Generate all possible feature intervals (choose 2 from V thresholds)
            for f_pair in combinations(range(V), 2):
                f_start = feature_thresholds[dim, f_pair[0]]
                f_end = feature_thresholds[dim, f_pair[1]]
                
                # Count how many points fall within this rectangle
                time_mask = (time_steps >= t_start) & (time_steps <= t_end)
                feature_mask = (feature_values >= f_start) & (feature_values <= f_end)
                points_in_rectangle = np.sum(time_mask & feature_mask)
                
                # For each k from 1 to K
                for k in range(1, K + 1):
                    # Check if curve passes through rectangle at least k times
                    passes_through = 1 if points_in_rectangle >= k else 0
                    
                    # Set the bit
                    binary_vector[bit_index] = passes_through
                    bit_index += 1
                    
                    # Set the inverted bit
                    binary_vector[bit_index] = 1 - passes_through
                    bit_index += 1
    
    return binary_vector

def check_exclusiveness(X: Set[int], S_positive: List[np.ndarray], S_negative: List[np.ndarray]) -> bool:
    """
    Check if adding samples in X to form a subclass maintains exclusiveness against S_negative.
    
    The correct exclusiveness condition is:
    No single sample in S_negative should have 1s at ALL positions where X_and = 1
    
    Args:
        X: Set of indices of samples from S_positive to form the subclass
        S_positive: List of positive class binary vectors
        S_negative: List of negative class binary vectors
    
    Returns:
        bool: True if exclusiveness condition is satisfied
    """
    if len(X) == 0:
        return True
    
    # Get the AND operation over all samples in X
    X_samples = [S_positive[i] for i in X]
    X_and = X_samples[0].copy()
    for sample in X_samples[1:]:
        X_and = X_and & sample  # Element-wise AND
    
    # Get positions where X_and = 1
    ones_positions = np.where(X_and == 1)[0]
    
    if len(ones_positions) == 0:
        return True  # No constraints, so exclusive
    
    # For each sample in S_negative, check if it has 1s at ALL positions where X_and = 1
    for neg_sample in S_negative:
        # Check if this negative sample has 1 at ALL positions where X_and = 1
        has_all_ones = all(neg_sample[pos] == 1 for pos in ones_positions)
        if has_all_ones:
            return False  # Found a negative sample that violates exclusiveness
    
    return True  # No negative sample has 1s at all required positions

def randomized_subclass_method(S_positive: List[np.ndarray], 
                             S_negative: List[np.ndarray], 
                             s: int = 1000, 
                             verbose: bool = False,
                             random_state: int = None) -> List[Set[int]]:
    """
    Randomized Subclass Method implementation.
    
    Args:
        S_positive: List of positive class binary vectors
        S_negative: List of negative class binary vectors  
        s: Number of iterations (permutations to examine)
        verbose: Whether to print progress messages
        random_state: Random seed for reproducibility
    
    Returns:
        List of subclasses (each subclass is a set of indices from S_positive)
    """
    # Set random seed for reproducibility
    if random_state is not None:
        random.seed(random_state)

    n = len(S_positive)  # |S+|
    v = 1.0  # Termination condition variable
    Omega = []  # Collection of found subclasses
    
    if verbose:
        print(f"Starting randomized subclass method with {n} positive samples, {s} iterations")
    
    for k in range(1, s+1):
        if verbose and k % 100 == 0:
            print(f"Iteration {k}/{s}, found {len(Omega)} subclasses so far")
        
        # Indices 0, 1, ..., n-1 of S_positive
        T = list(range(n))
        
        # Step 4: initalize empty set X
        X = set()
        
        # Steps 5-11: Generate random permutation and build subclass
        for i in range(n):
            # pick random number r in [0,1)
            r = random.random()
            
            # Here, T always has n-i elements and we don't add 1 since we do 0-based indexing
            j = int(r * len(T))
            
            # get j_th element of T
            x_idx = T[j]
            
            # remove x_idx from T
            T.remove(x_idx)
            
            # If adding x_idx to X maintains exclusiveness, add it
            X_candidate = X.union({x_idx})
            if check_exclusiveness(X_candidate, S_positive, S_negative):
                X = X_candidate
        
        # Avoid duplicated by checking if X already exists in Omega
        X_exists = any(existing_X == X for existing_X in Omega)
        
        # If X is new and non-empty, add to Omega
        if not X_exists and len(X) > 0:
            Omega.append(X)
            
            # Update the termination condition variable v
            try:
                v = v - 1.0 / comb(n, len(X))
            except (ValueError, ZeroDivisionError):
                print(f"Warning: comb(n, len(X)) caused error at iteration {k}, n={n}, len(X)={len(X)}")
                v = v - 0.001  # Fallback
            
            # Check termination condition
            if v <= 0:
                if verbose:
                    print(f"Termination condition met at iteration {k}")
                break

    if verbose:
        print(f"Completed: Found {len(Omega)} unique subclasses")
    return Omega

def get_speaker_positive_negative_sets(binary_vectors, start_idx, num_samples):
    """
    Extract positive and negative sets for a specific speaker.
    
    Args:
        binary_vectors: numpy array - all binary vectors
        start_idx: Starting index for this speaker's samples in binary_vectors
        num_samples: Number of samples for this speaker
    
    Returns:
        S_positive: List of binary vectors for this speaker
        S_negative: List of binary vectors for all other speakers
    """
    # Get binary vectors for this speaker (positive class)
    end_idx = start_idx + num_samples
    S_positive = [binary_vectors[i] for i in range(start_idx, end_idx)]
    
    # Get binary vectors for all other speakers (negative class)
    S_negative = []
    
    # Add all vectors before this speaker
    for i in range(start_idx):
        S_negative.append(binary_vectors[i])
    
    # Add all vectors after this speaker
    for i in range(end_idx, len(binary_vectors)):
        S_negative.append(binary_vectors[i])
    
    print(f"Speaker (start_idx={start_idx}, num_samples={num_samples}):")
    print(f"  Positive samples: {len(S_positive)}")
    print(f"  Negative samples: {len(S_negative)}")
    
    return S_positive, S_negative

def classify_test_sample(test_binary_vec, speaker_data):
    """
    Classify a single test binary vector using the subclass method.
    
    Args:
        test_binary_vec: binary vector of the test sample
        speaker_data: dictionary containing subclasses for each speaker
    
    Returns:
        tuple: (predicted_speaker, method_used, confidence_info)
    """
    speaker_ids = list(speaker_data.keys())
    subclass_match_counts = {}
    subclass_totals = {}
    
    # Step 1: Count subclass matches for each speaker
    for speaker_id in speaker_ids:
        # Default case - if no subclasses are found, continue
        if len(speaker_data[speaker_id]['subclasses']) == 0:
            subclass_match_counts[speaker_id] = 0
            subclass_totals[speaker_id] = 0
            continue
            
        matches = 0
        # check the number of subclasses, best case is when we have only 1 subclass that is S_positive
        total_subclasses = len(speaker_data[speaker_id]['subclasses'])
        
        # Go over all of the subclasses for this speaker and count how many subclass matches we have for the test_vec
        for subclass_indices in speaker_data[speaker_id]['subclasses']:
            # Get the AND pattern for this subclass
            subclass_samples = [speaker_data[speaker_id]['S_positive'][idx] for idx in subclass_indices]
            subclass_and = subclass_samples[0].copy()
            for sample in subclass_samples[1:]:
                subclass_and = subclass_and & sample
            
            # Check if test sample satisfies this subclass (has 1s where subclass_and has 1s)
            ones_positions = np.where(subclass_and == 1)[0]
            if len(ones_positions) == 0:
                # Empty subclass pattern, this means that we don't have any rectangle rules to satisfy so it's a match by default
                matches += 1
            else:
                # Check if test sample has 1s at all positions where subclass_and has 1s
                satisfies_subclass = all(test_binary_vec[pos] == 1 for pos in ones_positions)
                if satisfies_subclass:
                    matches += 1
        
        subclass_match_counts[speaker_id] = matches
        subclass_totals[speaker_id] = total_subclasses
    
    # Calculate match percentages
    match_percentages = {}
    for speaker_id in speaker_ids:
        if subclass_totals[speaker_id] > 0: # Don't want division by 0
            match_percentages[speaker_id] = subclass_match_counts[speaker_id] / subclass_totals[speaker_id]
        else:
            match_percentages[speaker_id] = 0.0
    
    # Step 1: Check if any speaker has matching subclasses and if so return the one where the largest percentage of subclasses is satisfied
    max_percentage = max(match_percentages.values())
    if max_percentage > 0:
        # Find speaker with highest percentage
        best_speaker = max(match_percentages.keys(), key=lambda k: match_percentages[k])
        return best_speaker, "subclass_match", {
            "percentage": max_percentage,
            "matches": subclass_match_counts[best_speaker],
            "total": subclass_totals[best_speaker]
        }
    
    # Step 2: If we did not find exact subclass matches then use nearness measure (number of differing bits)
    nearness_scores = {}
    
    for speaker_id in speaker_ids:
        # Base case - if not subclsses, set high distance
        if len(speaker_data[speaker_id]['subclasses']) == 0:
            nearness_scores[speaker_id] = float('inf')
            continue
            
        total_distance = 0
        total_subclasses = len(speaker_data[speaker_id]['subclasses'])
        
        # Go over all subclasses and for each speaker calculate the average "nearness" based on the paper definition
        for subclass_indices in speaker_data[speaker_id]['subclasses']:
            # Get the AND pattern for this subclass
            subclass_samples = [speaker_data[speaker_id]['S_positive'][idx] for idx in subclass_indices]
            subclass_and = subclass_samples[0].copy()
            for sample in subclass_samples[1:]:
                subclass_and = subclass_and & sample
            
            # Count bits that are 1 in subclass but 0 in test sample
            distance = np.sum((subclass_and == 1) & (test_binary_vec == 0))
            total_distance += distance
        
        # Average distance across all subclasses
        nearness_scores[speaker_id] = total_distance / total_subclasses if total_subclasses > 0 else float('inf')
    
    # Choose speaker with minimum average distance (highest nearness)
    best_speaker = min(nearness_scores.keys(), key=lambda k: nearness_scores[k])
    return best_speaker, "nearness", {
        "distance": nearness_scores[best_speaker],
        "all_distances": nearness_scores
    }