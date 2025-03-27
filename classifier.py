#!/usr/bin/env python3
# BCI Project - EEG Pattern Classifier
# This module provides a machine learning model to classify EEG patterns and interpret user states

import os
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import joblib

class BCIPatternClassifier:
    """
    Classifier for EEG patterns that can be used to interpret what the user might be doing
    based on their brain activity patterns.
    """
    
    def __init__(self, patterns_file='analysis/all_patterns.pkl', model_dir='models'):
        """
        Initialize the BCI pattern classifier.
        
        Parameters:
        -----------
        patterns_file : str
            Path to the pickle file containing pattern characteristics
        model_dir : str
            Directory to save/load trained models
        """
        self.patterns_file = patterns_file
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load patterns data
        self.load_patterns()
        
        # Initialize classifiers
        self.classifiers = {}
        self.interpreters = {
            'motor_imagery': self._interpret_motor_imagery,
            'ssvep': self._interpret_ssvep,
            'facial_expression': self._interpret_facial_expression
        }
        
        # Features for different pattern types
        self.feature_extractors = {
            'motor_imagery': self._extract_motor_imagery_features,
            'ssvep': self._extract_ssvep_features,
            'facial_expression': self._extract_facial_features
        }
    
    def load_patterns(self):
        """Load EEG patterns from the pickle file."""
        try:
            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, 'rb') as f:
                    self.patterns = pickle.load(f)
                
                print(f"Loaded {len(self.patterns)} patterns from {self.patterns_file}")
                
                # Group patterns by type
                self.pattern_types = {}
                for name, data in self.patterns.items():
                    pattern_type = data.get('type', 'unknown')
                    if pattern_type not in self.pattern_types:
                        self.pattern_types[pattern_type] = []
                    self.pattern_types[pattern_type].append(name)
                
                # Print pattern types
                for ptype, patterns in self.pattern_types.items():
                    print(f"Pattern type: {ptype}, count: {len(patterns)}")
            else:
                print(f"Patterns file not found: {self.patterns_file}")
                self.patterns = {}
                self.pattern_types = {}
        except Exception as e:
            print(f"Error loading patterns: {e}")
            self.patterns = {}
            self.pattern_types = {}
    
    def train_models(self, test_size=0.2, random_state=42, min_samples_per_class=2):
        """
        Train classification models for each pattern type.
        
        Parameters:
        -----------
        test_size : float
            Fraction of data to use for testing
        random_state : int
            Random seed for reproducibility
        min_samples_per_class : int
            Minimum number of samples required for each class
        """
        if not self.patterns:
            print("No patterns available for training. Please load patterns first.")
            return
        
        for pattern_type, pattern_names in self.pattern_types.items():
            print(f"\nTraining model for pattern type: {pattern_type}")
            
            # Skip if no feature extractor or interpreter for this type
            if pattern_type not in self.feature_extractors or pattern_type not in self.interpreters:
                print(f"No feature extractor or interpreter for {pattern_type}, skipping.")
                continue
            
            # Prepare training data
            X, y = self._prepare_training_data(pattern_type)
            
            if len(X) == 0 or len(y) == 0:
                print(f"No training data for {pattern_type}, skipping.")
                continue
            
            # Check for classes with too few samples
            class_counts = Counter(y)
            print(f"Class distribution: {class_counts}")
            
            # Filter out classes with too few samples
            insufficient_classes = [cls for cls, count in class_counts.items() if count < min_samples_per_class]
            
            if insufficient_classes:
                print(f"Warning: The following classes have fewer than {min_samples_per_class} samples and will be removed: {insufficient_classes}")
                
                # Filter out samples with insufficient data
                valid_indices = [i for i, label in enumerate(y) if label not in insufficient_classes]
                
                if len(valid_indices) < 2*min_samples_per_class:
                    print(f"Insufficient data for {pattern_type} after filtering, skipping.")
                    continue
                
                X = X[valid_indices]
                y = y[valid_indices]
                
                # Recheck class counts after filtering
                print(f"Class distribution after filtering: {Counter(y)}")
            
            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y)
                
                print(f"Training data: {X_train.shape}, Training labels: {len(y_train)}")
                print(f"Testing data: {X_test.shape}, Testing labels: {len(y_test)}")
                
                # Create and train model
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state))
                ])
                
                pipeline.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Model accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                # Store model
                self.classifiers[pattern_type] = pipeline
                
                # Save model
                model_path = os.path.join(self.model_dir, f"{pattern_type}_classifier.joblib")
                joblib.dump(pipeline, model_path)
                print(f"Saved model to {model_path}")
                
            except Exception as e:
                print(f"Error training model for {pattern_type}: {e}")
    
    def _prepare_training_data(self, pattern_type):
        """
        Prepare training data for a specific pattern type.
        
        Parameters:
        -----------
        pattern_type : str
            Type of pattern to prepare data for
            
        Returns:
        --------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target labels
        """
        X = []
        y = []
        
        # Get patterns of this type
        pattern_names = self.pattern_types.get(pattern_type, [])
        
        # Debug info
        print(f"Preparing training data for {pattern_type}, {len(pattern_names)} patterns available")
        
        for name in pattern_names:
            if name in self.patterns:
                # Extract features using the appropriate extractor
                features = self.feature_extractors[pattern_type](self.patterns[name])
                
                if features is not None:
                    # Extract label (remove type prefix)
                    if '_' in name:
                        label = name.split('_', 1)[1]
                    else:
                        label = name
                    
                    X.append(features)
                    y.append(label)
        
        print(f"Generated {len(X)} feature vectors with labels: {set(y)}")
        
        if len(X) > 0:
            # Convert to numpy arrays
            X_arr = np.array(X)
            y_arr = np.array(y)
            
            # Print shapes for debugging
            print(f"Feature matrix shape: {X_arr.shape}")
            print(f"Label vector shape: {y_arr.shape}")
            
            return X_arr, y_arr
        else:
            return np.array([]), np.array([])
    
    def _extract_motor_imagery_features(self, pattern_data):
        """
        Extract features for motor imagery patterns.
        
        Parameters:
        -----------
        pattern_data : dict
            Data for a specific pattern
            
        Returns:
        --------
        features : numpy.ndarray
            Feature vector
        """
        if 'spectra' not in pattern_data or 'freqs' not in pattern_data:
            return None
        
        spectra = pattern_data['spectra']
        freqs = pattern_data['freqs']
        
        # Extract relevant frequency bands
        alpha_mask = np.logical_and(freqs >= 8, freqs <= 13)
        beta_mask = np.logical_and(freqs >= 14, freqs <= 30)
        
        # Average across channels if needed
        if spectra.ndim > 1:
            # Calculate alpha power for each channel
            alpha_power = np.mean(spectra[:, alpha_mask], axis=1)
            
            # Calculate beta power for each channel
            beta_power = np.mean(spectra[:, beta_mask], axis=1)
            
            # Calculate alpha/beta ratio for each channel
            alpha_beta_ratio = alpha_power / (beta_power + 1e-10)  # Avoid division by zero
            
            # Calculate power in specific frequency bands
            theta_mask = np.logical_and(freqs >= 4, freqs <= 7)
            gamma_mask = np.logical_and(freqs >= 30, freqs <= 50)
            
            theta_power = np.mean(spectra[:, theta_mask], axis=1)
            gamma_power = np.mean(spectra[:, gamma_mask], axis=1)
            
            # Combine features
            features = np.concatenate([
                alpha_power, beta_power, alpha_beta_ratio,
                theta_power, gamma_power
            ])
            
            return features
        else:
            # Single channel data
            alpha_power = np.mean(spectra[alpha_mask])
            beta_power = np.mean(spectra[beta_mask])
            alpha_beta_ratio = alpha_power / (beta_power + 1e-10)  # Avoid division by zero
            
            # Additional frequency bands
            theta_mask = np.logical_and(freqs >= 4, freqs <= 7)
            gamma_mask = np.logical_and(freqs >= 30, freqs <= 50)
            
            theta_power = np.mean(spectra[theta_mask])
            gamma_power = np.mean(spectra[gamma_mask])
            
            # Features
            features = np.array([alpha_power, beta_power, alpha_beta_ratio, theta_power, gamma_power])
            
            return features
    
    def _extract_ssvep_features(self, pattern_data):
        """
        Extract features for SSVEP patterns.
        
        Parameters:
        -----------
        pattern_data : dict
            Data for a specific pattern
            
        Returns:
        --------
        features : numpy.ndarray
            Feature vector
        """
        if 'spectra' not in pattern_data or 'freqs' not in pattern_data:
            return None
        
        spectra = pattern_data['spectra']
        freqs = pattern_data['freqs']
        
        # For SSVEP, we need to identify peaks at stimulus frequencies
        if spectra.ndim > 1:
            # Calculate the average spectrum across channels
            avg_spectrum = np.mean(spectra, axis=0)
            
            # Find peaks (local maxima)
            peak_indices = signal.find_peaks(avg_spectrum, height=0)[0]
            
            # Check if there are peaks
            if len(peak_indices) > 0:
                peak_freqs = freqs[peak_indices]
                peak_powers = avg_spectrum[peak_indices]
                
                # Sort peaks by power
                sorted_indices = np.argsort(peak_powers)[::-1]  # Descending order
                sorted_peak_freqs = peak_freqs[sorted_indices]
                sorted_peak_powers = peak_powers[sorted_indices]
                
                # Take the top 5 peaks (or fewer if there aren't 5)
                num_peaks = min(5, len(sorted_peak_freqs))
                top_freqs = sorted_peak_freqs[:num_peaks]
                top_powers = sorted_peak_powers[:num_peaks]
                
                # Pad with zeros if fewer than 5 peaks
                top_freqs = np.pad(top_freqs, (0, 5 - num_peaks), 'constant', constant_values=0)
                top_powers = np.pad(top_powers, (0, 5 - num_peaks), 'constant', constant_values=0)
                
                # Combine frequencies and powers as features
                features = np.concatenate([top_freqs, top_powers])
                
                return features
            else:
                # Fallback if no peaks found
                # Use basic frequency band power
                ssvep_bands = [(6, 8), (8, 10), (10, 12), (12, 15), (15, 20)]
                band_powers = []
                
                for low, high in ssvep_bands:
                    band_mask = np.logical_and(freqs >= low, freqs <= high)
                    power = np.mean(avg_spectrum[band_mask]) if np.any(band_mask) else 0
                    band_powers.append(power)
                
                # Return band powers as features
                return np.array(band_powers)
        else:
            # Single channel data
            peak_indices = signal.find_peaks(spectra, height=0)[0]
            
            if len(peak_indices) > 0:
                peak_freqs = freqs[peak_indices]
                peak_powers = spectra[peak_indices]
                
                # Sort peaks by power
                sorted_indices = np.argsort(peak_powers)[::-1]  # Descending order
                sorted_peak_freqs = peak_freqs[sorted_indices]
                sorted_peak_powers = peak_powers[sorted_indices]
                
                # Take the top 5 peaks (or fewer if there aren't 5)
                num_peaks = min(5, len(sorted_peak_freqs))
                top_freqs = sorted_peak_freqs[:num_peaks]
                top_powers = sorted_peak_powers[:num_peaks]
                
                # Pad with zeros if fewer than 5 peaks
                top_freqs = np.pad(top_freqs, (0, 5 - num_peaks), 'constant', constant_values=0)
                top_powers = np.pad(top_powers, (0, 5 - num_peaks), 'constant', constant_values=0)
                
                # Combine frequencies and powers as features
                features = np.concatenate([top_freqs, top_powers])
                
                return features
            else:
                # Fallback if no peaks found
                ssvep_bands = [(6, 8), (8, 10), (10, 12), (12, 15), (15, 20)]
                band_powers = []
                
                for low, high in ssvep_bands:
                    band_mask = np.logical_and(freqs >= low, freqs <= high)
                    power = np.mean(spectra[band_mask]) if np.any(band_mask) else 0
                    band_powers.append(power)
                
                # Return band powers as features
                return np.array(band_powers)
    
    def _extract_facial_features(self, pattern_data):
        """
        Extract features for facial expression patterns.
        
        Parameters:
        -----------
        pattern_data : dict
            Data for a specific pattern
            
        Returns:
        --------
        features : numpy.ndarray
            Feature vector
        """
        if 'spectra' not in pattern_data or 'freqs' not in pattern_data:
            return None
        
        spectra = pattern_data['spectra']
        freqs = pattern_data['freqs']
        
        # Facial expressions have characteristic frequency band activities
        # We'll focus on frontal asymmetry which is related to emotional states
        
        # Extract relevant frequency bands
        alpha_mask = np.logical_and(freqs >= 8, freqs <= 13)
        beta_mask = np.logical_and(freqs >= 14, freqs <= 30)
        
        if spectra.ndim > 1:
            # For facial expressions, calculate frontal asymmetry
            # Assume the first half of channels are from left hemisphere,
            # and the second half from right hemisphere (simplified)
            num_channels = spectra.shape[0]
            half_channels = max(1, num_channels // 2)  # Ensure at least 1 channel
            
            # Alpha power in left and right frontal regions
            left_alpha = np.mean(spectra[:half_channels, alpha_mask]) if half_channels > 0 else 0
            right_alpha = np.mean(spectra[half_channels:, alpha_mask]) if half_channels < num_channels else 0
            
            # Beta power in left and right frontal regions
            left_beta = np.mean(spectra[:half_channels, beta_mask]) if half_channels > 0 else 0
            right_beta = np.mean(spectra[half_channels:, beta_mask]) if half_channels < num_channels else 0
            
            # Calculate asymmetry scores (right - left) / (right + left)
            # Higher right hemisphere activity is associated with negative emotions
            alpha_asymmetry = (right_alpha - left_alpha) / (right_alpha + left_alpha + 1e-10)
            beta_asymmetry = (right_beta - left_beta) / (right_beta + left_beta + 1e-10)
            
            # Additional frequency bands
            theta_mask = np.logical_and(freqs >= 4, freqs <= 7)
            gamma_mask = np.logical_and(freqs >= 30, freqs <= 50)
            
            left_theta = np.mean(spectra[:half_channels, theta_mask]) if half_channels > 0 else 0
            right_theta = np.mean(spectra[half_channels:, theta_mask]) if half_channels < num_channels else 0
            
            left_gamma = np.mean(spectra[:half_channels, gamma_mask]) if half_channels > 0 else 0
            right_gamma = np.mean(spectra[half_channels:, gamma_mask]) if half_channels < num_channels else 0
            
            theta_asymmetry = (right_theta - left_theta) / (right_theta + left_theta + 1e-10)
            gamma_asymmetry = (right_gamma - left_gamma) / (right_gamma + left_gamma + 1e-10)
            
            # Features
            features = np.array([
                left_alpha, right_alpha, alpha_asymmetry,
                left_beta, right_beta, beta_asymmetry,
                left_theta, right_theta, theta_asymmetry,
                left_gamma, right_gamma, gamma_asymmetry
            ])
            
            return features
        else:
            # Single channel data - can't calculate asymmetry
            alpha_power = np.mean(spectra[alpha_mask])
            beta_power = np.mean(spectra[beta_mask])
            
            # Additional frequency bands
            theta_mask = np.logical_and(freqs >= 4, freqs <= 7)
            gamma_mask = np.logical_and(freqs >= 30, freqs <= 50)
            
            theta_power = np.mean(spectra[theta_mask])
            gamma_power = np.mean(spectra[gamma_mask])
            
            # Features
            features = np.array([alpha_power, beta_power, theta_power, gamma_power])
            
            return features
    
    def load_models(self):
        """Load trained models from disk."""
        for pattern_type in self.pattern_types.keys():
            model_path = os.path.join(self.model_dir, f"{pattern_type}_classifier.joblib")
            if os.path.exists(model_path):
                try:
                    self.classifiers[pattern_type] = joblib.load(model_path)
                    print(f"Loaded model for {pattern_type} from {model_path}")
                except Exception as e:
                    print(f"Error loading model for {pattern_type}: {e}")
    
    def classify_eeg(self, eeg_data, sample_rate=250.0, window_size=2.0):
        """
        Classify EEG data and interpret what the user might be doing.
        
        Parameters:
        -----------
        eeg_data : numpy.ndarray
            EEG data array of shape (samples, channels)
        sample_rate : float
            Sampling rate in Hz
        window_size : float
            Analysis window size in seconds
            
        Returns:
        --------
        results : dict
            Classification results for different pattern types
        interpretation : str
            Human-readable interpretation of what the user might be doing
        """
        results = {}
        interpretations = []
        
        # Process each pattern type
        for pattern_type, classifier in self.classifiers.items():
            # Extract features for this pattern type
            features = self._extract_features_from_eeg(
                eeg_data, sample_rate, window_size, pattern_type)
            
            if features is None:
                continue
            
            # Classify
            try:
                prediction = classifier.predict([features])[0]
                probability = np.max(classifier.predict_proba([features])[0])
                
                results[pattern_type] = {
                    'prediction': prediction,
                    'probability': probability
                }
                
                # Add interpretation
                if probability > 0.6:  # Confidence threshold
                    interpretation = self.interpreters[pattern_type](prediction, probability)
                    interpretations.append(interpretation)
            
            except Exception as e:
                print(f"Error classifying {pattern_type}: {e}")
        
        # Combine interpretations
        if interpretations:
            combined_interpretation = " ".join(interpretations)
        else:
            combined_interpretation = "Unable to determine what the user might be doing."
        
        return results, combined_interpretation
    
    def _extract_features_from_eeg(self, eeg_data, sample_rate, window_size, pattern_type):
        """
        Extract features from raw EEG data for a specific pattern type.
        
        Parameters:
        -----------
        eeg_data : numpy.ndarray
            EEG data array of shape (samples, channels)
        sample_rate : float
            Sampling rate in Hz
        window_size : float
            Analysis window size in seconds
        pattern_type : str
            Type of pattern to extract features for
            
        Returns:
        --------
        features : numpy.ndarray
            Feature vector
        """
        if pattern_type not in self.feature_extractors:
            return None
        
        # Calculate number of samples in window
        window_samples = int(window_size * sample_rate)
        
        # Use the last window_samples samples
        if eeg_data.shape[0] > window_samples:
            window_data = eeg_data[-window_samples:, :]
        else:
            window_data = eeg_data
        
        # Calculate PSD for each channel
        n_channels = window_data.shape[1]
        spectra = []
        
        for ch in range(n_channels):
            # Ensure the data is long enough for welch
            if window_data.shape[0] >= 256:
                freqs, psd = signal.welch(window_data[:, ch], fs=sample_rate, nperseg=256)
            else:
                freqs, psd = signal.welch(window_data[:, ch], fs=sample_rate, 
                                          nperseg=min(256, window_data.shape[0]))
            spectra.append(psd)
        
        # Create pattern_data dictionary for the feature extractor
        pattern_data = {
            'spectra': np.array(spectra),
            'freqs': freqs,
            'type': pattern_type
        }
        
        # Extract features using the appropriate extractor
        features = self.feature_extractors[pattern_type](pattern_data)
        
        return features
    
    def _interpret_motor_imagery(self, prediction, probability):
        """
        Interpret motor imagery prediction.
        
        Parameters:
        -----------
        prediction : str
            Predicted class
        probability : float
            Prediction probability
            
        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        confidence = "might be" if probability < 0.8 else "is likely"
        
        if prediction == 'rest':
            return f"The user {confidence} in a relaxed state with minimal motor activity."
        elif prediction == 'left_hand':
            return f"The user {confidence} thinking about moving their left hand."
        elif prediction == 'right_hand':
            return f"The user {confidence} thinking about moving their right hand."
        elif prediction == 'both_hands':
            return f"The user {confidence} thinking about moving both hands."
        elif prediction == 'feet':
            return f"The user {confidence} thinking about moving their feet."
        else:
            return f"The user {confidence} performing some kind of motor imagery: {prediction}."
    
    def _interpret_ssvep(self, prediction, probability):
        """
        Interpret SSVEP prediction.
        
        Parameters:
        -----------
        prediction : str
            Predicted class
        probability : float
            Prediction probability
            
        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        confidence = "might be" if probability < 0.8 else "is likely"
        
        # Extract frequency if available (format: ssvep_XHz)
        if 'Hz' in prediction:
            freq = prediction.split('_')[1].replace('Hz', '')
            return f"The user {confidence} looking at a visual stimulus blinking at {freq} Hz."
        else:
            return f"The user {confidence} focusing on a visual target."
    
    def _interpret_facial_expression(self, prediction, probability):
        """
        Interpret facial expression prediction.
        
        Parameters:
        -----------
        prediction : str
            Predicted class
        probability : float
            Prediction probability
            
        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        confidence = "might be" if probability < 0.8 else "is likely"
        
        if prediction == 'neutral':
            return f"The user {confidence} maintaining a neutral facial expression."
        elif prediction == 'happy':
            return f"The user {confidence} feeling happy or experiencing positive emotions."
        elif prediction == 'sad':
            return f"The user {confidence} feeling sad or experiencing negative emotions."
        elif prediction == 'angry':
            return f"The user {confidence} feeling angry or frustrated."
        elif prediction == 'surprised':
            return f"The user {confidence} feeling surprised or startled."
        else:
            return f"The user {confidence} expressing {prediction}."
    
    def visualize_classification(self, eeg_data, sample_rate=250.0, window_size=2.0):
        """
        Visualize classification results for EEG data.
        
        Parameters:
        -----------
        eeg_data : numpy.ndarray
            EEG data array of shape (samples, channels)
        sample_rate : float
            Sampling rate in Hz
        window_size : float
            Analysis window size in seconds
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object with visualization
        """
        results, interpretation = self.classify_eeg(eeg_data, sample_rate, window_size)
        
        # Create figure
        fig, axes = plt.subplots(len(results) + 1, 1, figsize=(10, 2 + 3 * len(results)))
        
        # Plot EEG data
        if len(results) > 0:
            ax_eeg = axes[0]
        else:
            ax_eeg = axes
        
        time_vec = np.arange(eeg_data.shape[0]) / sample_rate
        for ch in range(min(3, eeg_data.shape[1])):  # Plot up to 3 channels
            ax_eeg.plot(time_vec, eeg_data[:, ch] + ch * 50, label=f'Ch {ch+1}')
        
        ax_eeg.set_title('EEG Data')
        ax_eeg.set_xlabel('Time (s)')
        ax_eeg.set_ylabel('Amplitude (μV)')
        ax_eeg.legend()
        
        # Plot classification results
        for i, (pattern_type, result) in enumerate(results.items()):
            if len(results) > 0:
                ax = axes[i + 1]
            else:
                continue
            
            prediction = result['prediction']
            probability = result['probability']
            
            # Create bar plot
            ax.bar([prediction], [probability], color='skyblue')
            ax.set_title(f'Classification: {pattern_type}')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.text(0, probability + 0.05, f"{probability:.2f}", ha='center')
        
        # Add interpretation as figure title
        fig.suptitle(interpretation, fontsize=14)
        
        fig.tight_layout()
        return fig


# Integration with real-time streaming
class RealtimeEEGClassifier:
    """
    Real-time classifier for EEG data that can be used with streaming EEG.
    """
    
    def __init__(self, classifier, buffer_size=500, step_size=100, sample_rate=250.0):
        """
        Initialize the real-time EEG classifier.
        
        Parameters:
        -----------
        classifier : BCIPatternClassifier
            Trained classifier
        buffer_size : int
            Size of the EEG buffer in samples
        step_size : int
            Number of samples to step for each update
        sample_rate : float
            Sampling rate in Hz
        """
        self.classifier = classifier
        self.buffer_size = buffer_size
        self.step_size = step_size
        self.sample_rate = sample_rate
        
        # Initialize buffer
        self.buffer = np.zeros((buffer_size, 8))  # Default to 8 channels
        self.buffer_idx = 0
        self.buffer_full = False
        
        # Results
        self.latest_results = {}
        self.latest_interpretation = "No classification yet."
    
    def update(self, chunk):
        """
        Update the buffer with a new chunk of EEG data.
        
        Parameters:
        -----------
        chunk : numpy.ndarray
            New chunk of EEG data, shape (samples, channels)
            
        Returns:
        --------
        update_ready : bool
            True if a new classification is ready
        """
        n_samples, n_channels = chunk.shape
        
        # Resize buffer if number of channels changed
        if n_channels != self.buffer.shape[1]:
            new_buffer = np.zeros((self.buffer_size, n_channels))
            
            # Copy old data if buffer was previously filled
            if self.buffer_full:
                old_samples = min(self.buffer_size, self.buffer.shape[1])
                new_buffer[-old_samples:, :min(n_channels, self.buffer.shape[1])] = \
                    self.buffer[-old_samples:, :min(n_channels, n_channels)]
            
            self.buffer = new_buffer
        
        # Add chunk to buffer
        if n_samples >= self.buffer_size:
            # Chunk is larger than buffer, just use the last buffer_size samples
            self.buffer = chunk[-self.buffer_size:, :]
            self.buffer_full = True
            self.buffer_idx = 0
        else:
            # Fill buffer
            space_left = self.buffer_size - self.buffer_idx
            
            if n_samples <= space_left:
                # Enough space to add entire chunk
                self.buffer[self.buffer_idx:self.buffer_idx + n_samples, :] = chunk
                self.buffer_idx += n_samples
                
                # Check if buffer is now full
                if self.buffer_idx >= self.buffer_size:
                    self.buffer_full = True
                    self.buffer_idx = 0
            else:
                # Not enough space, need to wrap around
                # Add what fits
                self.buffer[self.buffer_idx:, :] = chunk[:space_left, :]
                
                # Wrap around and add the rest
                remaining = n_samples - space_left
                self.buffer[:remaining, :] = chunk[space_left:, :]
                self.buffer_idx = remaining
                self.buffer_full = True
        
        # Classify if buffer is full
        update_ready = False
        if self.buffer_full and self.buffer_idx % self.step_size == 0:
            self.latest_results, self.latest_interpretation = \
                self.classifier.classify_eeg(self.buffer, self.sample_rate)
            update_ready = True
        
        return update_ready
    
    def get_latest_results(self):
        """
        Get the latest classification results.
        
        Returns:
        --------
        results : dict
            Classification results for different pattern types
        interpretation : str
            Human-readable interpretation of what the user might be doing
        """
        return self.latest_results, self.latest_interpretation


# Example usage
if __name__ == "__main__":
    # Create classifier
    classifier = BCIPatternClassifier()
    
    # Train models (or load if already trained)
    if not os.path.exists(os.path.join('models', 'motor_imagery_classifier.joblib')):
        classifier.train_models()
    else:
        classifier.load_models()
    
    # Example: Classify simulated data
    import numpy as np
    
    # Generate sample EEG data for a specific pattern (simulated)
    # In a real application, this would come from your EEG device or simulator
    sample_rate = 250.0
    duration = 5.0  # seconds
    n_samples = int(sample_rate * duration)
    n_channels = 8
    
    # Create example EEG data (simulated motor imagery: right hand)
    eeg_data = np.random.randn(n_samples, n_channels) * 5  # Base noise
    
    # Add mu rhythm suppression in motor cortex (simulating right hand movement imagery)
    t = np.arange(n_samples) / sample_rate
    mu_rhythm = np.sin(2 * np.pi * 10 * t)  # 10 Hz mu rhythm
    
    # Add to left motor cortex channels (assuming channels 2 and 3 are over left motor cortex)
    eeg_data[:, 2] += mu_rhythm * 2
    eeg_data[:, 3] += mu_rhythm * 2
    
    # Classify the data
    results, interpretation = classifier.classify_eeg(eeg_data, sample_rate)
    
    print("\nClassification Results:")
    for pattern_type, result in results.items():
        print(f"  {pattern_type}: {result['prediction']} (probability: {result['probability']:.2f})")
    
    print("\nInterpretation:")
    print(f"  {interpretation}")
    
    # Test with real-time classifier
    realtime_classifier = RealtimeEEGClassifier(classifier, buffer_size=500, step_size=125, sample_rate=sample_rate)
    
    # Simulate streaming by breaking data into chunks
    chunk_size = 50  # 50 samples per chunk (typical for BCI applications)
    n_chunks = n_samples // chunk_size
    
    print("\nSimulating real-time streaming:")
    for i in range(n_chunks):
        chunk = eeg_data[i*chunk_size:(i+1)*chunk_size, :]
        update_ready = realtime_classifier.update(chunk)
        
        if update_ready:
            results, interpretation = realtime_classifier.get_latest_results()
            print(f"Chunk {i+1}/{n_chunks}: {interpretation}")
    
    print("\nReal-time classification complete.")