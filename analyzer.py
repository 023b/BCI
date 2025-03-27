#!/usr/bin/env python3
# BCI Project - Pattern Analyzer (Fixed Version)
# This script analyzes EEG data from the downloaded datasets to extract characteristic patterns

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mne
from mne.io import read_raw_edf
import pandas as pd
from pathlib import Path
import glob
import pickle
from scipy.io import loadmat

class PatternAnalyzer:
    """
    Analyzes EEG data from training datasets to extract key characteristics
    for different mental states/patterns.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the pattern analyzer with the data directory.
        
        Parameters:
        -----------
        data_dir : str
            Path to the BCI datasets directory
        """
        if data_dir is None:
            # Use current directory if not specified
            self.data_dir = os.getcwd()
        else:
            self.data_dir = data_dir
            
        self.patterns = {}
        self.pattern_characteristics = {}
        
        # Create analysis directory if it doesn't exist
        self.analysis_dir = os.path.join(self.data_dir, 'analysis')
        os.makedirs(self.analysis_dir, exist_ok=True)
        
    def analyze_physionet_mi(self, num_subjects=5, save_results=True):
        """
        Analyze the PhysioNet Motor Imagery dataset to extract patterns.
        
        Parameters:
        -----------
        num_subjects : int
            Number of subjects to analyze (limit for faster processing)
        save_results : bool
            Whether to save the results to a pickle file
        """
        print(f"Analyzing PhysioNet Motor Imagery dataset (first {num_subjects} subjects)...")
        
        # Path to the PhysioNet dataset (based on the actual directory structure)
        physionet_dir = os.path.join(self.data_dir, 'motor_imagery', 'physionet_eeg', 'physionet.org')
        
        if not os.path.exists(physionet_dir):
            print(f"Directory not found: {physionet_dir}")
            return
        
        # Mapping for event annotations in the PhysioNet dataset
        # Based on the dataset description
        event_id = {
            'T0': 0,     # Rest
            'T1': 1,     # Left fist movement
            'T2': 2,     # Right fist movement
            'T3': 3,     # Both fists movement
            'T4': 4      # Both feet movement
        }
        
        # Map event IDs to our pattern names
        event_to_pattern = {
            0: 'rest',
            1: 'left_hand',
            2: 'right_hand',
            3: 'both_hands',
            4: 'feet'
        }
        
        # Store frequency spectra for each pattern
        spectra = {pattern: [] for pattern in event_to_pattern.values()}
        
        # Process each subject
        subject_dirs = sorted(glob.glob(os.path.join(physionet_dir, 'S*')))[:num_subjects]
        
        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)
            print(f"Processing subject: {subject_id}")
            
            # We're interested in motor imagery tasks (tasks 4-7)
            # Task 4: Opening and closing left or right fist
            # Task 5: Imagining opening and closing left or right fist
            # Task 6: Opening and closing both fists or both feet
            # Task 7: Imagining opening and closing both fists or both feet
            tasks = [4, 5, 6, 7]
            
            for task in tasks:
                # File naming format in PhysioNet dataset: S<subject>R<task>.edf
                # e.g., S001R04.edf for subject 1, task 4
                task_str = f"{task:02d}"
                edf_files = glob.glob(os.path.join(subject_dir, f"*R{task_str}.edf"))
                
                for edf_file in edf_files:
                    try:
                        print(f"Processing {edf_file}...")
                        
                        # Load EDF file using MNE
                        raw = read_raw_edf(edf_file, preload=True)
                        
                        # Fix: Convert annotations to events
                        events, event_mapping = mne.events_from_annotations(raw, event_id=event_id)
                        
                        # Skip if no events found
                        if len(events) == 0:
                            print(f"No events found in {edf_file}")
                            continue
                        
                        # Define epochs from events
                        tmin, tmax = 0.0, 4.0  # Time window after stimulus
                        epochs = mne.Epochs(raw, events, event_id=event_mapping, tmin=tmin, tmax=tmax, 
                                            baseline=None, preload=True)
                        
                        # Get unique event IDs in this file
                        unique_events = np.unique(events[:, 2])
                        
                        # Process each event type
                        for event_code in unique_events:
                            if event_code in event_to_pattern:
                                pattern = event_to_pattern[event_code]
                                print(f"  Extracting '{pattern}' patterns...")
                                
                                # Get epochs for this event
                                # Use string representation of the event code as per MNE conventions
                                event_epochs = epochs[f"{event_code}"]
                                
                                if len(event_epochs) == 0:
                                    print(f"  No epochs found for event {event_code}")
                                    continue
                                
                                # Calculate power spectral density
                                psds, freqs = mne.time_frequency.psd_welch(
                                    event_epochs, fmin=1, fmax=50, n_fft=256)
                                
                                # Average across trials
                                mean_psd = np.mean(psds, axis=0)
                                spectra[pattern].append(mean_psd)
                                
                                print(f"  Added {len(event_epochs)} trials for '{pattern}'")
                    
                    except Exception as e:
                        print(f"Error processing {edf_file}: {e}")
        
        # Average across subjects for each pattern
        for pattern, pattern_spectra in spectra.items():
            if pattern_spectra:
                # Convert to numpy array and average
                pattern_spectra_array = np.array(pattern_spectra)
                avg_spectra = np.mean(pattern_spectra_array, axis=0)
                
                # Store characteristics
                self.pattern_characteristics[f'mi_{pattern}'] = {
                    'spectra': avg_spectra,
                    'freqs': freqs,
                    'channels': raw.ch_names if 'raw' in locals() else None,
                    'type': 'motor_imagery'
                }
                
                # Print some statistics
                print(f"Pattern: mi_{pattern}")
                print(f"  Samples: {len(pattern_spectra)}")
                print(f"  Channels: {avg_spectra.shape[0]}")
                print(f"  Frequency bins: {avg_spectra.shape[1]}")
                
                # Calculate dominant frequency bands
                alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
                beta_idx = np.logical_and(freqs >= 14, freqs <= 30)
                
                if alpha_idx.any() and avg_spectra.size > 0:
                    alpha_power = np.mean(avg_spectra[:, alpha_idx], axis=1)
                    print(f"  Alpha power (8-13 Hz): {np.mean(alpha_power):.2f}")
                
                if beta_idx.any() and avg_spectra.size > 0:
                    beta_power = np.mean(avg_spectra[:, beta_idx], axis=1)
                    print(f"  Beta power (14-30 Hz): {np.mean(beta_power):.2f}")
        
        if save_results and self.pattern_characteristics:
            output_file = os.path.join(self.analysis_dir, 'physionet_mi_patterns.pkl')
            
            with open(output_file, 'wb') as f:
                pickle.dump(self.pattern_characteristics, f)
            
            print(f"Saved pattern characteristics to {output_file}")
            
        self._plot_pattern_spectra('motor_imagery')

    def analyze_bnci_ssvep(self, save_results=True):
        """
        Analyze the BNCI SSVEP dataset to extract eye movement/gaze patterns.
        
        Parameters:
        -----------
        save_results : bool
            Whether to save the results to a pickle file
        """
        print("Analyzing BNCI SSVEP dataset...")
        
        # Path to the BNCI SSVEP dataset (based on actual directory structure)
        ssvep_dir = os.path.join(self.data_dir, 'eye_movement', 'bnci_ssvep')
        
        if not os.path.exists(ssvep_dir):
            print(f"Directory not found: {ssvep_dir}")
            return
        
        # Load the .mat files
        data_files = glob.glob(os.path.join(ssvep_dir, '*.mat'))
        
        if not data_files:
            print("No .mat files found in BNCI SSVEP directory")
            return
        
        # Store frequency spectra for each pattern
        spectra = {}
        
        # Process each data file
        for data_file in data_files:
            try:
                print(f"Processing {data_file}...")
                
                # Load .mat file
                mat_data = loadmat(data_file)
                
                # The data is likely in a nested structure
                X, y = None, None
                
                # Try to extract data from the 'data' field
                if 'data' in mat_data and isinstance(mat_data['data'], np.ndarray):
                    nested_data = mat_data['data']
                    
                    # If it's a structured array with field names
                    if nested_data.dtype.names is not None:
                        print(f"  Found structured array with fields: {nested_data.dtype.names}")
                        
                        # Extract X and y from the first element if available
                        if nested_data.size > 0:
                            first_item = nested_data[0]
                            
                            # Access the data depending on the structure
                            try:
                                if 'X' in nested_data.dtype.names:
                                    X_field = first_item['X']
                                    if isinstance(X_field, np.ndarray) and X_field.size > 0:
                                        if X_field.ndim == 0:  # Handle scalar case
                                            X = X_field.item()
                                        else:
                                            X = X_field
                                
                                if 'y' in nested_data.dtype.names:
                                    y_field = first_item['y']
                                    if isinstance(y_field, np.ndarray) and y_field.size > 0:
                                        if y_field.ndim == 0:  # Handle scalar case
                                            y = y_field.item()
                                        else:
                                            y = y_field
                                            
                                            # Flatten if needed
                                            if y.ndim > 1:
                                                y = y.flatten()
                                                
                                # Alternative field names
                                if X is None and 'trial' in nested_data.dtype.names:
                                    X_field = first_item['trial']
                                    if isinstance(X_field, np.ndarray) and X_field.size > 0:
                                        X = X_field
                                
                                if y is None and 'class' in nested_data.dtype.names:
                                    y_field = first_item['class']
                                    if isinstance(y_field, np.ndarray) and y_field.size > 0:
                                        y = y_field
                                        if y.ndim > 1:
                                            y = y.flatten()
                            
                            except Exception as e:
                                print(f"  Error accessing structured data: {e}")
                    
                    # If it's not a structured array or we couldn't extract the data
                    if X is None:
                        print("  Using fallback approach for data extraction...")
                        # Treat the entire data array as X
                        X = nested_data
                        # Create dummy labels
                        y = np.zeros(X.shape[0] if X.ndim > 1 else 1)
                
                # If we still couldn't find X, look for any usable array
                if X is None:
                    print("  Trying direct array extraction...")
                    for key, value in mat_data.items():
                        if key not in ['__header__', '__version__', '__globals__'] and isinstance(value, np.ndarray) and value.size > 0:
                            print(f"  Using '{key}' as data")
                            X = value
                            break
                    
                    # Create dummy labels if needed
                    if X is not None and y is None:
                        y = np.zeros(X.shape[0] if X.ndim > 1 else 1)
                
                # Check if we found usable data
                if X is None:
                    print(f"  Could not find usable data in {data_file}")
                    continue
                
                print(f"  Extracted data with shape: {X.shape if hasattr(X, 'shape') else 'scalar'}")
                if y is not None:
                    print(f"  Labels shape: {y.shape if hasattr(y, 'shape') else 'scalar'}")
                
                # Process data to extract patterns
                # This depends on the exact structure of the data, which varies by dataset
                
                # If X is a 3D array (trials, channels, time)
                if isinstance(X, np.ndarray) and X.ndim == 3:
                    print("  Processing 3D data array (trials, channels, time)")
                    
                    # If y is available, use it to separate patterns
                    if y is not None and y.size == X.shape[0]:
                        unique_labels = np.unique(y)
                        print(f"  Found {len(unique_labels)} unique classes: {unique_labels}")
                        
                        for label in unique_labels:
                            pattern_name = f'ssvep_{int(label) if isinstance(label, (int, float)) else label}'
                            pattern_data = X[y == label]
                            
                            print(f"  Processing pattern {pattern_name} with {pattern_data.shape[0]} trials")
                            
                            # Calculate PSD for each trial and channel
                            pattern_spectra = []
                            
                            for trial in pattern_data:
                                trial_spectra = []
                                for ch in range(trial.shape[0]):
                                    # Ensure the data is long enough for welch
                                    if trial[ch].size >= 256:
                                        freqs, psd = signal.welch(trial[ch], fs=250, nperseg=256)
                                    else:
                                        freqs, psd = signal.welch(trial[ch], fs=250, nperseg=min(256, trial[ch].size))
                                    trial_spectra.append(psd)
                                pattern_spectra.append(np.array(trial_spectra))
                            
                            # Average across trials
                            if pattern_spectra:
                                avg_spectra = np.mean(pattern_spectra, axis=0)
                                
                                # Store in spectra dictionary
                                if pattern_name not in spectra:
                                    spectra[pattern_name] = []
                                
                                spectra[pattern_name].append(avg_spectra)
                                print(f"  Added spectrum for {pattern_name}")
                    
                    # If no labels, treat all trials as one pattern
                    else:
                        pattern_name = 'ssvep_0'  # Default name
                        print(f"  No labels found, treating all trials as {pattern_name}")
                        
                        # Calculate PSD for each trial and channel
                        pattern_spectra = []
                        
                        for trial in X:
                            trial_spectra = []
                            for ch in range(trial.shape[0]):
                                # Ensure the data is long enough for welch
                                if trial[ch].size >= 256:
                                    freqs, psd = signal.welch(trial[ch], fs=250, nperseg=256)
                                else:
                                    freqs, psd = signal.welch(trial[ch], fs=250, nperseg=min(256, trial[ch].size))
                                trial_spectra.append(psd)
                            pattern_spectra.append(np.array(trial_spectra))
                        
                        # Average across trials
                        if pattern_spectra:
                            avg_spectra = np.mean(pattern_spectra, axis=0)
                            
                            # Store in spectra dictionary
                            if pattern_name not in spectra:
                                spectra[pattern_name] = []
                            
                            spectra[pattern_name].append(avg_spectra)
                            print(f"  Added spectrum for {pattern_name}")
                
                # If X is a 2D array, the interpretation depends on the shape
                elif isinstance(X, np.ndarray) and X.ndim == 2:
                    print("  Processing 2D data array")
                    
                    # If more columns than rows, likely (channels, time)
                    if X.shape[1] > X.shape[0]:
                        print("  Interpreting as (channels, time)")
                        
                        pattern_name = 'ssvep_0'  # Default name
                        pattern_spectra = []
                        
                        # Calculate PSD for each channel
                        for ch in range(X.shape[0]):
                            # Ensure the data is long enough for welch
                            if X[ch].size >= 256:
                                freqs, psd = signal.welch(X[ch], fs=250, nperseg=256)
                            else:
                                freqs, psd = signal.welch(X[ch], fs=250, nperseg=min(256, X[ch].size))
                            pattern_spectra.append(psd)
                        
                        # Store in spectra dictionary
                        if pattern_name not in spectra:
                            spectra[pattern_name] = []
                        
                        spectra[pattern_name].append(np.array(pattern_spectra))
                        print(f"  Added spectrum for {pattern_name}")
                    
                    # Otherwise, likely (trials, time) for a single channel
                    else:
                        print("  Interpreting as (trials, time)")
                        
                        # If y is available, use it to separate patterns
                        if y is not None and y.size == X.shape[0]:
                            unique_labels = np.unique(y)
                            print(f"  Found {len(unique_labels)} unique classes: {unique_labels}")
                            
                            for label in unique_labels:
                                pattern_name = f'ssvep_{int(label) if isinstance(label, (int, float)) else label}'
                                pattern_data = X[y == label]
                                
                                print(f"  Processing pattern {pattern_name} with {pattern_data.shape[0]} trials")
                                
                                # Calculate PSD for each trial
                                pattern_spectra = []
                                
                                for trial in pattern_data:
                                    # Ensure the data is long enough for welch
                                    if trial.size >= 256:
                                        freqs, psd = signal.welch(trial, fs=250, nperseg=256)
                                    else:
                                        freqs, psd = signal.welch(trial, fs=250, nperseg=min(256, trial.size))
                                    pattern_spectra.append(np.array([psd]))  # Add channel dimension
                                
                                # Average across trials
                                if pattern_spectra:
                                    avg_spectra = np.mean(pattern_spectra, axis=0)
                                    
                                    # Store in spectra dictionary
                                    if pattern_name not in spectra:
                                        spectra[pattern_name] = []
                                    
                                    spectra[pattern_name].append(avg_spectra)
                                    print(f"  Added spectrum for {pattern_name}")
                        
                        # If no labels, treat all trials as one pattern
                        else:
                            pattern_name = 'ssvep_0'  # Default name
                            print(f"  No labels found, treating all trials as {pattern_name}")
                            
                            # Calculate PSD for each trial
                            pattern_spectra = []
                            
                            for trial in X:
                                # Ensure the data is long enough for welch
                                if trial.size >= 256:
                                    freqs, psd = signal.welch(trial, fs=250, nperseg=256)
                                else:
                                    freqs, psd = signal.welch(trial, fs=250, nperseg=min(256, trial.size))
                                pattern_spectra.append(np.array([psd]))  # Add channel dimension
                            
                            # Average across trials
                            if pattern_spectra:
                                avg_spectra = np.mean(pattern_spectra, axis=0)
                                
                                # Store in spectra dictionary
                                if pattern_name not in spectra:
                                    spectra[pattern_name] = []
                                
                                spectra[pattern_name].append(avg_spectra)
                                print(f"  Added spectrum for {pattern_name}")
                
                # Handle other cases
                else:
                    print(f"  Unsupported data structure: {type(X)}")
            
            except Exception as e:
                print(f"Error processing {data_file}: {e}")
        
        # Average across data files for each pattern
        for pattern, pattern_spectra in spectra.items():
            if pattern_spectra:
                try:
                    # Convert to numpy array and average
                    pattern_spectra_array = np.array(pattern_spectra)
                    avg_spectra = np.mean(pattern_spectra_array, axis=0)
                    
                    # Store characteristics
                    self.pattern_characteristics[pattern] = {
                        'spectra': avg_spectra,
                        'freqs': freqs,
                        'type': 'ssvep'
                    }
                    
                    # Print some statistics
                    print(f"Pattern: {pattern}")
                    print(f"  Samples: {len(pattern_spectra)}")
                    print(f"  Channels: {avg_spectra.shape[0]}")
                    print(f"  Frequency bins: {avg_spectra.shape[1]}")
                    
                    # Find dominant frequencies (SSVEP typically has peaks at specific frequencies)
                    if avg_spectra.ndim > 1:
                        channel_peak_freqs = []
                        for ch in range(avg_spectra.shape[0]):
                            max_idx = np.argmax(avg_spectra[ch])
                            channel_peak_freqs.append(freqs[max_idx])
                        mean_peak = np.mean(channel_peak_freqs)
                        print(f"  Dominant frequency: {mean_peak:.2f} Hz")
                    else:
                        max_idx = np.argmax(avg_spectra)
                        print(f"  Dominant frequency: {freqs[max_idx]:.2f} Hz")
                
                except Exception as e:
                    print(f"Error averaging spectra for {pattern}: {e}")
        
        if save_results and self.pattern_characteristics:
            output_file = os.path.join(self.analysis_dir, 'bnci_ssvep_patterns.pkl')
            
            with open(output_file, 'wb') as f:
                pickle.dump(self.pattern_characteristics, f)
            
            print(f"Saved pattern characteristics to {output_file}")
            
        self._plot_pattern_spectra('ssvep')

    def analyze_facial_expression(self, save_results=True):
        """
        Analyze the facial expression dataset to extract patterns.
        Since the OpenNeuro dataset is a zip file, we'll simulate the analysis.
        
        Parameters:
        -----------
        save_results : bool
            Whether to save the results to a pickle file
        """
        print("Analyzing Facial Expression dataset (simulation)...")
        
        # Path to the facial expression dataset
        facial_dir = os.path.join(self.data_dir, 'facial_expression')
        facial_zip = os.path.join(facial_dir, 'openneuro_facial.zip')
        
        if not os.path.exists(facial_zip):
            print(f"File not found: {facial_zip}")
            print("Will generate simulated facial expression patterns instead.")
        
        # Generate simulated data for facial expressions
        facial_patterns = {
            'neutral': {'peak_freq': 10, 'alpha_power': 0.8, 'beta_power': 0.3},
            'happy': {'peak_freq': 12, 'alpha_power': 0.6, 'beta_power': 0.7},
            'sad': {'peak_freq': 8, 'alpha_power': 1.0, 'beta_power': 0.2},
            'angry': {'peak_freq': 14, 'alpha_power': 0.4, 'beta_power': 0.9},
            'surprised': {'peak_freq': 16, 'alpha_power': 0.5, 'beta_power': 0.6}
        }
        
        # Generate simulated spectra
        freqs = np.linspace(1, 50, 100)
        n_channels = 64  # Standard EEG cap
        
        for pattern, params in facial_patterns.items():
            # Create a simulated spectrum with a peak at the characteristic frequency
            spectra = np.zeros((n_channels, len(freqs)))
            
            for ch in range(n_channels):
                # Base spectrum shape (1/f noise common in EEG)
                spectra[ch] = 10.0 / (freqs + 1) ** 1.5
                
                # Add alpha peak (8-13 Hz)
                alpha_mask = np.logical_and(freqs >= 8, freqs <= 13)
                spectra[ch, alpha_mask] += params['alpha_power'] * np.exp(
                    -(freqs[alpha_mask] - params['peak_freq']) ** 2 / 5)
                
                # Add beta activity (14-30 Hz)
                beta_mask = np.logical_and(freqs >= 14, freqs <= 30)
                spectra[ch, beta_mask] += params['beta_power'] * 0.5
            
            # Add channel variability (frontal channels show more emotion response)
            frontal_boost = np.ones(n_channels)
            frontal_boost[:20] = 1.5  # Boost frontal channels for emotions
            spectra *= frontal_boost[:, np.newaxis]
            
            # Add random variability
            spectra *= np.random.uniform(0.9, 1.1, (n_channels, 1))
            
            # Store characteristics
            self.pattern_characteristics[f'facial_{pattern}'] = {
                'spectra': spectra,
                'freqs': freqs,
                'type': 'facial_expression',
                'simulated': True
            }
            
            print(f"Pattern: facial_{pattern}")
            print(f"  Channels: {spectra.shape[0]}")
            print(f"  Frequency bins: {spectra.shape[1]}")
            print(f"  Peak frequency: {params['peak_freq']} Hz")
            print(f"  Alpha power factor: {params['alpha_power']}")
            print(f"  Beta power factor: {params['beta_power']}")
        
        if save_results and self.pattern_characteristics:
            output_file = os.path.join(self.analysis_dir, 'facial_expression_patterns.pkl')
            
            with open(output_file, 'wb') as f:
                pickle.dump(self.pattern_characteristics, f)
            
            print(f"Saved pattern characteristics to {output_file}")
            
        self._plot_pattern_spectra('facial_expression')
    
    def create_simulated_motor_patterns(self, save_results=True):
        """
        Create simulated patterns for motor imagery if real data processing fails.
        These are based on neurophysiological principles.
        
        Parameters:
        -----------
        save_results : bool
            Whether to save the results to a pickle file
        """
        print("Creating simulated motor imagery patterns...")
        
        # Define pattern parameters based on neurophysiology
        motor_patterns = {
            'rest': {
                'peak_freq': 10,
                'alpha_power': 1.0,
                'beta_power': 0.3,
                'laterality': 'bilateral'
            },
            'left_hand': {
                'peak_freq': 11,
                'alpha_power': 0.4,
                'beta_power': 0.8,
                'laterality': 'right'  # Activity in right hemisphere for left hand movement
            },
            'right_hand': {
                'peak_freq': 11,
                'alpha_power': 0.4,
                'beta_power': 0.8,
                'laterality': 'left'  # Activity in left hemisphere for right hand movement
            },
            'both_hands': {
                'peak_freq': 11,
                'alpha_power': 0.4,
                'beta_power': 0.9,
                'laterality': 'bilateral'
            },
            'feet': {
                'peak_freq': 12,
                'alpha_power': 0.5,
                'beta_power': 0.7,
                'laterality': 'central'  # Activity in central areas for feet movement
            }
        }
        
        # Generate simulated spectra
        freqs = np.linspace(1, 50, 100)
        n_channels = 64  # Standard EEG cap
        
        for pattern, params in motor_patterns.items():
            # Create a simulated spectrum with a peak at the characteristic frequency
            spectra = np.zeros((n_channels, len(freqs)))
            
            for ch in range(n_channels):
                # Base spectrum shape (1/f noise common in EEG)
                spectra[ch] = 5.0 / (freqs + 1) ** 1.5
                
                # Add alpha peak (8-13 Hz)
                alpha_mask = np.logical_and(freqs >= 8, freqs <= 13)
                spectra[ch, alpha_mask] += params['alpha_power'] * np.exp(
                    -(freqs[alpha_mask] - params['peak_freq']) ** 2 / 5)
                
                # Add beta activity (14-30 Hz)
                beta_mask = np.logical_and(freqs >= 14, freqs <= 30)
                spectra[ch, beta_mask] += params['beta_power'] * 0.5
            
            # Apply laterality (topographical distribution)
            if params['laterality'] == 'left':
                # Boost left hemisphere channels (0-31) for right hand movement
                left_boost = np.ones(n_channels)
                left_boost[:32] = 1.8
                spectra *= left_boost[:, np.newaxis]
            elif params['laterality'] == 'right':
                # Boost right hemisphere channels (32-63) for left hand movement
                right_boost = np.ones(n_channels)
                right_boost[32:] = 1.8
                spectra *= right_boost[:, np.newaxis]
            elif params['laterality'] == 'central':
                # Boost central channels (24-40) for feet movement
                central_boost = np.ones(n_channels)
                central_boost[24:40] = 1.8
                spectra *= central_boost[:, np.newaxis]
            
            # Add random variability
            spectra *= np.random.uniform(0.9, 1.1, (n_channels, 1))
            
            # Store characteristics
            self.pattern_characteristics[f'mi_{pattern}'] = {
                'spectra': spectra,
                'freqs': freqs,
                'type': 'motor_imagery',
                'simulated': True
            }
            
            print(f"Pattern: mi_{pattern}")
            print(f"  Channels: {spectra.shape[0]}")
            print(f"  Frequency bins: {spectra.shape[1]}")
            print(f"  Peak frequency: {params['peak_freq']} Hz")
            print(f"  Alpha power factor: {params['alpha_power']}")
            print(f"  Beta power factor: {params['beta_power']}")
            print(f"  Laterality: {params['laterality']}")
        
        if save_results and any(k.startswith('mi_') for k in self.pattern_characteristics):
            output_file = os.path.join(self.analysis_dir, 'motor_imagery_simulated_patterns.pkl')
            
            # Filter only motor imagery patterns
            mi_patterns = {k: v for k, v in self.pattern_characteristics.items() if k.startswith('mi_')}
            
            with open(output_file, 'wb') as f:
                pickle.dump(mi_patterns, f)
            
            print(f"Saved simulated motor imagery patterns to {output_file}")
            
        self._plot_pattern_spectra('motor_imagery')
    
    def create_simulated_ssvep_patterns(self, save_results=True):
        """
        Create simulated patterns for SSVEP if real data processing fails.
        These are based on neurophysiological principles.
        
        Parameters:
        -----------
        save_results : bool
            Whether to save the results to a pickle file
        """
        print("Creating simulated SSVEP patterns...")
        
        # SSVEP stimuli typically have specific frequencies
        # Create patterns for common SSVEP frequencies
        ssvep_frequencies = [6, 10, 12, 15, 20]
        
        # Generate simulated spectra
        freqs = np.linspace(1, 50, 100)
        n_channels = 64  # Standard EEG cap
        
        for freq in ssvep_frequencies:
            # Create a simulated spectrum with a peak at the stimulus frequency
            spectra = np.zeros((n_channels, len(freqs)))
            
            for ch in range(n_channels):
                # Base spectrum shape (1/f noise common in EEG)
                spectra[ch] = 3.0 / (freqs + 1) ** 1.5
                
                # Add peak at stimulus frequency and harmonics
                for harmonic in range(1, 4):  # Add fundamentals and harmonics
                    stimulus_freq = freq * harmonic
                    if stimulus_freq < 50:  # Only include harmonics under 50Hz
                        # Create a narrow peak at the stimulus frequency
                        peak_mask = np.abs(freqs - stimulus_freq) < 0.5  # Narrow band
                        peak_amplitude = 2.0 if harmonic == 1 else 1.0 / harmonic  # Fundamentals stronger than harmonics
                        spectra[ch, peak_mask] += peak_amplitude
            
            # Boost occipital channels (typically where SSVEP is strongest)
            occipital_boost = np.ones(n_channels)
            occipital_boost[40:] = 2.0  # Boost occipital channels
            spectra *= occipital_boost[:, np.newaxis]
            
            # Add random variability
            spectra *= np.random.uniform(0.9, 1.1, (n_channels, 1))
            
            # Store characteristics
            self.pattern_characteristics[f'ssvep_{freq}Hz'] = {
                'spectra': spectra,
                'freqs': freqs,
                'type': 'ssvep',
                'simulated': True,
                'stimulus_frequency': freq
            }
            
            print(f"Pattern: ssvep_{freq}Hz")
            print(f"  Channels: {spectra.shape[0]}")
            print(f"  Frequency bins: {spectra.shape[1]}")
            print(f"  Stimulus frequency: {freq} Hz")
        
        if save_results and any(k.startswith('ssvep_') for k in self.pattern_characteristics):
            output_file = os.path.join(self.analysis_dir, 'ssvep_simulated_patterns.pkl')
            
            # Filter only SSVEP patterns
            ssvep_patterns = {k: v for k, v in self.pattern_characteristics.items() if k.startswith('ssvep_')}
            
            with open(output_file, 'wb') as f:
                pickle.dump(ssvep_patterns, f)
            
            print(f"Saved simulated SSVEP patterns to {output_file}")
            
        self._plot_pattern_spectra('ssvep')

    def analyze_all_datasets(self, save_merged=True):
        """
        Analyze all available datasets and combine their characteristics.
        
        Parameters:
        -----------
        save_merged : bool
            Whether to save the merged results to a pickle file
        """
        # Clear existing characteristics
        self.pattern_characteristics = {}
        
        # Try to analyze real datasets first
        self.analyze_physionet_mi(save_results=False)
        self.analyze_bnci_ssvep(save_results=False)
        self.analyze_facial_expression(save_results=False)
        
        # If any dataset type is missing, create simulated patterns
        if not any(k.startswith('mi_') for k in self.pattern_characteristics):
            print("No motor imagery patterns found. Creating simulated patterns...")
            self.create_simulated_motor_patterns(save_results=False)
        
        if not any(k.startswith('ssvep_') for k in self.pattern_characteristics):
            print("No SSVEP patterns found. Creating simulated patterns...")
            self.create_simulated_ssvep_patterns(save_results=False)
        
        # Save merged results
        if save_merged and self.pattern_characteristics:
            output_file = os.path.join(self.analysis_dir, 'all_patterns.pkl')
            
            with open(output_file, 'wb') as f:
                pickle.dump(self.pattern_characteristics, f)
            
            print(f"Saved merged pattern characteristics to {output_file}")
            
            # Also save a summary CSV for easy reference
            summary_data = []
            for pattern, chars in self.pattern_characteristics.items():
                peak_freq = None
                alpha_power = None
                beta_power = None
                
                if 'freqs' in chars and 'spectra' in chars:
                    freqs = chars['freqs']
                    spectra = chars['spectra']
                    
                    # Calculate peak frequency
                    if spectra.ndim > 1:
                        # Average across channels
                        avg_spectrum = np.mean(spectra, axis=0)
                    else:
                        avg_spectrum = spectra
                    
                    peak_idx = np.argmax(avg_spectrum)
                    peak_freq = freqs[peak_idx]
                    
                    # Calculate alpha and beta power
                    alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
                    beta_idx = np.logical_and(freqs >= 14, freqs <= 30)
                    
                    if spectra.ndim > 1:
                        # Average across channels
                        alpha_power = np.mean(np.mean(spectra[:, alpha_idx], axis=1))
                        beta_power = np.mean(np.mean(spectra[:, beta_idx], axis=1))
                    else:
                        alpha_power = np.mean(spectra[alpha_idx])
                        beta_power = np.mean(spectra[beta_idx])
                
                summary_data.append({
                    'pattern': pattern,
                    'type': chars.get('type', 'unknown'),
                    'peak_freq': peak_freq,
                    'alpha_power': alpha_power,
                    'beta_power': beta_power,
                    'simulated': chars.get('simulated', False)
                })
            
            # Create summary DataFrame and save to CSV
            summary_df = pd.DataFrame(summary_data)
            summary_csv = os.path.join(self.analysis_dir, 'pattern_summary.csv')
            summary_df.to_csv(summary_csv, index=False)
            print(f"Saved pattern summary to {summary_csv}")

    def _plot_pattern_spectra(self, pattern_type):
        """
        Plot the spectra for patterns of a specific type.
        
        Parameters:
        -----------
        pattern_type : str
            Type of pattern to plot (e.g., 'motor_imagery', 'ssvep', 'facial_expression')
        """
        # Filter patterns by type
        patterns = {name: chars for name, chars in self.pattern_characteristics.items() 
                   if chars.get('type') == pattern_type}
        
        if not patterns:
            print(f"No patterns of type {pattern_type} found")
            return
        
        # Create figure
        n_patterns = len(patterns)
        fig, axes = plt.subplots(n_patterns, 1, figsize=(10, 4 * n_patterns), sharex=True)
        
        if n_patterns == 1:
            axes = [axes]
        
        # Plot each pattern
        for i, (name, chars) in enumerate(patterns.items()):
            ax = axes[i]
            
            if 'freqs' in chars and 'spectra' in chars:
                freqs = chars['freqs']
                spectra = chars['spectra']
                
                # Average across channels if needed
                if spectra.ndim > 1:
                    avg_spectrum = np.mean(spectra, axis=0)
                else:
                    avg_spectrum = spectra
                
                # Plot mean spectrum
                ax.semilogy(freqs, avg_spectrum, 'b-', linewidth=2)
                
                # Highlight frequency bands
                alpha_mask = np.logical_and(freqs >= 8, freqs <= 13)
                beta_mask = np.logical_and(freqs >= 14, freqs <= 30)
                
                ax.fill_between(freqs[alpha_mask], 0, avg_spectrum[alpha_mask], alpha=0.3, color='g', label='Alpha (8-13 Hz)')
                ax.fill_between(freqs[beta_mask], 0, avg_spectrum[beta_mask], alpha=0.3, color='r', label='Beta (14-30 Hz)')
                
                ax.set_title(f'Pattern: {name}')
                ax.set_ylabel('Power Spectral Density')
                ax.legend()
            
            # For the last subplot, add x-label
            if i == n_patterns - 1:
                ax.set_xlabel('Frequency (Hz)')
        
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.analysis_dir, f'{pattern_type}_spectra.png')
        plt.savefig(fig_path)
        plt.close()
        
        print(f"Saved {pattern_type} spectra plot to {fig_path}")

    def load_characteristics(self, filename='all_patterns.pkl'):
        """
        Load previously saved pattern characteristics.
        
        Parameters:
        -----------
        filename : str
            Name of the pickle file containing pattern characteristics
            
        Returns:
        --------
        bool
            True if loading was successful, False otherwise
        """
        file_path = os.path.join(self.analysis_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'rb') as f:
                self.pattern_characteristics = pickle.load(f)
            
            print(f"Loaded pattern characteristics from {file_path}")
            print(f"Number of patterns: {len(self.pattern_characteristics)}")
            
            # Print pattern types
            pattern_types = set(chars.get('type', 'unknown') for chars in self.pattern_characteristics.values())
            print(f"Pattern types: {pattern_types}")
            
            return True
            
        except Exception as e:
            print(f"Error loading pattern characteristics: {e}")
            return False


if __name__ == "__main__":
    # Create analyzer
    analyzer = PatternAnalyzer()
    
    # Analyze all datasets - this will try real data processing first,
    # then fall back to simulated data when needed
    analyzer.analyze_all_datasets()
    
    print("\nAnalysis complete. Pattern characteristics saved in 'analysis/' directory")