#!/usr/bin/env python3
# BCI Project - Interactive EEG Simulation Controller
# This script provides a graphical interface to simulate EEG patterns

import os
import sys
import numpy as np
import pickle
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

class EEGSimulationController:
    """
    Interactive controller for simulating EEG patterns based on the analyzed data.
    Provides a GUI to select and visualize different mental states.
    """
    
    def __init__(self, master, patterns_file='analysis/all_patterns.pkl'):
        """
        Initialize the interactive EEG simulation controller.
        
        Parameters:
        -----------
        master : tk.Tk
            The main Tkinter window
        patterns_file : str
            Path to the pickle file containing pattern characteristics
        """
        self.master = master
        self.master.title("BCI Simulation - Interactive EEG Controller")
        self.master.geometry("1000x800")
        self.master.minsize(800, 600)
        
        # Load EEG patterns
        self.load_patterns(patterns_file)
        
        # Current state
        self.current_pattern = None
        
        # Create UI
        self.create_widgets()
        
        # Start with a default pattern
        if self.pattern_names:
            self.set_pattern(self.pattern_names[0])
        
        # Streaming control
        self.streaming = False
        self.stream_thread = None
        
        # Visualization update timer
        self.update_interval = 100  # ms
        self.master.after(self.update_interval, self.update_visualization)
    
    def load_patterns(self, patterns_file):
        """Load EEG patterns from the pickle file."""
        try:
            if not os.path.exists(patterns_file):
                # Try relative path
                base_dir = os.getcwd()
                patterns_file = os.path.join(base_dir, patterns_file)
            
            with open(patterns_file, 'rb') as f:
                self.patterns = pickle.load(f)
            
            print(f"Loaded {len(self.patterns)} patterns from {patterns_file}")
            
            # Group patterns by type
            self.pattern_types = {}
            for name, data in self.patterns.items():
                pattern_type = data.get('type', 'unknown')
                if pattern_type not in self.pattern_types:
                    self.pattern_types[pattern_type] = []
                self.pattern_types[pattern_type].append(name)
            
            # Create flat list of all pattern names
            self.pattern_names = list(self.patterns.keys())
            
            # Print pattern types and count
            for ptype, patterns in self.pattern_types.items():
                print(f"Pattern type: {ptype}, count: {len(patterns)}")
            
        except Exception as e:
            print(f"Error loading patterns: {e}")
            self.patterns = {}
            self.pattern_types = {}
            self.pattern_names = []
            messagebox.showerror("Error", f"Failed to load patterns: {e}")
    
    def create_widgets(self):
        """Create the GUI widgets."""
        # Create main frame with padding
        self.main_frame = ttk.Frame(self.master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for controls
        self.control_frame = ttk.LabelFrame(self.main_frame, text="EEG Pattern Controls", padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create pattern type selector
        ttk.Label(self.control_frame, text="Pattern Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.type_var = tk.StringVar()
        if self.pattern_types:
            self.type_var.set(list(self.pattern_types.keys())[0])
        
        self.type_combo = ttk.Combobox(self.control_frame, textvariable=self.type_var, 
                                       values=list(self.pattern_types.keys()), width=20)
        self.type_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.type_combo.bind("<<ComboboxSelected>>", self.on_type_selected)
        
        # Create pattern selector
        ttk.Label(self.control_frame, text="Pattern:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.pattern_var = tk.StringVar()
        if self.pattern_names:
            self.pattern_var.set(self.pattern_names[0])
        
        self.pattern_combo = ttk.Combobox(self.control_frame, textvariable=self.pattern_var, 
                                         values=self.pattern_names, width=20)
        self.pattern_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        self.pattern_combo.bind("<<ComboboxSelected>>", self.on_pattern_selected)
        
        # Create pattern buttons for quick selection
        self.pattern_button_frame = ttk.LabelFrame(self.control_frame, text="Quick Select")
        self.pattern_button_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        self.update_pattern_buttons()
        
        # Separator
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).grid(
            row=3, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # Signal parameters
        ttk.Label(self.control_frame, text="Signal Parameters:").grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Noise level
        ttk.Label(self.control_frame, text="Noise Level:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.noise_var = tk.DoubleVar(value=0.1)
        self.noise_scale = ttk.Scale(self.control_frame, from_=0.0, to=1.0, 
                                    orient=tk.HORIZONTAL, variable=self.noise_var)
        self.noise_scale.grid(row=5, column=1, sticky=tk.EW, pady=5)
        ttk.Label(self.control_frame, textvariable=self.noise_var).grid(row=5, column=2, sticky=tk.W, pady=5)
        
        # Amplitude modulation
        ttk.Label(self.control_frame, text="Amplitude:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.amplitude_var = tk.DoubleVar(value=1.0)
        self.amplitude_scale = ttk.Scale(self.control_frame, from_=0.1, to=2.0, 
                                        orient=tk.HORIZONTAL, variable=self.amplitude_var)
        self.amplitude_scale.grid(row=6, column=1, sticky=tk.EW, pady=5)
        ttk.Label(self.control_frame, textvariable=self.amplitude_var).grid(row=6, column=2, sticky=tk.W, pady=5)
        
        # Separator
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).grid(
            row=7, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # Streaming controls
        ttk.Label(self.control_frame, text="Streaming:").grid(
            row=8, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Sample rate
        ttk.Label(self.control_frame, text="Sample Rate (Hz):").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.rate_var = tk.IntVar(value=250)
        self.rate_entry = ttk.Entry(self.control_frame, textvariable=self.rate_var, width=6)
        self.rate_entry.grid(row=9, column=1, sticky=tk.W, pady=5)
        
        # Number of channels
        ttk.Label(self.control_frame, text="Channels:").grid(row=10, column=0, sticky=tk.W, pady=5)
        self.channels_var = tk.IntVar(value=8)
        self.channels_entry = ttk.Entry(self.control_frame, textvariable=self.channels_var, width=6)
        self.channels_entry.grid(row=10, column=1, sticky=tk.W, pady=5)
        
        # Start/Stop streaming button
        self.stream_var = tk.StringVar(value="Start Streaming")
        self.stream_button = ttk.Button(self.control_frame, textvariable=self.stream_var,
                                       command=self.toggle_streaming)
        self.stream_button.grid(row=11, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        # Separator
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).grid(
            row=12, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # Output controls
        ttk.Label(self.control_frame, text="Output:").grid(
            row=13, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # UDP streaming
        self.udp_var = tk.BooleanVar(value=False)
        self.udp_check = ttk.Checkbutton(self.control_frame, text="Stream via UDP", 
                                        variable=self.udp_var)
        self.udp_check.grid(row=14, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # UDP host/port
        ttk.Label(self.control_frame, text="UDP Host:").grid(row=15, column=0, sticky=tk.W, pady=5)
        self.udp_host_var = tk.StringVar(value="localhost")
        self.udp_host_entry = ttk.Entry(self.control_frame, textvariable=self.udp_host_var, width=15)
        self.udp_host_entry.grid(row=15, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(self.control_frame, text="UDP Port:").grid(row=16, column=0, sticky=tk.W, pady=5)
        self.udp_port_var = tk.IntVar(value=5005)
        self.udp_port_entry = ttk.Entry(self.control_frame, textvariable=self.udp_port_var, width=6)
        self.udp_port_entry.grid(row=16, column=1, sticky=tk.W, pady=5)
        
        # Save to file
        self.save_var = tk.BooleanVar(value=False)
        self.save_check = ttk.Checkbutton(self.control_frame, text="Save to File", 
                                         variable=self.save_var)
        self.save_check.grid(row=17, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # File settings
        ttk.Label(self.control_frame, text="File Format:").grid(row=18, column=0, sticky=tk.W, pady=5)
        self.file_format_var = tk.StringVar(value="CSV")
        self.file_format_combo = ttk.Combobox(self.control_frame, textvariable=self.file_format_var, 
                                            values=["CSV", "NPY", "EDF"], width=6)
        self.file_format_combo.grid(row=18, column=1, sticky=tk.W, pady=5)
        
        # Create visualization frame on the right
        self.viz_frame = ttk.Frame(self.main_frame)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create visualization
        self.create_visualization()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.master, textvariable=self.status_var, 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_pattern_buttons(self):
        """Update the pattern buttons based on the selected type."""
        # Clear existing buttons
        for widget in self.pattern_button_frame.winfo_children():
            widget.destroy()
        
        # Get patterns for selected type
        selected_type = self.type_var.get()
        if selected_type in self.pattern_types:
            patterns = self.pattern_types[selected_type]
            
            # Create buttons in a grid
            for i, pattern in enumerate(patterns):
                row = i // 3
                col = i % 3
                # Create a simpler name for the button (remove prefix)
                if "_" in pattern:
                    parts = pattern.split("_", 1)
                    button_text = parts[1]
                else:
                    button_text = pattern
                
                button = ttk.Button(self.pattern_button_frame, text=button_text,
                                   command=lambda p=pattern: self.set_pattern(p))
                button.grid(row=row, column=col, padx=2, pady=2, sticky=tk.EW)
        
        # Configure grid to expand buttons
        for i in range(3):
            self.pattern_button_frame.columnconfigure(i, weight=1)
    
    def create_visualization(self):
        """Create the visualization components."""
        # Create tabs for different visualizations
        self.viz_notebook = ttk.Notebook(self.viz_frame)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Spectrum tab
        self.spectrum_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.spectrum_frame, text="Spectrum")
        
        # Create spectrum figure
        self.spectrum_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.spectrum_ax = self.spectrum_fig.add_subplot(111)
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_fig, master=self.spectrum_frame)
        self.spectrum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Time series tab
        self.timeseries_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.timeseries_frame, text="Time Series")
        
        # Create time series figure
        self.timeseries_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.timeseries_ax = self.timeseries_fig.add_subplot(111)
        self.timeseries_canvas = FigureCanvasTkAgg(self.timeseries_fig, master=self.timeseries_frame)
        self.timeseries_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Topography tab
        self.topo_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.topo_frame, text="Topography")
        
        # Create topography figure
        self.topo_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.topo_ax = self.topo_fig.add_subplot(111)
        self.topo_canvas = FigureCanvasTkAgg(self.topo_fig, master=self.topo_frame)
        self.topo_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.init_plots()
    
    def init_plots(self):
        """Initialize the visualization plots."""
        # Spectrum plot
        self.spectrum_ax.clear()
        self.spectrum_ax.set_title("Spectrum")
        self.spectrum_ax.set_xlabel("Frequency (Hz)")
        self.spectrum_ax.set_ylabel("Power")
        self.spectrum_ax.set_xlim(0, 50)
        self.spectrum_ax.set_yscale('log')
        self.spectrum_line, = self.spectrum_ax.plot([], [], 'b-', linewidth=2)
        self.spectrum_fig.tight_layout()
        self.spectrum_canvas.draw()
        
        # Time series plot
        self.timeseries_ax.clear()
        self.timeseries_ax.set_title("EEG Time Series")
        self.timeseries_ax.set_xlabel("Time (s)")
        self.timeseries_ax.set_ylabel("Amplitude (μV)")
        self.timeseries_ax.set_xlim(0, 2)
        self.timeseries_ax.set_ylim(-50, 50)
        
        # Create multiple lines for different channels
        self.timeseries_lines = []
        num_channels = 8  # Default
        for i in range(num_channels):
            line, = self.timeseries_ax.plot([], [], label=f"Ch {i+1}")
            self.timeseries_lines.append(line)
        
        self.timeseries_ax.legend(loc='upper right', fontsize='small')
        self.timeseries_fig.tight_layout()
        self.timeseries_canvas.draw()
        
        # Topography plot - simplified version
        self.topo_ax.clear()
        self.topo_ax.set_title("EEG Topography")
        
        # Create a simple head outline
        circle = plt.Circle((0.5, 0.5), 0.45, fill=False, edgecolor='black')
        self.topo_ax.add_patch(circle)
        
        # Add nose
        self.topo_ax.plot([0.5, 0.5], [0.95, 1.05], 'k-')
        
        # Add ears
        self.topo_ax.plot([0.05, 0], [0.5, 0.5], 'k-')
        self.topo_ax.plot([0.95, 1], [0.5, 0.5], 'k-')
        
        # Create dots for electrodes (simplified 10-20 system)
        self.electrode_positions = {
            'Fp1': (0.35, 0.85), 'Fp2': (0.65, 0.85),
            'F7': (0.2, 0.7), 'F3': (0.35, 0.7), 'Fz': (0.5, 0.7), 'F4': (0.65, 0.7), 'F8': (0.8, 0.7),
            'T3': (0.15, 0.5), 'C3': (0.35, 0.5), 'Cz': (0.5, 0.5), 'C4': (0.65, 0.5), 'T4': (0.85, 0.5),
            'T5': (0.2, 0.3), 'P3': (0.35, 0.3), 'Pz': (0.5, 0.3), 'P4': (0.65, 0.3), 'T6': (0.8, 0.3),
            'O1': (0.35, 0.15), 'O2': (0.65, 0.15)
        }
        
        self.electrode_dots = {}
        self.electrode_labels = {}
        
        for name, pos in self.electrode_positions.items():
            dot = self.topo_ax.scatter(pos[0], pos[1], c='blue', s=50, edgecolor='black', zorder=2)
            label = self.topo_ax.text(pos[0], pos[1], name, ha='center', va='center', 
                                     fontsize=8, fontweight='bold', color='white', zorder=3)
            self.electrode_dots[name] = dot
            self.electrode_labels[name] = label
        
        self.topo_ax.set_xlim(0, 1)
        self.topo_ax.set_ylim(0, 1)
        self.topo_ax.axis('off')
        self.topo_fig.tight_layout()
        self.topo_canvas.draw()
    
    def on_type_selected(self, event):
        """Handle pattern type selection."""
        selected_type = self.type_var.get()
        if selected_type in self.pattern_types:
            # Update pattern list
            patterns = self.pattern_types[selected_type]
            self.pattern_combo['values'] = patterns
            
            # Select first pattern of this type
            if patterns:
                self.pattern_var.set(patterns[0])
                self.set_pattern(patterns[0])
            
            # Update quick selection buttons
            self.update_pattern_buttons()
    
    def on_pattern_selected(self, event):
        """Handle pattern selection."""
        selected_pattern = self.pattern_var.get()
        self.set_pattern(selected_pattern)
    
    def set_pattern(self, pattern_name):
        """Set the current pattern to the specified one."""
        if pattern_name in self.patterns:
            self.current_pattern = pattern_name
            self.pattern_var.set(pattern_name)
            
            # Update pattern type if needed
            pattern_type = self.patterns[pattern_name].get('type', 'unknown')
            if pattern_type != self.type_var.get():
                self.type_var.set(pattern_type)
                self.update_pattern_buttons()
            
            # Update status
            self.status_var.set(f"Pattern: {pattern_name}")
            
            # Update visualization
            self.update_spectrum_plot()
            self.generate_timeseries_data()
            self.update_topography_plot()
            
            print(f"Set pattern to: {pattern_name}")
    
    def update_spectrum_plot(self):
        """Update the spectrum plot with the current pattern."""
        if self.current_pattern is None or self.current_pattern not in self.patterns:
            return
        
        pattern_data = self.patterns[self.current_pattern]
        if 'freqs' not in pattern_data or 'spectra' not in pattern_data:
            return
        
        # Get spectrum data
        freqs = pattern_data['freqs']
        spectra = pattern_data['spectra']
        
        # Average across channels if needed
        if spectra.ndim > 1:
            avg_spectrum = np.mean(spectra, axis=0)
        else:
            avg_spectrum = spectra
        
        # Apply amplitude scaling
        avg_spectrum = avg_spectrum * self.amplitude_var.get()
        
        # Update plot
        self.spectrum_ax.clear()
        self.spectrum_ax.set_title(f"Spectrum: {self.current_pattern}")
        self.spectrum_ax.set_xlabel("Frequency (Hz)")
        self.spectrum_ax.set_ylabel("Power")
        self.spectrum_ax.semilogy(freqs, avg_spectrum, 'b-', linewidth=2)
        
        # Highlight frequency bands
        alpha_mask = np.logical_and(freqs >= 8, freqs <= 13)
        beta_mask = np.logical_and(freqs >= 14, freqs <= 30)
        
        self.spectrum_ax.fill_between(freqs[alpha_mask], 0, avg_spectrum[alpha_mask], 
                                     alpha=0.3, color='g', label='Alpha (8-13 Hz)')
        self.spectrum_ax.fill_between(freqs[beta_mask], 0, avg_spectrum[beta_mask], 
                                     alpha=0.3, color='r', label='Beta (14-30 Hz)')
        
        self.spectrum_ax.set_xlim(0, 50)
        self.spectrum_ax.set_ylim(0.01, np.max(avg_spectrum) * 2)
        self.spectrum_ax.legend(loc='upper right', fontsize='small')
        self.spectrum_fig.tight_layout()
        self.spectrum_canvas.draw()
    
    def generate_timeseries_data(self):
        """Generate time series data based on the current pattern."""
        if self.current_pattern is None or self.current_pattern not in self.patterns:
            return
        
        pattern_data = self.patterns[self.current_pattern]
        if 'freqs' not in pattern_data or 'spectra' not in pattern_data:
            return
        
        # Get spectrum data
        freqs = pattern_data['freqs']
        spectra = pattern_data['spectra']
        
        # Generate time series data through inverse FFT
        sample_rate = self.rate_var.get()
        num_samples = 2 * sample_rate  # 2 seconds of data
        num_channels = min(self.channels_var.get(), spectra.shape[0] if spectra.ndim > 1 else 1)
        
        # Create time vector
        t = np.arange(num_samples) / sample_rate
        
        # Generate data for each channel
        eeg_data = np.zeros((num_samples, num_channels))
        
        for ch in range(num_channels):
            # Get spectrum for this channel
            if spectra.ndim > 1:
                channel_spectrum = spectra[ch]
            else:
                channel_spectrum = spectra
            
            # Use inverse FFT approach for realistic time domain signal
            complex_spectrum = np.zeros(num_samples // 2 + 1, dtype=complex)
            
            # Map the frequencies to FFT bins
            for i, f in enumerate(freqs):
                bin_idx = int(f * num_samples / sample_rate)
                if bin_idx < len(complex_spectrum):
                    # Amplitude from spectrum, random phase
                    amplitude = np.sqrt(channel_spectrum[i])
                    phase = np.random.uniform(0, 2 * np.pi)
                    complex_spectrum[bin_idx] = amplitude * np.exp(1j * phase)
            
            # Inverse FFT to get time-domain signal
            channel_data = np.fft.irfft(complex_spectrum, num_samples)
            
            # Scale amplitude
            channel_data *= 20.0 * self.amplitude_var.get() / np.max(np.abs(channel_data))
            
            # Add noise
            channel_data += np.random.normal(0, self.noise_var.get() * 5, num_samples)
            
            # Store in EEG data array
            eeg_data[:, ch] = channel_data
        
        # Store generated data
        self.eeg_data = eeg_data
        self.eeg_time = t
        
        # Update time series plot
        self.update_timeseries_plot()
    
    def update_timeseries_plot(self):
        """Update the time series plot with the generated data."""
        if not hasattr(self, 'eeg_data') or not hasattr(self, 'eeg_time'):
            return
        
        # Update plot
        self.timeseries_ax.clear()
        self.timeseries_ax.set_title(f"EEG Time Series: {self.current_pattern}")
        self.timeseries_ax.set_xlabel("Time (s)")
        self.timeseries_ax.set_ylabel("Amplitude (μV)")
        
        # Plot each channel with an offset for clarity
        num_channels = self.eeg_data.shape[1]
        self.timeseries_lines = []
        
        for ch in range(num_channels):
            # Add offset to separate channels
            offset = 30 * (ch - num_channels/2)
            line, = self.timeseries_ax.plot(self.eeg_time, self.eeg_data[:, ch] + offset, 
                                           label=f"Ch {ch+1}")
            self.timeseries_lines.append(line)
        
        self.timeseries_ax.set_xlim(0, 2)
        self.timeseries_ax.set_ylim(-50 * num_channels/2, 50 * num_channels/2)
        self.timeseries_ax.legend(loc='upper right', fontsize='small')
        self.timeseries_fig.tight_layout()
        self.timeseries_canvas.draw()
    
    def update_topography_plot(self):
        """Update the topography plot with the current pattern."""
        if self.current_pattern is None or self.current_pattern not in self.patterns:
            return
        
        pattern_data = self.patterns[self.current_pattern]
        
        # Simulate topographical distribution based on pattern type
        electrode_values = {}
        pattern_type = pattern_data.get('type', 'unknown')
        
        # Different distributions for different pattern types
        if pattern_type == 'motor_imagery':
            # Motor imagery patterns
            if 'left_hand' in self.current_pattern:
                # Right hemisphere activation for left hand
                for name in self.electrode_positions:
                    x, y = self.electrode_positions[name]
                    if x > 0.5:  # Right hemisphere
                        electrode_values[name] = 0.8 + 0.2 * np.random.random()
                    else:
                        electrode_values[name] = 0.3 + 0.2 * np.random.random()
            elif 'right_hand' in self.current_pattern:
                # Left hemisphere activation for right hand
                for name in self.electrode_positions:
                    x, y = self.electrode_positions[name]
                    if x < 0.5:  # Left hemisphere
                        electrode_values[name] = 0.8 + 0.2 * np.random.random()
                    else:
                        electrode_values[name] = 0.3 + 0.2 * np.random.random()
            elif 'feet' in self.current_pattern:
                # Central activation for feet
                for name in self.electrode_positions:
                    x, y = self.electrode_positions[name]
                    # Central electrodes (close to midline)
                    distance_from_center = abs(x - 0.5)
                    if distance_from_center < 0.2:
                        electrode_values[name] = 0.8 + 0.2 * np.random.random()
                    else:
                        electrode_values[name] = 0.3 + 0.2 * np.random.random()
            elif 'rest' in self.current_pattern:
                # Alpha activity over posterior regions
                for name in self.electrode_positions:
                    x, y = self.electrode_positions[name]
                    if y < 0.4:  # Posterior regions
                        electrode_values[name] = 0.8 + 0.2 * np.random.random()
                    else:
                        electrode_values[name] = 0.4 + 0.2 * np.random.random()
            else:
                # Default distribution
                for name in self.electrode_positions:
                    electrode_values[name] = 0.5 + 0.5 * np.random.random()
        
        elif pattern_type == 'ssvep':
            # SSVEP patterns - occipital activation
            for name in self.electrode_positions:
                x, y = self.electrode_positions[name]
                if y < 0.3:  # Occipital regions
                    electrode_values[name] = 0.9 + 0.1 * np.random.random()
                else:
                    # Gradual decrease toward frontal regions
                    electrode_values[name] = 0.9 - 0.8 * y + 0.1 * np.random.random()
        
        elif pattern_type == 'facial_expression':
            # Facial expression patterns - frontal and temporal activation
            if 'neutral' in self.current_pattern:
                # Balanced activity
                for name in self.electrode_positions:
                    electrode_values[name] = 0.5 + 0.2 * np.random.random()
            elif 'happy' in self.current_pattern or 'surprised' in self.current_pattern:
                # More right frontal activity for positive emotions
                for name in self.electrode_positions:
                    x, y = self.electrode_positions[name]
                    if x > 0.5 and y > 0.6:  # Right frontal
                        electrode_values[name] = 0.8 + 0.2 * np.random.random()
                    else:
                        electrode_values[name] = 0.4 + 0.2 * np.random.random()
            elif 'sad' in self.current_pattern or 'angry' in self.current_pattern:
                # More left frontal activity for negative emotions
                for name in self.electrode_positions:
                    x, y = self.electrode_positions[name]
                    if x < 0.5 and y > 0.6:  # Left frontal
                        electrode_values[name] = 0.8 + 0.2 * np.random.random()
                    else:
                        electrode_values[name] = 0.4 + 0.2 * np.random.random()
            else:
                # Default distribution
                for name in self.electrode_positions:
                    electrode_values[name] = 0.5 + 0.2 * np.random.random()
        else:
            # Default distribution for unknown types
            for name in self.electrode_positions:
                electrode_values[name] = 0.5 + 0.5 * np.random.random()
        
        # Apply amplitude scaling
        for name in electrode_values:
            electrode_values[name] *= self.amplitude_var.get()
        
        # Update electrode colors
        for name, value in electrode_values.items():
            # Create color mapping: blue (low) to red (high)
            r = min(1.0, value)
            g = 0.0
            b = max(0.0, 1.0 - value)
            
            self.electrode_dots[name].set_color((r, g, b))
            self.electrode_dots[name].set_sizes([50 + 100 * value])
        
        self.topo_fig.tight_layout()
        self.topo_canvas.draw()
    
    def update_visualization(self):
        """Update all visualization components periodically."""
        # Only update time series during streaming
        if self.streaming:
            self.generate_timeseries_data()
        
        # Schedule next update
        self.master.after(self.update_interval, self.update_visualization)
    
    def toggle_streaming(self):
        """Start or stop streaming EEG data."""
        if self.streaming:
            # Stop streaming
            self.streaming = False
            self.stream_var.set("Start Streaming")
            self.status_var.set("Streaming stopped")
            
            # Enable pattern selection
            self.type_combo.config(state="normal")
            self.pattern_combo.config(state="normal")
            for widget in self.pattern_button_frame.winfo_children():
                widget.config(state="normal")
        else:
            # Start streaming
            self.streaming = True
            self.stream_var.set("Stop Streaming")
            self.status_var.set(f"Streaming {self.current_pattern} EEG data...")
            
            # Disable pattern selection during streaming
            self.type_combo.config(state="disabled")
            self.pattern_combo.config(state="disabled")
            for widget in self.pattern_button_frame.winfo_children():
                widget.config(state="disabled")
            
            # Start streaming thread if UDP is enabled
            if self.udp_var.get():
                self.start_udp_streaming()
    
    def start_udp_streaming(self):
        """Start a thread to stream data via UDP."""
        if self.stream_thread is not None and self.stream_thread.is_alive():
            return
        
        self.stream_thread = threading.Thread(target=self.udp_stream_thread, daemon=True)
        self.stream_thread.start()
    
    def udp_stream_thread(self):
        """Thread function for streaming data via UDP."""
        import socket
        import struct
        import json
        import time
        
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Get UDP settings
        host = self.udp_host_var.get()
        port = self.udp_port_var.get()
        
        # Stream while streaming is enabled
        while self.streaming and self.udp_var.get():
            try:
                # Generate new data if needed
                if not hasattr(self, 'eeg_data') or self.eeg_data.shape[1] != self.channels_var.get():
                    self.generate_timeseries_data()
                
                # Get a small chunk of data to send
                sample_rate = self.rate_var.get()
                chunk_size = sample_rate // 10  # 0.1 second chunks
                
                # Randomly select a starting point in the data
                if self.eeg_data.shape[0] > chunk_size:
                    start_idx = np.random.randint(0, self.eeg_data.shape[0] - chunk_size)
                    chunk = self.eeg_data[start_idx:start_idx+chunk_size, :]
                else:
                    chunk = self.eeg_data
                
                # Prepare data packet
                packet = {
                    'timestamp': time.time(),
                    'pattern': self.current_pattern,
                    'sample_rate': sample_rate,
                    'channels': self.channels_var.get(),
                    'data': chunk.tolist()
                }
                
                # Send as JSON
                packet_json = json.dumps(packet)
                sock.sendto(packet_json.encode(), (host, port))
                
                # Wait a bit
                time.sleep(0.1)
                
            except Exception as e:
                print(f"UDP streaming error: {e}")
                time.sleep(1)
    
    def save_eeg_data(self, filename=None):
        """Save the current EEG data to a file."""
        if not hasattr(self, 'eeg_data'):
            messagebox.showerror("Error", "No EEG data to save")
            return
        
        # Generate filename if not provided
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            pattern = self.current_pattern.replace(" ", "_")
            file_format = self.file_format_var.get().lower()
            filename = f"eeg_{pattern}_{timestamp}.{file_format}"
        
        try:
            # Save based on format
            format_lower = self.file_format_var.get().lower()
            
            if format_lower == 'csv':
                # Save as CSV
                import pandas as pd
                
                # Create DataFrame
                columns = [f"channel_{i+1}" for i in range(self.eeg_data.shape[1])]
                df = pd.DataFrame(self.eeg_data, columns=columns)
                
                # Add time column
                df.insert(0, 'time', self.eeg_time)
                
                # Save to CSV
                df.to_csv(filename, index=False)
                
            elif format_lower == 'npy':
                # Save as NumPy array
                np.save(filename, self.eeg_data)
                
            elif format_lower == 'edf':
                # Save as EDF (if MNE is available)
                try:
                    import mne
                    
                    # Create info object
                    ch_names = [f"EEG{i+1}" for i in range(self.eeg_data.shape[1])]
                    ch_types = ['eeg'] * self.eeg_data.shape[1]
                    info = mne.create_info(ch_names=ch_names, sfreq=self.rate_var.get(), ch_types=ch_types)
                    
                    # Create raw object
                    raw = mne.io.RawArray(self.eeg_data.T, info)
                    
                    # Save to EDF
                    raw.export(filename)
                    
                except ImportError:
                    messagebox.showerror("Error", "MNE-Python is required for EDF export")
                    return
            else:
                # Default to NumPy
                np.save(filename, self.eeg_data)
            
            self.status_var.set(f"Saved EEG data to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save EEG data: {e}")
    
    def run(self):
        """Run the main application loop."""
        self.master.mainloop()


def main():
    root = tk.Tk()
    app = EEGSimulationController(root)
    app.run()

if __name__ == "__main__":
    main()