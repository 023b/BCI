#!/usr/bin/env python3
# BCI Project - Interface Module
# This module integrates the BCI Pattern Classifier with the EEG Simulation Controller

import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from queue import Queue

# Import our modules
from classifier import BCIPatternClassifier, RealtimeEEGClassifier

class BCIInterface:
    """
    Interface that integrates the EEG simulation controller with the pattern classifier
    to provide real-time feedback on what the user might be doing.
    """
    
    def __init__(self, master, eeg_controller, classifier=None):
        """
        Initialize the BCI interface.
        
        Parameters:
        -----------
        master : tk.Tk or tk.Frame
            The parent window or frame
        eeg_controller : EEGSimulationController
            The EEG simulation controller
        classifier : BCIPatternClassifier, optional
            The pattern classifier (if None, a new one will be created)
        """
        self.master = master
        self.eeg_controller = eeg_controller
        
        # Initialize or use the provided classifier
        if classifier is None:
            self.classifier = BCIPatternClassifier()
            self.classifier.load_models()
        else:
            self.classifier = classifier
        
        # Create real-time classifier
        self.realtime_classifier = RealtimeEEGClassifier(
            self.classifier, buffer_size=500, step_size=125, sample_rate=250.0)
        
        # Data queue for communication between threads
        self.data_queue = Queue()
        self.result_queue = Queue()
        
        # Create UI
        self.create_widgets()
        
        # Streaming control
        self.streaming = False
        self.stream_thread = None
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.process_eeg_data, daemon=True)
        self.processing_thread.start()
        
        # Update timer
        self.update_interval = 100  # ms
        self.master.after(self.update_interval, self.update_display)
    
    def create_widgets(self):
        """Create the interface widgets."""
        # Create main frame
        self.main_frame = ttk.Frame(self.master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create sidebar for controls
        self.control_frame = ttk.LabelFrame(self.main_frame, text="BCI Classifier Controls", padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create Start/Stop button
        self.start_stop_var = tk.StringVar(value="Start Classification")
        self.start_stop_button = ttk.Button(
            self.control_frame, textvariable=self.start_stop_var, 
            command=self.toggle_classification)
        self.start_stop_button.grid(row=0, column=0, sticky=tk.EW, pady=5)
        
        # Detection sensitivity
        ttk.Label(self.control_frame, text="Detection Sensitivity:").grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.sensitivity_var = tk.DoubleVar(value=0.7)
        self.sensitivity_scale = ttk.Scale(
            self.control_frame, from_=0.5, to=0.9, 
            orient=tk.HORIZONTAL, variable=self.sensitivity_var)
        self.sensitivity_scale.grid(row=2, column=0, sticky=tk.EW, pady=5)
        ttk.Label(self.control_frame, textvariable=self.sensitivity_var).grid(
            row=2, column=1, sticky=tk.W, pady=5)
        
        # Pattern types to detect
        ttk.Label(self.control_frame, text="Pattern Types:").grid(
            row=3, column=0, sticky=tk.W, pady=5)
        
        # Create checkboxes for pattern types
        self.pattern_vars = {}
        for i, pattern_type in enumerate(self.classifier.pattern_types):
            var = tk.BooleanVar(value=True)
            self.pattern_vars[pattern_type] = var
            ttk.Checkbutton(self.control_frame, text=pattern_type, 
                           variable=var).grid(row=4+i, column=0, sticky=tk.W)
        
        # Separator
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).grid(
            row=10, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # Current Interpretation Display
        ttk.Label(self.control_frame, text="Current Interpretation:").grid(
            row=11, column=0, sticky=tk.W, pady=5)
        
        self.interpretation_var = tk.StringVar(
            value="Classification not started. Press 'Start Classification' to begin.")
        self.interpretation_text = ttk.Label(
            self.control_frame, textvariable=self.interpretation_var,
            wraplength=250)
        self.interpretation_text.grid(row=12, column=0, sticky=tk.W)
        
        # Create display frame
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create visualization tabs
        self.viz_notebook = ttk.Notebook(self.display_frame)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.results_frame, text="Classification Results")
        
        # Create plot for classification results
        self.results_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.results_canvas = FigureCanvasTkAgg(self.results_fig, master=self.results_frame)
        self.results_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # EEG Data tab
        self.eeg_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.eeg_frame, text="EEG Data")
        
        # Create plot for EEG data
        self.eeg_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.eeg_canvas = FigureCanvasTkAgg(self.eeg_fig, master=self.eeg_frame)
        self.eeg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.init_plots()
    
    def init_plots(self):
        """Initialize the plots."""
        # Results plot
        self.results_ax = self.results_fig.add_subplot(111)
        self.results_ax.set_title("Classification Results")
        self.results_ax.set_ylim(0, 1)
        self.results_ax.set_ylabel("Probability")
        self.results_fig.tight_layout()
        self.results_canvas.draw()
        
        # EEG plot
        self.eeg_ax = self.eeg_fig.add_subplot(111)
        self.eeg_ax.set_title("EEG Data")
        self.eeg_ax.set_xlabel("Time (s)")
        self.eeg_ax.set_ylabel("Amplitude (μV)")
        self.eeg_fig.tight_layout()
        self.eeg_canvas.draw()
    
    def toggle_classification(self):
        """Start or stop the classification process."""
        if not self.streaming:
            # Start streaming
            self.streaming = True
            self.start_stop_var.set("Stop Classification")
            
            # Start the stream thread if not already running
            if self.stream_thread is None or not self.stream_thread.is_alive():
                self.stream_thread = threading.Thread(
                    target=self.stream_eeg_data, daemon=True)
                self.stream_thread.start()
        else:
            # Stop streaming
            self.streaming = False
            self.start_stop_var.set("Start Classification")
    
    def stream_eeg_data(self):
        """
        Thread function that gets EEG data from the simulation controller
        and puts it in the data queue for processing.
        """
        while self.streaming:
            try:
                # Get data from the simulation controller
                if hasattr(self.eeg_controller, 'eeg_data'):
                    eeg_data = self.eeg_controller.eeg_data
                    self.data_queue.put(eeg_data)
                
                # Sleep to avoid overloading the queue
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in stream thread: {e}")
                time.sleep(1)
    
    def process_eeg_data(self):
        """
        Thread function that processes EEG data from the queue and classifies it.
        """
        while True:
            try:
                # Get data from the queue
                if not self.data_queue.empty():
                    eeg_data = self.data_queue.get()
                    
                    # Only process if streaming is active
                    if self.streaming:
                        # Get the sensitivity threshold
                        sensitivity = self.sensitivity_var.get()
                        
                        # Get enabled pattern types
                        enabled_types = [pt for pt, var in self.pattern_vars.items() if var.get()]
                        
                        # Process the data
                        update_ready = self.realtime_classifier.update(eeg_data)
                        
                        if update_ready:
                            results, interpretation = self.realtime_classifier.get_latest_results()
                            
                            # Filter results by enabled types
                            filtered_results = {pt: res for pt, res in results.items() if pt in enabled_types}
                            
                            # Apply sensitivity threshold
                            filtered_results = {
                                pt: res for pt, res in filtered_results.items() 
                                if res['probability'] >= sensitivity
                            }
                            
                            # Update interpretation based on filtered results
                            if not filtered_results:
                                interpretation = "No confident predictions with current settings."
                            
                            # Put results in the queue for the main thread to display
                            self.result_queue.put((filtered_results, interpretation, eeg_data))
                
                # Sleep to yield to other threads
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Error in processing thread: {e}")
                time.sleep(1)
    
    def update_display(self):
        """Update the display with the latest classification results."""
        try:
            # Check if there are new results
            if not self.result_queue.empty():
                results, interpretation, eeg_data = self.result_queue.get()
                
                # Update interpretation text
                self.interpretation_var.set(interpretation)
                
                # Update results plot
                self.update_results_plot(results)
                
                # Update EEG plot
                self.update_eeg_plot(eeg_data)
            
            # Schedule next update
            self.master.after(self.update_interval, self.update_display)
            
        except Exception as e:
            print(f"Error updating display: {e}")
            # Ensure the update loop continues even if there's an error
            self.master.after(self.update_interval, self.update_display)
    
    def update_results_plot(self, results):
        """Update the results plot with the latest classification results."""
        self.results_ax.clear()
        
        if not results:
            self.results_ax.text(0.5, 0.5, "No results to display", 
                                ha='center', va='center')
        else:
            # Prepare data for bar chart
            pattern_types = []
            predictions = []
            probabilities = []
            colors = []
            
            for pt, result in results.items():
                pattern_types.append(pt)
                predictions.append(result['prediction'])
                probabilities.append(result['probability'])
                
                # Different color for each pattern type
                if 'motor' in pt:
                    colors.append('skyblue')
                elif 'ssvep' in pt:
                    colors.append('lightgreen')
                elif 'facial' in pt:
                    colors.append('salmon')
                else:
                    colors.append('lightgray')
            
            # Create labels for x-axis
            x_labels = [f"{pt}\n({pred})" for pt, pred in zip(pattern_types, predictions)]
            
            # Create bar chart
            bars = self.results_ax.bar(x_labels, probabilities, color=colors)
            
            # Add probability values on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                self.results_ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{probabilities[i]:.2f}', ha='center', va='bottom')
            
            # Set y-axis limit
            self.results_ax.set_ylim(0, 1.1)
            
            # Set title and labels
            self.results_ax.set_title("Classification Results")
            self.results_ax.set_ylabel("Probability")
            
            # Rotate x-axis labels for better readability
            self.results_ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        self.results_fig.tight_layout()
        self.results_canvas.draw()
    
    def update_eeg_plot(self, eeg_data):
        """Update the EEG plot with the latest data."""
        self.eeg_ax.clear()
        
        # Plot EEG data
        n_samples, n_channels = eeg_data.shape
        time_vec = np.arange(n_samples) / 250.0  # Assuming 250 Hz
        
        # Plot up to 4 channels for clarity
        plot_channels = min(4, n_channels)
        channel_offset = 100  # Vertical offset between channels
        
        for ch in range(plot_channels):
            # Add offset to separate channels visually
            offset = (plot_channels - ch) * channel_offset
            self.eeg_ax.plot(time_vec, eeg_data[:, ch] + offset, label=f"Ch {ch+1}")
        
        # Set title and labels
        self.eeg_ax.set_title("EEG Data")
        self.eeg_ax.set_xlabel("Time (s)")
        self.eeg_ax.set_ylabel("Amplitude (μV)")
        self.eeg_ax.legend(loc='upper right')
        
        # Remove y-ticks for cleaner display
        self.eeg_ax.set_yticks([])
        
        self.eeg_fig.tight_layout()
        self.eeg_canvas.draw()

# Initialization and integration code
def integrate_with_controller(controller):
    """
    Integrate the BCI interface with the EEG simulation controller.
    
    Parameters:
    -----------
    controller : EEGSimulationController
        The EEG simulation controller
        
    Returns:
    --------
    interface : BCIInterface
        The BCI interface
    """
    # Create a new Toplevel window for the BCI interface
    interface_window = tk.Toplevel()
    interface_window.title("BCI Pattern Classifier")
    interface_window.geometry("800x600")
    
    # Create classifier
    classifier = BCIPatternClassifier()
    
    # Load models if available, train if not
    if not os.path.exists(os.path.join('models', 'motor_imagery_classifier.joblib')):
        print("Training classification models...")
        classifier.train_models()
    else:
        print("Loading existing classification models...")
        classifier.load_models()
    
    # Create and return the interface
    interface = BCIInterface(interface_window, controller, classifier)
    
    # Setup the UI to work in conjunction with the controller
    interface_window.protocol("WM_DELETE_WINDOW", 
                              lambda: on_interface_close(interface_window))
    
    return interface

def on_interface_close(window):
    """Handle interface window closing."""
    window.withdraw()  # Hide instead of destroy to maintain the thread

# Extension for EEGSimulationController
def add_bci_classification(controller):
    """
    Add BCI classification capability to the EEG simulation controller.
    
    Parameters:
    -----------
    controller : EEGSimulationController
        The EEG simulation controller to extend
        
    Returns:
    --------
    interface : BCIInterface
        The created BCI interface
    """
    # Add menu option to controller window
    if hasattr(controller, 'master') and controller.master is not None:
        # Create classifier button if it doesn't exist
        if not hasattr(controller, 'classifier_button'):
            controller.classifier_button = ttk.Button(
                controller.control_frame, text="Open BCI Classifier",
                command=lambda: open_classifier(controller))
            
            # Find a suitable place to put the button
            # Assuming we have a control_frame
            row = len(controller.control_frame.grid_slaves()) + 1
            controller.classifier_button.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=5)
    
    # Create interface
    interface = integrate_with_controller(controller)
    
    # Store interface reference
    controller.bci_interface = interface
    
    return interface

def open_classifier(controller):
    """Open the classifier interface."""
    if hasattr(controller, 'bci_interface'):
        # If interface exists, show it
        controller.bci_interface.master.deiconify()
    else:
        # Create new interface
        controller.bci_interface = integrate_with_controller(controller)

# Example usage
if __name__ == "__main__":
    # This would normally be done from the main application
    import tkinter as tk
    from controller import EEGSimulationController
    
    # Create root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    # Create EEG controller
    controller_window = tk.Toplevel()
    controller_window.title("EEG Simulation Controller")
    controller = EEGSimulationController(controller_window)
    
    # Add BCI classification capability
    interface = add_bci_classification(controller)
    
    # Run the application
    root.mainloop()