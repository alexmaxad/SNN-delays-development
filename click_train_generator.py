import torch
import numpy as np
from scipy.signal import butter, filtfilt

def generate_single_click_train(duration=2, sampling_rate=44100, min_ici=10, max_ici=100, click_duration=2, click_value=1, filter_cutoff=2000, filter_order=8):
    """
    Generate a single random click train with specified parameters and apply high-pass filtering.
    
    Parameters:
    - duration: The nominal duration of each click train in seconds.
    - sampling_rate: The sampling rate in Hz.
    - min_ici: The minimum inter-click interval (ICI) in milliseconds.
    - max_ici: The maximum inter-click interval (ICI) in milliseconds.
    - click_duration: The duration of each click in samples.
    - click_value: The value of the click.
    - filter_cutoff: The cutoff frequency for the high-pass Butterworth filter in Hz.
    - filter_order: The order of the Butterworth filter.
    
    Returns:
    - filtered_click_train: The filtered click train as a PyTorch tensor.
    """
    # Convert ICI bounds from milliseconds to samples
    min_ici_samples = int(min_ici * sampling_rate / 1000)
    max_ici_samples = int(max_ici * sampling_rate / 1000)
    
    # Generate click train
    num_samples = int(duration * sampling_rate)
    click_train = np.zeros(num_samples)
    current_time = 0
    
    while current_time < num_samples:  # Generate for the specified duration
        ici = np.random.uniform(min_ici_samples, max_ici_samples)
        if current_time + ici < num_samples:
            click_train[int(current_time):int(current_time)+click_duration] = click_value  # Click is a rectangular pulse of specified duration
        current_time += ici
    
    # High-pass filter the click train
    nyquist = 0.5 * sampling_rate
    normal_cutoff = filter_cutoff / nyquist
    b, a = butter(filter_order, normal_cutoff, btype='high', analog=False)
    filtered_click_train = filtfilt(b, a, click_train)
    
    # Convert to PyTorch tensor
    filtered_click_train_tensor = torch.tensor(filtered_click_train, dtype=torch.float32)
    
    return filtered_click_train_tensor

# Example usage
click_train_tensor = generate_single_click_train(duration=2, min_ici=10, max_ici=100, click_duration=2, click_value=1)
print(click_train_tensor[:100])  # Print first 100 samples for inspection
