import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
import logging
import os

# Настройка логирования
logger = logging.getLogger(__name__)


class SpectrumProcessingError(Exception):
    """Кастомное исключение для ошибок обработки спектров."""
    pass


def get_unique_values_indices(tensor: torch.Tensor) -> Dict[float, List[int]]:
    """
    Extract unique values from tensor and return their indices.
    
    Args:
        tensor: Input tensor to analyze
        
    Returns:
        Dictionary with unique values as keys and their indices as values
        
    Example:
        >>> tensor = torch.tensor([0, 1, 2, 1, 0, 3])
        >>> get_unique_values_indices(tensor)
        {1.0: [1, 3], 2.0: [2], 3.0: [5]}
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    non_zero_mask = tensor != 0
    unique_vals, indices = torch.unique(tensor[non_zero_mask], return_inverse=True)
    
    unique_dict = {}
    for i, val in enumerate(unique_vals):
        val_indices = torch.nonzero(non_zero_mask & (tensor == val), as_tuple=False)
        unique_dict[val.item()] = val_indices.view(-1).tolist()
    
    return unique_dict


def ppm_to_hz(ppm: Union[List[float], np.ndarray, torch.Tensor], 
              spec_freq: float) -> torch.Tensor:
    """
    Convert chemical shifts from ppm to Hz.
    
    Args:
        ppm: Chemical shifts in ppm
        spec_freq: Spectrometer frequency in MHz
        
    Returns:
        Tensor of chemical shifts in Hz with gradient tracking
        
    Raises:
        TypeError: If inputs are of incorrect type
        ValueError: If inputs are invalid
    """
   
    # Convert input to numpy array for consistent processing
    if isinstance(ppm, torch.Tensor):
        ppm_array = ppm.detach().numpy()
    elif isinstance(ppm, list):
        ppm_array = np.array(ppm)
    elif isinstance(ppm, np.ndarray):
        ppm_array = ppm
    else:
        raise TypeError("ppm must be list, numpy array, or torch tensor")
    
    if ppm_array.size == 0:
        raise ValueError("ppm array cannot be empty")
    
    result_array = ppm_array * spec_freq
    return torch.tensor(result_array, requires_grad=True, dtype=torch.float32)


def integrate_exp_spectrum(intensity: np.ndarray, frequency: np.ndarray) -> float:
    """
    Integrate experimental spectrum using trapezoidal rule with gap detection.
    
    Args:
        intensity: Spectrum intensity values
        frequency: Frequency values
        
    Returns:
        Integrated area under the spectrum
        
    Raises:
        ValueError: If inputs have different lengths or are empty
    """
    if len(intensity) != len(frequency):
        raise ValueError("Intensity and frequency arrays must have same length")
    
    if len(intensity) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Calculate typical spacing
    dx = 2.0 * (frequency[1] - frequency[0])
    
    start_idx = 0
    integral = 0.0
    
    for i in range(len(frequency) - 1):
        d = frequency[i + 1] - frequency[i]
        
        # Detect gap in data (discontinuity)
        if d > dx:
            stop_idx = i
            if stop_idx > start_idx:
                integral += np.trapz(intensity[start_idx:stop_idx], 
                                   frequency[start_idx:stop_idx])
            start_idx = i + 1
    
    # Integrate remaining segment
    if start_idx < len(frequency) - 1:
        integral += np.trapz(intensity[start_idx:], frequency[start_idx:])
    
    return integral


def exp_spectrum_from_csv(filename: str, 
                         frequency: float = 499.8,
                         intensity_threshold: float = 1e-4) -> torch.Tensor:
    """
    Load and preprocess experimental spectrum from CSV file.
    
    Args:
        filename: Path to CSV file containing spectrum data
        frequency: Spectrometer frequency in MHz
        intensity_threshold: Threshold for significant intensity values
        
    Returns:
        Tensor with frequency-intensity pairs [Hz, normalized_intensity]
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        SpectrumProcessingError: If spectrum processing fails
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Spectrum file not found: {filename}")
    
    if frequency <= 0:
        raise ValueError("Frequency must be positive")
    
    try:
        # Load spectrum data
        exp_spectrum = pd.read_csv(filename).values
        
        if exp_spectrum.size == 0:
            raise SpectrumProcessingError("CSV file is empty")
        
        # Identify significant region
        significant_indices = np.where(exp_spectrum[:, 1] > intensity_threshold)[0]
        
        if len(significant_indices) == 0:
            raise SpectrumProcessingError("No significant intensity values found above threshold")
        
        start_idx = significant_indices[0]
        end_idx = significant_indices[-1]
        
        # Expand region around significant values
        region_width = end_idx - start_idx
        expanded_start = max(start_idx - region_width // 2, 0)
        expanded_end = min(end_idx + region_width // 2, exp_spectrum.shape[0] - 1)
        
        # Extract region of interest
        spectrum_region = exp_spectrum[expanded_start:expanded_end + 1, :]
        
        # Convert ppm to Hz
        exp_hz = ppm_to_hz(spectrum_region[:, 0], frequency).view(-1)
        exp_intensity = spectrum_region[:, 1]
        
        # Normalize intensity
        exp_area = integrate_exp_spectrum(exp_intensity, exp_hz.detach().numpy())
        
        if exp_area == 0:
            raise SpectrumProcessingError("Cannot normalize spectrum with zero area")
        
        normalized_intensity = exp_intensity / exp_area
        
        # Create result tensor
        tensor_hz = torch.FloatTensor(exp_hz.float())
        tensor_intensity = torch.FloatTensor(normalized_intensity)
        target = torch.stack((tensor_hz, tensor_intensity), dim=-1).detach()
        
        logger.info(f"Successfully processed spectrum from {filename}, "
                   f"region: [{expanded_start}:{expanded_end}], "
                   f"area: {exp_area:.6f}")
        
        return target
        
    except pd.errors.EmptyDataError:
        raise SpectrumProcessingError(f"CSV file is empty: {filename}")
    except pd.errors.ParserError as e:
        raise SpectrumProcessingError(f"Error parsing CSV file {filename}: {e}")
    except Exception as e:
        raise SpectrumProcessingError(f"Unexpected error processing spectrum {filename}: {e}")

