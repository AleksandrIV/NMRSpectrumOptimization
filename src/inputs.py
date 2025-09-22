import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import torch
from collections import OrderedDict
import os
import shutil
from utils import StateDictGenerator



def get_unique_values_indices(tensor):
    '''
    Takes a tensor as input and returns a dictionary containing unique values as keys and their corresponding indices as values.
    '''
    non_zero_mask = tensor != 0
    unique_vals, indices = torch.unique(tensor[non_zero_mask], return_inverse=True)
    unique_dict = {}
    for i, val in enumerate(unique_vals):
        unique_dict[val.item()] = torch.nonzero(non_zero_mask & (tensor == val)).view(-1).tolist()
    return unique_dict


def ppm_to_hz(ppm, spec_freq):
    """Given a chemical shift in ppm and spectrometer frequency in MHz, return the corresponding chemical shift in Hz."""
    return torch.tensor([d * spec_freq for d in ppm],requires_grad=True)


def integrate_exp_spectrum(I, Hz):
    dx = 2.0 * (Hz[1] - Hz[0])
    #print(dx)
    start = 0
    stop = 0
    integral = 0
    for i in range(len(Hz) - 1):
        d = Hz[i + 1] - Hz[i]
        #print("d:",d)
        if d > dx:
            stop = i
            integral = integral + np.trapz(I[start:stop], Hz[start:stop])
            #print(integral)
            start = i + 1
    integral = integral + np.trapz(I[start:i], Hz[start:i])
    return integral


