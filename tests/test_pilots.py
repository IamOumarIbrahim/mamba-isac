import pytest
import numpy as np
from data.pilots import ISACPilotAllocator
from utils.seed import set_seed

def test_comb_pilot_allocation():
    set_seed(42)
    Nc, T, spacing = 64, 16, 4
    allocator = ISACPilotAllocator(num_subcarriers=Nc, num_time_slots=T, pilot_spacing=spacing)
    
    pilot_indices, pilot_mask = allocator.get_pilot_indices()
    
    # Check number of allocated pilot subcarriers
    assert len(pilot_indices) == Nc // spacing
    assert np.all(pilot_indices == np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]))
    
    # Check pilot mask properties
    assert pilot_mask.shape == (Nc, T)
    assert np.sum(pilot_mask) == (Nc // spacing) * T
    
    # Check generated pilot grid
    pilot_grid = allocator.allocate_pilots(batch_size=2)
    assert pilot_grid.shape == (2, Nc, T)
    
    # Non-pilot subcarriers should be zero in pilot grid
    data_subcarriers = np.setdiff1d(np.arange(Nc), pilot_indices)
    assert np.all(pilot_grid[:, data_subcarriers, :] == 0.0)
