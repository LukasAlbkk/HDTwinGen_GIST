#!/usr/bin/env python3
"""
Test script to verify CBIO dataset integration with HDTwinGen
"""

import sys
sys.path.append('.')

from envs import get_env
from utils.exp_utils import to_dot_dict
import logging
import torch
import torch.nn as nn
from typing import Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def test_cbio_dataset():
    """Test the CBIO dataset loading and basic environment setup"""
    print("=" * 60)
    print("Testing CBIO Dataset Integration")
    print("=" * 60)
    
    # Configuration
    config = {
        'run': {
            'trajectories': 1000,
            'pytorch_as_optimizer': {
                'batch_size': 1,
                'learning_rate': 1e-2,
                'weight_decay': 0.0,
                'epochs': 10,
                'log_interval': 5
            },
            'optimizer': 'pytorch',
            'optimize_params': True,
            'optimization': {
                'patience': 10,
                'log_optimization': True
            }
        },
        'setup': {
            'force_recache': False, 
            'load_from_cache': True
        }
    }
    
    try:
        # Test environment creation
        print("1. Creating environment...")
        env = get_env('Dataset-CBIO', to_dot_dict(config), logger, 42)
        env.reset()
        print(f"   ✓ Environment created: {env.env_name}")
        print(f"   ✓ Train data shape: states={env.train_data[0].shape}, actions={env.train_data[1].shape}")
        print(f"   ✓ Val data shape: states={env.val_data[0].shape}, actions={env.val_data[1].shape}")
        print(f"   ✓ Test data shape: states={env.test_data[0].shape}, actions={env.test_data[1].shape}")
        
        # Test with a simple neural model
        print("\n2. Testing with a simple hybrid model...")
        
        class SimpleStateDifferential(nn.Module):
            def __init__(self):
                super(SimpleStateDifferential, self).__init__()
                # Parameters for tumor dynamics
                self.tumor_growth_rate = nn.Parameter(torch.tensor(0.1))
                self.msi_effect = nn.Parameter(torch.tensor(0.05))
                self.tmb_effect = nn.Parameter(torch.tensor(0.03))
                self.treatment_efficacy = nn.Parameter(torch.tensor(0.02))
                
                # Neural network for complex interactions
                self.interaction_nn = nn.Sequential(
                    nn.Linear(4, 8),
                    nn.ReLU(),
                    nn.Linear(8, 3)
                )

            def forward(self, tumor_size: torch.Tensor, msi_score: torch.Tensor, 
                       tmb_nonsynonymous: torch.Tensor, treatment_duration: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                # Mechanistic components
                d_tumor_size_dt = self.tumor_growth_rate * tumor_size - self.treatment_efficacy * treatment_duration * tumor_size
                d_msi_score_dt = self.msi_effect * torch.tanh(tumor_size) - 0.01 * msi_score
                d_tmb_dt = self.tmb_effect * torch.sigmoid(tumor_size) - 0.005 * tmb_nonsynonymous
                
                # Neural network residuals
                inputs = torch.cat([tumor_size.unsqueeze(1), msi_score.unsqueeze(1), 
                                  tmb_nonsynonymous.unsqueeze(1), treatment_duration.unsqueeze(1)], dim=1)
                residuals = self.interaction_nn(inputs)
                
                # Add residuals
                d_tumor_size_dt += residuals[:, 0]
                d_msi_score_dt += residuals[:, 1] 
                d_tmb_dt += residuals[:, 2]
                
                return (d_tumor_size_dt, d_msi_score_dt, d_tmb_dt)
        
        # Test model evaluation
        train_loss, val_loss, optimized_parameters, loss_per_dim_dict, test_loss = env.evaluate_simulator_code(
            StateDifferential=SimpleStateDifferential, 
            config=to_dot_dict(config), 
            logger=logger
        )
        
        print(f"   ✓ Training completed successfully")
        print(f"   ✓ Final train loss: {train_loss:.4f}")
        print(f"   ✓ Final val loss: {val_loss:.4f}")
        print(f"   ✓ Final test loss: {test_loss:.4f}")
        print(f"   ✓ Loss per dimension: {loss_per_dim_dict}")
        print(f"   ✓ Optimized parameters: {len(optimized_parameters)} parameters")
        
        print("\n" + "=" * 60)
        print("CBIO Dataset Integration: SUCCESS!")
        print("=" * 60)
        print(f"Your dataset has been successfully integrated!")
        print(f"Dataset contains {env.train_data[0].shape[1]} time steps with 3 state variables:")
        print(f"  - tumor_size")
        print(f"  - msi_score") 
        print(f"  - tmb_nonsynonymous")
        print(f"And 1 action variable:")
        print(f"  - treatment_duration")
        print(f"\nYou can now run the full HDTwinGen algorithm with:")
        print(f"  python run.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cbio_dataset()