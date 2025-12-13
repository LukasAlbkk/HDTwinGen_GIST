import unittest
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from torch import multiprocessing
import os

import numpy as np
import random
from collections import defaultdict
import time
import traceback
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from functools import partial
from copy import deepcopy
from enum import Enum
from scipy.stats import truncnorm 
import atexit
import click
import datetime
import requests
import sys
import yaml
import json
import openai
from collections import deque
from scipy.optimize import minimize
import math
from typing import Tuple

def dict_to_array(constants_dict):
    keys = sorted(constants_dict.keys())
    return [constants_dict[key] for key in keys]

def array_to_dict(constants_array, template_dict):
    keys = sorted(template_dict.keys())
    return {key: value for key, value in zip(keys, constants_array)}

def generate_bounds(param_dict):
    bounds = []
    keys = sorted(param_dict.keys())
    for key in keys:
        value = param_dict[key]
        order_of_magnitude = 10 ** (int(math.log10(abs(value))) + 1)
        bounds.append((-order_of_magnitude, order_of_magnitude))
    return bounds

probabilistic = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_model_parameters(model):
    param_dict = {}
    for name, param in model.named_parameters():
        if '.' not in name:
            param_dict[name] = param.item()
    return param_dict

class DatasetEnv:
    def __init__(self):
        pass

    def reset(self, num_patients=1):
        pass
    
    def evaluate_simulator_code_wrapper(self, StateDifferential, train_data, val_data, test_data, config={}, logger=None, env_name=''):
        if config.run.optimizer == 'pytorch':
            train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss = self.evaluate_simulator_code_using_pytorch(StateDifferential, train_data, val_data, test_data, config=config, logger=logger, env_name=env_name)
        elif 'evotorch' in config.run.optimizer:
            train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss = self.evaluate_simulator_code_using_pytorch_with_neuroevolution(StateDifferential, train_data, val_data, test_data, config=config, logger=logger)
        
        loss_per_dim_dict = {}
        if env_name == 'Dataset-3DLV':
            loss_per_dim_dict = {'prey_population': loss_per_dim[0], 'intermediate_population': loss_per_dim[1], 'top_predators_population': loss_per_dim[2]}
        elif env_name == 'Dataset-HL':
            loss_per_dim_dict = {'hare_population': loss_per_dim[0], 'lynx_population': loss_per_dim[1]}
        elif env_name == 'Dataset-CBIO':
            # Order matches state_cols: ['msi_score', 'tmb_nonsynonymous']
            loss_per_dim_dict = {
                'msi_score': loss_per_dim[0], 
                'tmb_nonsynonymous': loss_per_dim[1],
                'tumor_size': 0.0 # Dummy para evitar KeyError no agents.py
            }
        return train_loss, val_loss, optimized_parameters, loss_per_dim_dict, test_loss
    
    def evaluate_simulator_code_using_pytorch(self, StateDifferential, train_data, val_data, test_data, config={}, logger=None, env_name=''):
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if hasattr(config.run, 'pytorch_as_optimizer'):
             batch_size = config.run.pytorch_as_optimizer.batch_size
        else:
             batch_size = 1
        config.run.pytorch_as_optimizer.batch_size = 1 

        f_model = StateDifferential()
        f_model.to(device)
        f_model.train()

        def prepare_tensor(data):
            if data is None: return None
            return torch.tensor(data, dtype=torch.float32, device=device)

        # Handle datasets
        if len(train_data) == 3:
            states_train, actions_train, static_train = train_data
        else:
            states_train, actions_train = train_data
            static_train = None

        states_train = prepare_tensor(states_train)
        actions_train = prepare_tensor(actions_train)
        static_train = prepare_tensor(static_train)

        if len(val_data) == 3:
            states_val, actions_val, static_val = val_data
        else:
            states_val, actions_val = val_data
            static_val = None
        
        states_val = prepare_tensor(states_val)
        actions_val = prepare_tensor(actions_val)
        static_val = prepare_tensor(static_val)

        MSE = torch.nn.MSELoss()
        optimizer = optim.Adam(f_model.parameters(), lr=config.run.pytorch_as_optimizer.learning_rate, weight_decay=config.run.pytorch_as_optimizer.weight_decay)

        def train(model, states_train_batch_i, actions_train_batch_i, static_train_batch_i=None):
            optimizer.zero_grad(True)
            pred_states = []
            pred_state = states_train_batch_i[:,0]
            for t in range(states_train_batch_i.shape[1]):
                pred_states.append(pred_state)
                if env_name == 'Dataset-3DLV':
                    prey_population, intermediate_population, top_predators_population = states_train_batch_i[:,t,0], states_train_batch_i[:,t,1], states_train_batch_i[:,t,2]
                    dx_dt = model(prey_population, intermediate_population, top_predators_population)
                elif env_name == 'Dataset-HL':
                    hare, lynx, time = states_train_batch_i[:,t,0], states_train_batch_i[:,t,1], actions_train_batch_i[:,t,0]
                    dx_dt = model(hare, lynx, time)
                elif env_name == 'Dataset-CBIO':
                    # State variables (2)
                    msi_score = states_train_batch_i[:,t,0]
                    tmb_nonsynonymous = states_train_batch_i[:,t,1]

                    # Control inputs (2)
                    treatment_duration_days = actions_train_batch_i[:,t,0]
                    recurrence_free_months = actions_train_batch_i[:,t,1]

                    # Static features (12 features)
                    if static_train_batch_i is not None:
                        age_at_diagnosis = static_train_batch_i[:,t,0]
                        gender_encoded = static_train_batch_i[:,t,1]
                        stage_encoded = static_train_batch_i[:,t,2]
                        primary_site_group_encoded = static_train_batch_i[:,t,3]
                        race_encoded = static_train_batch_i[:,t,4]
                        recurrence_encoded = static_train_batch_i[:,t,5]
                        tumor_purity = static_train_batch_i[:,t,6]
                        msi_type_encoded = static_train_batch_i[:,t,7]
                        sample_type_encoded = static_train_batch_i[:,t,8]
                        tumor_size = static_train_batch_i[:,t,9]
                        mitotic_rate = static_train_batch_i[:,t,10]
                        sample_coverage = static_train_batch_i[:,t,11]

                        dx_dt = model(
                            msi_score, tmb_nonsynonymous,
                            age_at_diagnosis, gender_encoded, stage_encoded, primary_site_group_encoded,
                            race_encoded, recurrence_encoded,
                            tumor_purity, msi_type_encoded, sample_type_encoded, tumor_size, mitotic_rate, sample_coverage,
                            treatment_duration_days, recurrence_free_months
                        )
                    else:
                        dx_dt = model(msi_score, tmb_nonsynonymous, treatment_duration_days, recurrence_free_months)
                
                dx_dt = torch.stack(dx_dt, dim=-1)
                pred_state = states_train_batch_i[:,t] + dx_dt
                
            pred_states = torch.stack(pred_states, dim=1)
            loss = MSE(pred_states, states_train_batch_i)
            loss.backward()
            optimizer.step()
            return loss.item()
        
        train_opt = train

        def compute_eval_loss(model, dataset):
            if len(dataset) == 3:
                states, actions, static = dataset
            else:
                states, actions = dataset
                static = None

            model.eval()
            with torch.no_grad():
                pred_states = []
                pred_state = states[:,0]
                for t in range(states.shape[1]):
                    pred_states.append(pred_state)
                    if env_name == 'Dataset-3DLV':
                        prey_population, intermediate_population, top_predators_population = states[:,t,0], states[:,t,1], states[:,t,2]
                        dx_dt = model(prey_population, intermediate_population, top_predators_population)
                    elif env_name == 'Dataset-HL':
                        hare, lynx, time = states[:,t,0], states[:,t,1], actions[:,t,0]
                        dx_dt = model(hare, lynx, time)
                    elif env_name == 'Dataset-CBIO':
                        msi_score = states[:,t,0]
                        tmb_nonsynonymous = states[:,t,1]
                        treatment_duration_days = actions[:,t,0]
                        recurrence_free_months = actions[:,t,1]

                        if static is not None:
                            age_at_diagnosis = static[:,t,0]
                            gender_encoded = static[:,t,1]
                            stage_encoded = static[:,t,2]
                            primary_site_group_encoded = static[:,t,3]
                            race_encoded = static[:,t,4]
                            recurrence_encoded = static[:,t,5]
                            tumor_purity = static[:,t,6]
                            msi_type_encoded = static[:,t,7]
                            sample_type_encoded = static[:,t,8]
                            tumor_size = static[:,t,9]
                            mitotic_rate = static[:,t,10]
                            sample_coverage = static[:,t,11]

                            dx_dt = model(
                                msi_score, tmb_nonsynonymous,
                                age_at_diagnosis, gender_encoded, stage_encoded, primary_site_group_encoded,
                                race_encoded, recurrence_encoded,
                                tumor_purity, msi_type_encoded, sample_type_encoded, tumor_size, mitotic_rate, sample_coverage,
                                treatment_duration_days, recurrence_free_months
                            )
                        else:
                            dx_dt = model(msi_score, tmb_nonsynonymous, treatment_duration_days, recurrence_free_months)
                    
                    dx_dt = torch.stack(dx_dt, dim=-1)
                    pred_state = states[:,t] + dx_dt
                
                pred_states = torch.stack(pred_states, dim=1)
                val_loss = MSE(pred_states, states).item()
                loss_per_dim = torch.mean(torch.square(pred_states - states), dim=(0,1)).cpu().tolist()
            model.train()
            return val_loss, loss_per_dim
                
        best_model = None
        if config.run.optimize_params:
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(config.run.pytorch_as_optimizer.epochs):
                iters = 0 
                cum_loss = 0
                t0 = time.perf_counter()
                permutation = torch.randperm(states_train.shape[0])
                for iter_i in range(int(permutation.shape[0]/config.run.pytorch_as_optimizer.batch_size)):
                    indices = permutation[iter_i*config.run.pytorch_as_optimizer.batch_size:iter_i*config.run.pytorch_as_optimizer.batch_size+config.run.pytorch_as_optimizer.batch_size]
                    states_train_batch = states_train[indices]
                    actions_train_batch = actions_train[indices] if actions_train is not None else None
                    static_train_batch = static_train[indices] if static_train is not None else None
                    
                    cum_loss += train_opt(f_model, states_train_batch, actions_train_batch, static_train_batch)
                    iters += 1
                time_taken = time.perf_counter() - t0
                
                if epoch % config.run.pytorch_as_optimizer.log_interval == 0:
                    dataset_val = (states_val, actions_val, static_val) if static_val is not None else (states_val, actions_val)
                    val_loss, _ = compute_eval_loss(f_model, dataset_val)
                    
                    print(f'[EPOCH {epoch}] Train MSE {cum_loss/iters:.4f} | Val MSE {val_loss:.4f} | {time_taken:.2f}s')
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = deepcopy(f_model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= config.run.optimization.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
        else:
            cum_loss, iters = 1, 1

        f_model.eval()
        if best_model is not None:
            f_model.load_state_dict(best_model)
            print('Loaded best model')

        dataset_val = (states_val, actions_val, static_val) if static_val is not None else (states_val, actions_val)
        val_loss, loss_per_dim = compute_eval_loss(f_model, dataset_val)
        train_loss = cum_loss/iters if iters > 0 else 0
        optimized_parameters = get_model_parameters(f_model)

        # Prepare Test Data
        if len(test_data) == 3:
            states_test, actions_test, static_test = test_data
        else:
            states_test, actions_test = test_data
            static_test = None
            
        states_test = prepare_tensor(states_test)
        actions_test = prepare_tensor(actions_test)
        static_test = prepare_tensor(static_test)

        dataset_test = (states_test, actions_test, static_test) if static_test is not None else (states_test, actions_test)
        test_loss, _ = compute_eval_loss(f_model, dataset_test)

        # Save model weights for this specific training run
        # The agent will pick the best one
        if best_model is not None and hasattr(config.run, 'dynode_saved_models_folder'):
            save_dir = config.run.dynode_saved_models_folder
            os.makedirs(save_dir, exist_ok=True)
            # Save with validation loss in filename so agent can pick best
            weights_path = f'{save_dir}/candidate_weights_val{val_loss:.6f}.pt'
            torch.save(f_model.state_dict(), weights_path)
            print(f'[Weights saved to] {weights_path}')

        return train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss
    

def load_data(config={}, seed=0, env_name='', train_ratio=0.7, val_ratio=0.15):
    if env_name == 'Dataset-3DLV':
        pandas_csv_path = './libs/datasets/data/TS_3DLV.csv'
        df = pd.read_csv(pandas_csv_path, sep=';')
        total_time_steps = df.shape[0]
        train_data = (df.iloc[:int(total_time_steps*train_ratio),1:].values[np.newaxis, :, :], None)
        val_data = (df.iloc[int(total_time_steps*train_ratio):int(total_time_steps*(train_ratio+val_ratio)),1:].values[np.newaxis, :, :], None)
        test_data = (df.iloc[int(total_time_steps*(train_ratio+val_ratio)):,1:].values[np.newaxis, :, :], None)
        
    elif env_name == 'Dataset-HL':
        pandas_csv_path = './libs/datasets/data/TS_HL.csv'
        df = pd.read_csv(pandas_csv_path, sep=';')
        total_time_steps = df.shape[0]
        train_data = (df.iloc[:int(total_time_steps*train_ratio),1:].values[np.newaxis, :, :], df.iloc[:int(total_time_steps*train_ratio),:1].values[np.newaxis, :, :])
        val_data = (df.iloc[int(total_time_steps*train_ratio):int(total_time_steps*(train_ratio+val_ratio)),1:].values[np.newaxis, :, :], df.iloc[int(total_time_steps*train_ratio):int(total_time_steps*(train_ratio+val_ratio)),:1].values[np.newaxis, :, :])
        test_data = (df.iloc[int(total_time_steps*(train_ratio+val_ratio)):,1:].values[np.newaxis, :, :], df.iloc[int(total_time_steps*(train_ratio+val_ratio)):,:1].values[np.newaxis, :, :])
        
    elif env_name == 'Dataset-CBIO':
        # PATH TO DATASET
        pandas_csv_path = './libs/datasets/data/cbio_longitudianal_completo.csv'
        if not os.path.exists(pandas_csv_path):
             pandas_csv_path = 'cbio_longitudianal_completo.csv'
             
        df = pd.read_csv(pandas_csv_path)
        print(f'[Dataset-CBIO] Loaded: {len(df)} rows, {df["patient_id"].nunique()} patients')

        # 1. Handle Missing Values
        numeric_cols = ['Tumor Purity', 'Sample coverage', 'mitotic_rate', 'tumor_size', 
                        'treatment_duration_days', 'recurrence_free_months', 'age_at_diagnosis']
        for col in numeric_cols:
             if col in df.columns:
                 df[col] = df[col].fillna(df[col].median())
        
        df['treatment_duration_days'] = df['treatment_duration_days'].fillna(0)
        df['recurrence_free_months'] = df['recurrence_free_months'].fillna(0)

        # 2. GENE PARSING & ENCODING (NOVO)
        # Criar colunas numéricas a partir de strings complexas
        
        # Tratamento: Factorize
        if 'treatment' in df.columns:
            df['treatment_encoded'] = pd.factorize(df['treatment'])[0]
        else: df['treatment_encoded'] = 0
        
        if 'treatment_response' in df.columns:
            df['treatment_response_encoded'] = pd.factorize(df['treatment_response'])[0]
        else: df['treatment_response_encoded'] = 0

        # Mutações (Parse simples de string)
        def check_gene(x, gene_name):
            if not isinstance(x, str): return 0.0
            return 1.0 if gene_name in x else 0.0

        if 'mutated_genes' in df.columns:
            df['has_kit_mutation'] = df['mutated_genes'].apply(lambda x: check_gene(x, 'KIT'))
            df['has_tp53_mutation'] = df['mutated_genes'].apply(lambda x: check_gene(x, 'TP53'))
            df['has_pdgfra_mutation'] = df['mutated_genes'].apply(lambda x: check_gene(x, 'PDGFRA'))
        else:
            df['has_kit_mutation'] = 0.0
            df['has_tp53_mutation'] = 0.0
            df['has_pdgfra_mutation'] = 0.0

        # Demais Encodings
        if 'gender' in df.columns:
            df['gender_encoded'] = df['gender'].map({'Female': 0, 'Male': 1}).fillna(0)
        else: df['gender_encoded'] = 0
        
        if 'stage_at_diagnosis' in df.columns:
            df['stage_encoded'] = df['stage_at_diagnosis'].apply(lambda x: 0 if x == 'Localized' else 1)
        else: df['stage_encoded'] = 0
        
        if 'recurrence_status' in df.columns:
            df['recurrence_encoded'] = df['recurrence_status'].apply(lambda x: 0 if str(x).lower() == 'no recurrence' else 1)
        else: df['recurrence_encoded'] = 0
        
        if 'primary_site_group' in df.columns:
            site_map = {'Gastric': 0, 'Small Bowel': 1}
            df['primary_site_group_encoded'] = df['primary_site_group'].map(site_map).fillna(2)
        else: df['primary_site_group_encoded'] = 2
        
        if 'race' in df.columns:
            race_map = {'White': 0, 'Black or African American': 1, 'Asian': 2}
            df['race_encoded'] = df['race'].map(race_map).fillna(3)
        else: df['race_encoded'] = 3
        
        if 'msi_type' in df.columns:
            msi_map = {'Stable': 0, 'Indeterminate': 1, 'Do not report': 2}
            df['msi_type_encoded'] = df['msi_type'].map(msi_map).fillna(0)
        else: df['msi_type_encoded'] = 0
        
        if 'sample_type' in df.columns:
            df['sample_type_encoded'] = df['sample_type'].apply(lambda x: 0 if x == 'Primary' else 1)
        else: df['sample_type_encoded'] = 0
        
        # Sort
        df = df.sort_values(['patient_id', 'order']).reset_index(drop=True)

        # 3. Define Feature Arrays (12 STATIC features)
        state_cols = ['msi_score', 'tmb_nonsynonymous']
        static_feature_cols = [
            'age_at_diagnosis',            # 0
            'gender_encoded',              # 1
            'stage_encoded',               # 2
            'primary_site_group_encoded',  # 3
            'race_encoded',                # 4
            'recurrence_encoded',          # 5
            'Tumor Purity',                # 6
            'msi_type_encoded',            # 7
            'sample_type_encoded',         # 8
            'tumor_size',                  # 9
            'mitotic_rate',                # 10
            'Sample coverage'              # 11
        ]
        control_cols = ['treatment_duration_days', 'recurrence_free_months']

        # Extract values
        states = df[state_cols].values[np.newaxis, :, :]
        actions = df[control_cols].values[np.newaxis, :, :]
        static_features = df[static_feature_cols].values[np.newaxis, :, :]

        # ============================================================
        # NORMALIZATION (Mantendo Z-Score)
        # ============================================================
        def normalize(data):
            flat_data = data[0] 
            mean = np.mean(flat_data, axis=0)
            std = np.std(flat_data, axis=0)
            std[std == 0] = 1.0 
            norm_data = (data - mean) / std
            return norm_data

        states = normalize(states)
        actions = normalize(actions)
        static_features = normalize(static_features)
        # ============================================================

        # Split Data
        total_rows = len(df)
        train_end = int(total_rows * train_ratio)
        val_end = int(total_rows * (train_ratio + val_ratio))

        train_data = (states[:, :train_end, :], actions[:, :train_end, :], static_features[:, :train_end, :])
        val_data = (states[:, train_end:val_end, :], actions[:, train_end:val_end, :], static_features[:, train_end:val_end, :])
        test_data = (states[:, val_end:, :], actions[:, val_end:, :], static_features[:, val_end:, :])
        
    else:
        raise NotImplementedError
    
    return train_data, val_data, test_data, ''

class TestEnvOptim(unittest.TestCase):
    def setUp(self):
        from hydra import initialize, compose
        try:
            initialize(config_path="../../config", version_base=None)
            self.config = compose(config_name="config.yaml")
        except:
             self.config = OmegaConf.create({
                 'run': {
                     'optimizer': 'pytorch',
                     'pytorch_as_optimizer': {'batch_size': 1, 'epochs': 2, 'learning_rate': 0.001, 'weight_decay': 0, 'log_interval': 1},
                     'optimize_params': True,
                     'optimization': {'patience': 5}
                 },
                 'setup': {'seed_start': 0}
             })
             
        self.env = DatasetEnv()
        # Test with CBIO by default here
        self.train_data, self.val_data, self.test_data, _ = load_data(env_name='Dataset-CBIO', train_ratio=0.7)
        self.optimizer = 'pytorch'

    def test_latest_with_pytorch_model(self):
        pass

if __name__ == "__main__":
    pass