"""
Inferencia de Digital Twins - Geracao de Dataset Sintetico Contrafactual

Este script gera um dataset sintetico que SUBSTITUI o original, contendo
APENAS as 34 colunas originais. Para cada paciente, cria variantes contrafactuais
alterando features especificas e usando o modelo para inferir msi_score e tmb.
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Tuple
from itertools import product

# Adicionar o diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# CONFIGURACOES
# ==============================================================================
DATASET_PATH = '/Users/lucasalbuquerque/Downloads/HDTwinGen/libs/datasets/data/cbio_longitudianal_completo.csv'
MODEL_CODE_PATH = '/Users/lucasalbuquerque/Downloads/HDTwinGen/modelos_novos_3/best_model_Dataset-CBIO_seed42.py'
MODEL_WEIGHTS_PATH = '/Users/lucasalbuquerque/Downloads/HDTwinGen/modelos_novos_3/best_model_Dataset-CBIO_seed42.pt'

# As 34 colunas originais do dataset (na ordem exata)
ORIGINAL_COLUMNS = [
    'sample_id', 'Sample coverage', 'Tumor Purity', 'age_at_diagnosis',
    'age_at_seq_reported_years', 'ethnicity', 'exon_number', 'gender',
    'metastic_site', 'mitotic_rate', 'msi_score', 'msi_type', 'mutated_genes',
    'note', 'order', 'os_months', 'os_status', 'patient_id', 'ped_ind',
    'pre_therapy_group', 'primary_site', 'primary_site_group', 'race',
    'recurrence_free_months', 'recurrence_status', 'sample_type',
    'stage_at_diagnosis', 'tmb_nonsynonymous', 'treatment', 'treatment_details',
    'treatment_duration_days', 'treatment_response', 'tumor_size', 'variant_type'
]

# Features que podem ser alteradas para contrafactual
COUNTERFACTUAL_FEATURES = {
    'gender': ['Female', 'Male'],
    'mutated_genes': ['KIT', 'TP53', 'PDGFRA', 'KIT;TP53', 'KIT;PDGFRA'],
    'treatment': ['IMATINIB', 'SUNITINIB', 'REGORAFENIB'],
    'primary_site': ['Stomach', 'Small Intestine', 'Colon', 'Rectum'],
    'sample_type': ['Primary', 'Metastasis'],
    'variant_type': ['SNP', 'DEL', 'INS'],
}


# ==============================================================================
# FUNCOES DE ENCODING (mesmo do env.py)
# ==============================================================================
def encode_dataframe(df):
    """Aplica todos os encodings necessarios ao dataframe."""
    df = df.copy()

    # Handle Missing Values
    numeric_cols = ['Tumor Purity', 'Sample coverage', 'mitotic_rate', 'tumor_size',
                    'treatment_duration_days', 'recurrence_free_months', 'age_at_diagnosis']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    df['treatment_duration_days'] = df['treatment_duration_days'].fillna(0)
    df['recurrence_free_months'] = df['recurrence_free_months'].fillna(0)
    df['msi_score'] = pd.to_numeric(df['msi_score'], errors='coerce').fillna(0)
    df['tmb_nonsynonymous'] = pd.to_numeric(df['tmb_nonsynonymous'], errors='coerce').fillna(0)

    # Tratamento: Encoding
    treatment_map = {
        'IMATINIB': 0, 'SUNITINIB': 1, 'REGORAFENIB': 2,
        'CLINICAL_TRIAL': 3, 'OTHER': 4, 'SORAFENIB': 5,
        'NILOTINIB': 6, 'PAZOPANIB': 7, 'DASATINIB': 8,
    }
    if 'treatment' in df.columns:
        df['treatment_encoded'] = df['treatment'].map(treatment_map).fillna(4)
    else:
        df['treatment_encoded'] = 0

    # Treatment response
    response_map = {'CR': 0, 'SD': 1, 'NR': 2, 'NE': 3, 'UNKNOWN': 4}
    if 'treatment_response' in df.columns:
        df['treatment_response_encoded'] = df['treatment_response'].map(response_map).fillna(4)
    else:
        df['treatment_response_encoded'] = 0

    # Mutacoes
    def check_gene(x, gene_name):
        if not isinstance(x, str):
            return 0.0
        return 1.0 if gene_name in x else 0.0

    if 'mutated_genes' in df.columns:
        df['has_kit_mutation'] = df['mutated_genes'].apply(lambda x: check_gene(x, 'KIT'))
        df['has_tp53_mutation'] = df['mutated_genes'].apply(lambda x: check_gene(x, 'TP53'))
        df['has_pdgfra_mutation'] = df['mutated_genes'].apply(lambda x: check_gene(x, 'PDGFRA'))
    else:
        df['has_kit_mutation'] = 0.0
        df['has_tp53_mutation'] = 0.0
        df['has_pdgfra_mutation'] = 0.0

    # Gender
    if 'gender' in df.columns:
        df['gender_encoded'] = df['gender'].map({'Female': 0, 'Male': 1}).fillna(0)
    else:
        df['gender_encoded'] = 0

    # Stage
    if 'stage_at_diagnosis' in df.columns:
        df['stage_encoded'] = df['stage_at_diagnosis'].apply(lambda x: 0 if x == 'Localized' else 1)
    else:
        df['stage_encoded'] = 0

    # Recurrence
    if 'recurrence_status' in df.columns:
        df['recurrence_encoded'] = df['recurrence_status'].apply(
            lambda x: 0 if str(x).lower() == 'no recurrence' else 1
        )
    else:
        df['recurrence_encoded'] = 0

    # Primary site group
    if 'primary_site_group' in df.columns:
        site_map = {'Gastric': 0, 'Small Bowel': 1}
        df['primary_site_group_encoded'] = df['primary_site_group'].map(site_map).fillna(2)
    else:
        df['primary_site_group_encoded'] = 2

    # Race
    if 'race' in df.columns:
        race_map = {'White': 0, 'Black or African American': 1, 'Asian': 2}
        df['race_encoded'] = df['race'].map(race_map).fillna(3)
    else:
        df['race_encoded'] = 3

    # MSI type
    if 'msi_type' in df.columns:
        msi_map = {'Stable': 0, 'Indeterminate': 1, 'Do not report': 2}
        df['msi_type_encoded'] = df['msi_type'].map(msi_map).fillna(0)
    else:
        df['msi_type_encoded'] = 0

    # Sample type
    if 'sample_type' in df.columns:
        df['sample_type_encoded'] = df['sample_type'].apply(lambda x: 0 if x == 'Primary' else 1)
    else:
        df['sample_type_encoded'] = 0

    return df


def get_feature_arrays(df):
    """Extrai arrays de features do dataframe encodado."""
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

    states = df[state_cols].values.astype(np.float32)
    static_features = df[static_feature_cols].values.astype(np.float32)
    controls = df[control_cols].values.astype(np.float32)

    return states, static_features, controls


# ==============================================================================
# CARREGAR MODELO
# ==============================================================================
def load_trained_model(code_path, weights_path):
    """Carrega o modelo treinado do disco (codigo + pesos)."""
    if not os.path.exists(code_path):
        print(f"AVISO: Codigo do modelo nao encontrado em {code_path}")
        print("Usando modelo dummy para demonstracao")
        return None

    if not os.path.exists(weights_path):
        print(f"AVISO: Pesos do modelo nao encontrados em {weights_path}")
        print("Usando modelo dummy para demonstracao")
        return None

    # Carregar o codigo do modelo dinamicamente
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", code_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # Instanciar o modelo
    model = model_module.StateDifferential()

    # Carregar os pesos
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Modelo carregado de: {code_path}")
    print(f"Pesos carregados de: {weights_path}")

    return model


# ==============================================================================
# INFERENCIA
# ==============================================================================
def run_inference(model, states, static_features, controls):
    """
    Executa inferencia com o modelo para prever msi_score e tmb.

    Returns:
        Tuple[float, float]: (msi_score_predicted, tmb_predicted)
    """
    device = 'cpu'

    if model is None:
        # Modo dummy - retorna estados com pequena perturbacao baseada nas features
        d_msi = np.random.normal(0, 0.05) * (1 + static_features[7])   # msi_type_encoded
        d_tmb = np.random.normal(0, 0.05) * (1 + static_features[6])   # tumor_purity
        new_msi = np.clip(states[0] + d_msi, 0, 50)
        new_tmb = np.clip(states[1] + d_tmb, 0, 100)
        return float(new_msi), float(new_tmb)

    # Modo real com modelo
    model.eval()
    with torch.no_grad():
        # Converter para tensors
        msi_score = torch.tensor([[states[0]]], dtype=torch.float32, device=device)
        tmb = torch.tensor([[states[1]]], dtype=torch.float32, device=device)

        # Static features (12)
        age = torch.tensor([[static_features[0]]], dtype=torch.float32, device=device)
        gender = torch.tensor([[static_features[1]]], dtype=torch.float32, device=device)
        stage = torch.tensor([[static_features[2]]], dtype=torch.float32, device=device)
        site_group = torch.tensor([[static_features[3]]], dtype=torch.float32, device=device)
        race = torch.tensor([[static_features[4]]], dtype=torch.float32, device=device)
        recurrence = torch.tensor([[static_features[5]]], dtype=torch.float32, device=device)
        tumor_purity = torch.tensor([[static_features[6]]], dtype=torch.float32, device=device)
        msi_type = torch.tensor([[static_features[7]]], dtype=torch.float32, device=device)
        sample_type = torch.tensor([[static_features[8]]], dtype=torch.float32, device=device)
        tumor_size = torch.tensor([[static_features[9]]], dtype=torch.float32, device=device)
        mitotic_rate = torch.tensor([[static_features[10]]], dtype=torch.float32, device=device)
        sample_coverage = torch.tensor([[static_features[11]]], dtype=torch.float32, device=device)

        # Controls
        treatment_days = torch.tensor([[controls[0]]], dtype=torch.float32, device=device)
        recurrence_months = torch.tensor([[controls[1]]], dtype=torch.float32, device=device)

        # Forward pass (16 inputs: 2 state + 12 static + 2 control)
        d_msi, d_tmb = model(
            msi_score, tmb,
            age, gender, stage, site_group, race, recurrence,
            tumor_purity, msi_type, sample_type, tumor_size, mitotic_rate, sample_coverage,
            treatment_days, recurrence_months
        )

        # Calcular novos valores
        new_msi = float((msi_score + d_msi).cpu().numpy()[0, 0])
        new_tmb = float((tmb + d_tmb).cpu().numpy()[0, 0])

        # Clamp para valores plausíveis
        new_msi = np.clip(new_msi, 0, 50)
        new_tmb = np.clip(new_tmb, 0, 100)

        return new_msi, new_tmb


# ==============================================================================
# GERACAO DE DATASET SINTETICO
# ==============================================================================
def generate_synthetic_dataset(df_original, model, output_path='synthetic_dataset_novo_2.csv'):
    """
    Gera dataset sintetico com variantes contrafactuais para cada paciente.

    O dataset gerado contem APENAS as 34 colunas originais.
    Para cada paciente, para cada combinacao de features contrafactuais,
    preserva todas as linhas originais do paciente.
    """

    synthetic_rows = []
    patients = df_original['patient_id'].unique()

    print(f"\nGerando dataset sintetico para {len(patients)} pacientes...")
    print(f"Features contrafactuais: {list(COUNTERFACTUAL_FEATURES.keys())}")

    total_variants = 0

    for i, patient_id in enumerate(patients):
        if (i + 1) % 20 == 0:
            print(f"  Processando paciente {i+1}/{len(patients)}... ({total_variants} linhas geradas)")

        # Dados originais do paciente
        patient_df = df_original[df_original['patient_id'] == patient_id].copy()
        patient_df = patient_df.sort_values('order')
        num_rows = len(patient_df)

        # Valores originais para comparacao
        original_values = {}
        for feat in COUNTERFACTUAL_FEATURES.keys():
            if feat in patient_df.columns:
                original_values[feat] = patient_df[feat].iloc[0]

        # Para cada feature contrafactual, gerar variantes
        for feature_name, possible_values in COUNTERFACTUAL_FEATURES.items():
            if feature_name not in patient_df.columns:
                continue

            original_value = original_values.get(feature_name)

            # Para cada valor alternativo
            for new_value in possible_values:
                # Pular se for o mesmo valor original
                if new_value == original_value:
                    continue

                # Criar copia contrafactual com todas as linhas do paciente
                cf_df = patient_df.copy()
                cf_df[feature_name] = new_value

                # Encodar para inferencia
                encoded_df = encode_dataframe(cf_df)

                # Para cada linha, inferir novos msi_score e tmb
                for idx in range(len(encoded_df)):
                    row_encoded = encoded_df.iloc[idx]
                    states, static_features, controls = get_feature_arrays(encoded_df.iloc[[idx]])

                    # Rodar inferencia
                    new_msi, new_tmb = run_inference(
                        model,
                        states[0],
                        static_features[0],
                        controls[0]
                    )

                    # Atualizar msi_score e tmb na linha
                    cf_df.iloc[idx, cf_df.columns.get_loc('msi_score')] = new_msi
                    cf_df.iloc[idx, cf_df.columns.get_loc('tmb_nonsynonymous')] = new_tmb

                # Adicionar apenas as 34 colunas originais
                cf_df_clean = cf_df[ORIGINAL_COLUMNS].copy()
                synthetic_rows.append(cf_df_clean)
                total_variants += num_rows

    # Concatenar todos os dados sinteticos
    if synthetic_rows:
        synthetic_df = pd.concat(synthetic_rows, ignore_index=True)

        # Salvar (APENAS colunas originais)
        synthetic_df.to_csv(output_path, index=False)
        print(f"\nDataset sintetico salvo em: {output_path}")
        print(f"Total de linhas: {len(synthetic_df)}")
        print(f"Colunas: {len(synthetic_df.columns)} (34 originais)")

        return synthetic_df

    return None


def estimate_synthetic_size(df_original):
    """Estima o tamanho do dataset sintetico que sera gerado."""
    total_variants = 0
    patients = df_original['patient_id'].unique()

    for patient_id in patients:
        patient_df = df_original[df_original['patient_id'] == patient_id]
        num_rows = len(patient_df)

        variants_per_patient = 0
        for feature_name, possible_values in COUNTERFACTUAL_FEATURES.items():
            if feature_name not in patient_df.columns:
                continue

            original_value = patient_df[feature_name].iloc[0]
            # Contar valores diferentes do original
            num_alternatives = sum(1 for v in possible_values if v != original_value)
            variants_per_patient += num_alternatives

        total_variants += variants_per_patient * num_rows

    return total_variants


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("="*70)
    print("INFERENCIA DE DIGITAL TWINS - GERACAO DE DATASET SINTETICO")
    print("="*70)
    print("\nEste script gera um dataset sintetico com APENAS as 34 colunas originais.")
    print("O dataset sintetico SUBSTITUI o original para analise contrafactual.\n")

    # 1. Carregar dataset original
    print("1. Carregando dataset original...")
    if os.path.exists(DATASET_PATH):
        df_original = pd.read_csv(DATASET_PATH)
    else:
        # Tentar path alternativo
        alt_path = '../libs/datasets/data/cbio_longitudianal_completo.csv'
        if os.path.exists(alt_path):
            df_original = pd.read_csv(alt_path)
        else:
            print("ERRO: Dataset nao encontrado!")
            return

    print(f"   Carregado: {len(df_original)} linhas, {df_original['patient_id'].nunique()} pacientes")
    print(f"   Colunas: {len(df_original.columns)}")

    # Verificar colunas
    missing_cols = set(ORIGINAL_COLUMNS) - set(df_original.columns)
    if missing_cols:
        print(f"   AVISO: Colunas faltando: {missing_cols}")

    # 2. Estimar tamanho
    print("\n2. Estimando tamanho do dataset sintetico...")
    estimated_size = estimate_synthetic_size(df_original)
    print(f"   Estimativa: ~{estimated_size} linhas")

    # 3. Carregar modelo treinado
    print("\n3. Carregando modelo treinado...")
    model = load_trained_model(MODEL_CODE_PATH, MODEL_WEIGHTS_PATH)

    # 4. Gerar dataset sintetico completo
    print("\n4. Gerando dataset sintetico completo...")
    os.makedirs('./inferencia', exist_ok=True)
    output_path = './inferencia/synthetic_dataset_novo_2.csv'
    synthetic_df = generate_synthetic_dataset(df_original, model, output_path)

    # 5. Estatisticas
    if synthetic_df is not None:
        print("\n5. Estatisticas do dataset gerado:")
        print(f"   Total de linhas: {len(synthetic_df)}")
        print(f"   Total de pacientes: {synthetic_df['patient_id'].nunique()}")
        print(f"   Colunas: {list(synthetic_df.columns)}")

        # Mostrar distribuicao por feature alterada
        print(f"\n   Amostra das primeiras linhas:")
        print(synthetic_df[['patient_id', 'gender', 'treatment', 'msi_score', 'tmb_nonsynonymous']].head(10))

    print("\n" + "="*70)
    print("GERACAO CONCLUIDA!")
    print(f"Dataset sintetico salvo em: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
