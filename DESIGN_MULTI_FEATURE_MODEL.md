# Design: Modelo Multi-Feature para HDTwinGen

## Problema Atual
O modelo atual usa apenas **3 features** de 35 disponíveis no dataset:
- `tumor_size` (state variable)
- `msi_score` (state variable)
- `tmb_nonsynonymous` (state variable)
- `treatment_duration_days` (control input)

## Novo Design Proposto

### 1. State Variables (Evoluem com Tempo)
Variáveis que mudam ao longo do tempo durante o tratamento:

- ✅ `tumor_size` (cm) - tamanho do tumor
- ✅ `msi_score` - score de instabilidade de microssatélites
- ✅ `tmb_nonsynonymous` - carga mutacional tumoral
- 🆕 `mitotic_rate` (mitoses/50 HPF) - taxa mitótica (823/870 non-null)

### 2. Static Features (Características Fixas do Paciente)
Variáveis que não mudam durante o tratamento, mas influenciam a evolução:

- 🆕 `age_at_diagnosis` (anos) - idade ao diagnóstico (870/870 non-null)
- 🆕 `stage_at_diagnosis` (categorical) - estágio clínico (870/870 non-null)
  - Encoding: Localized=0, Metastatic=1
- 🆕 `primary_site` (categorical) - localização primária do tumor (870/870 non-null)
  - Encoding one-hot: Stomach, Small Intestine, Colon, etc.
- 🆕 `treatment_type` (categorical) - tipo de tratamento (870/870 non-null)
  - Encoding: IMATINIB=0, SUNITINIB=1, REGORAFENIB=2, CLINICAL_TRIAL=3, etc.
- 🆕 `mutated_genes` (categorical) - genes mutados (870/870 non-null)
  - Encoding one-hot: KIT, TP53, RB1, SDHB, MTOR, NF1, TSC2
- 🆕 `tumor_purity` (%) - pureza tumoral da amostra (816/870 non-null)

### 3. Control Inputs
Variáveis que representam intervenções médicas:

- ✅ `treatment_duration_days` - duração acumulada do tratamento

## Arquitetura do Modelo

### Modelo Atual (Simples)
```python
Input: [tumor_size, msi_score, tmb, treatment_duration]  # 4 features
Architecture: Linear(4→64) → ReLU → Linear(64→64) → ReLU → Linear(64→3)
Output: [d_tumor/dt, d_msi/dt, d_tmb/dt]  # 3 derivatives
```

### Modelo Novo (Multi-Feature)
```python
State Variables (4):
  - tumor_size
  - msi_score
  - tmb_nonsynonymous
  - mitotic_rate

Static Features (7):
  - age_at_diagnosis
  - stage_at_diagnosis (encoded 0/1)
  - primary_site_stomach (one-hot)
  - primary_site_small_intestine (one-hot)
  - treatment_type (encoded 0-5)
  - has_kit_mutation (binary)
  - tumor_purity

Control Input (1):
  - treatment_duration_days

TOTAL INPUT SIZE: 4 + 7 + 1 = 12 features

Architecture:
  Linear(12→128) → ReLU →
  Linear(128→128) → ReLU →
  Linear(128→64) → ReLU →
  Linear(64→4)

Output: [d_tumor/dt, d_msi/dt, d_tmb/dt, d_mitotic/dt]  # 4 derivatives
```

## Alterações Necessárias nos Arquivos

### 1. `libs/datasets/env.py`

```python
# Linha ~380: Expandir state_cols
state_cols = ['tumor_size', 'msi_score', 'tmb_nonsynonymous', 'mitotic_rate']

# Adicionar processamento de static features
static_cols = ['age_at_diagnosis', 'stage_at_diagnosis', 'primary_site',
               'treatment', 'mutated_genes', 'Tumor Purity']

# Encoding de variáveis categóricas
df_clean['stage_encoded'] = (df_clean['stage_at_diagnosis'] == 'Metastatic').astype(int)

# Encoding de tratamento
treatment_map = {'IMATINIB': 0, 'SUNITINIB': 1, 'REGORAFENIB': 2,
                 'CLINICAL_TRIAL': 3, 'OTHER': 4}
df_clean['treatment_encoded'] = df_clean['treatment'].map(treatment_map).fillna(4)

# Encoding de mutações (KIT é o mais comum)
df_clean['has_kit_mutation'] = df_clean['mutated_genes'].str.contains('KIT', na=False).astype(int)

# One-hot encoding de primary_site
primary_sites = pd.get_dummies(df_clean['primary_site'], prefix='site')

# Concatenar tudo
static_features = [
    df_clean[['age_at_diagnosis', 'stage_encoded', 'treatment_encoded',
              'has_kit_mutation', 'Tumor Purity']].fillna(0),
    primary_sites
]
static_array = pd.concat(static_features, axis=1).values

# Passar para training
states = df_clean[state_cols].fillna(method='ffill').values[np.newaxis, :, :]
actions = df_clean[time_col].values[np.newaxis, :, :]
static_context = static_array[np.newaxis, :, :]  # NOVO!
```

### 2. `utils/prompts.py`

```python
def get_system_description(env_name):
    if env_name == 'Dataset-CBIO':
        return """Treatment Response Model for GIST under Multiple Targeted Therapies (CBIO Dataset) - MULTI-FEATURE VERSION

Here you must model the state differential of:
- tumor_size (cm)
- msi_score
- tmb_nonsynonymous
- mitotic_rate (mitoses per 50 HPF)

With static patient features:
- age_at_diagnosis (years)
- stage_encoded (0=Localized, 1=Metastatic)
- treatment_type (0=IMATINIB, 1=SUNITINIB, 2=REGORAFENIB, 3=CLINICAL_TRIAL, 4=OTHER)
- has_kit_mutation (0/1 binary indicator)
- tumor_purity (0-100%)
- primary_site features (one-hot encoded)

And control input:
- treatment_duration_days

The model must predict how the 4 state variables evolve based on:
1. Current state values
2. Patient characteristics (static features)
3. Treatment duration

IMPORTANT: Static features DO NOT have derivatives - they are constant per patient.
Only the 4 state variables have time derivatives.
"""
```

### 3. Forward Function Signature

```python
def forward(self,
            tumor_size, msi_score, tmb_nonsynonymous, mitotic_rate,  # State variables
            age_at_diagnosis, stage_encoded, treatment_type, has_kit_mutation, tumor_purity,  # Static
            site_stomach, site_small_intestine,  # One-hot sites
            treatment_duration) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    # Concatenate ALL inputs
    x = torch.cat([
        tumor_size, msi_score, tmb_nonsynonymous, mitotic_rate,
        age_at_diagnosis, stage_encoded, treatment_type, has_kit_mutation, tumor_purity,
        site_stomach, site_small_intestine,
        treatment_duration
    ], dim=-1)

    # Neural network
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = self.fc4(x)

    # Split outputs (4 derivatives)
    d_tumor__dt, d_msi__dt, d_tmb__dt, d_mitotic__dt = torch.split(x, 1, dim=-1)

    return d_tumor__dt, d_msi__dt, d_tmb__dt, d_mitotic__dt
```

## Benefícios da Expansão

1. **Maior Precisão**: Modelo considera características individuais do paciente
2. **Medicina Personalizada**: Previsões específicas por idade, estágio, mutações
3. **Diferenciação de Tratamentos**: Modelagem explícita de diferentes drogas
4. **Informação Genética**: Mutações influenciam resposta ao tratamento
5. **Realismo Biológico**: Mitotic rate é preditor importante de agressividade

## Impacto no Treinamento

- **Tempo**: ~20-30% maior (mais parâmetros)
- **Dados**: Mesmo dataset (870 observações)
- **Complexidade**: Aumenta de 4→3 para 12→4
- **Parâmetros**: ~10x mais parâmetros (mas ainda pequeno)

## Próximos Passos

1. ✅ Criar este documento de design
2. ⏳ Atualizar `libs/datasets/env.py` com carregamento de features
3. ⏳ Atualizar `utils/prompts.py` com nova descrição
4. ⏳ Atualizar validação para checar 4 outputs
5. ⏳ Atualizar notebook com novo modelo
6. ⏳ Usuário roda treinamento
7. ⏳ Comparar resultados: modelo simples vs multi-feature
