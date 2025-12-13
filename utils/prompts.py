def system_prompt():
  return '''
Objective: Write fully functional PyTorch code that fills the provided class skeleton ONLY, to create an effective differential-equation-like simulator for the specified environment. No placeholders, no extra files.

Non-negotiable rules:
- Do NOT change the class name, method names, argument order, or return types.
- Fill ONLY the TODOs INSIDE the skeleton. You may define submodules and small private helper methods INSIDE the class, but must not add new public methods or change signatures.
- Code must run with batch size = 1 (and any B>=1). Therefore:
  * Do NOT use BatchNorm of any kind. If you need normalization, use LayerNorm.
- Use only PyTorch standard library (torch, torch.nn, torch.nn.functional) and typing.Tuple.
- All tensors are float32; assume inputs come as shape [B,1]. Concatenate on last dim.
- Enforce numerical stability: clamp outputs to physiologically plausible ranges.
- Treatment/control inputs MUST influence the dynamics as required by the environment.
- Prefer monotone-increasing or monotone-decreasing effects when medically sensible (see env description).
- Use LeakyReLU or ELU (avoid dead ReLU), Dropout in {0.1–0.3}, and avoid weight explosions.

Quality bar:
- Prefer a hybrid design: simple mechanistic prior + residual MLP correction.
- If using gates, ensure positivity via Softplus.
- No randomness, no training loops here, just the model class.

Self-checks inside forward:
- Assert the concatenated input has last-dim size equal to the specified count.
- Never return NaN/Inf; clamp derivatives.

Indent code with tabs. Return the complete class code body.'''

def get_system_description(env_name):
  if env_name == 'Dataset-CBIO':
    return """Digital Twin Model for GIST Cancer (CBIO Longitudinal Dataset)

GOAL:
Implement a neural ODE-like differential model for Digital Twin generation. The model outputs 2 derivatives
for STATE variables, using 12 STATIC features and 2 CONTROL inputs.

STRUCTURAL CAUSAL MODEL (SCM) - CRITICAL:
The model architecture MUST respect the Causal Graph provided below. 
You must ensure that the derivative calculations for the state variables (msi_score, tmb_nonsynonymous)
incorporate their specific PARENTS defined in the SCM.

TODO: USE THE FOLLOWING CAUSAL GRAPH (DOT FORMAT) TO DESIGN THE ARCHITECTURE:
```dot
digraph "G" {
sample_coverage [causal_mechanism="AdditiveNoiseModel using ExtraTreesRegressor", parents_during_fit="['msi_score']"];
tumor_purity [causal_mechanism="AdditiveNoiseModel using ExtraTreesRegressor", parents_during_fit="['msi_score']"];
age_at_diagnosis [causal_mechanism="AdditiveNoiseModel using ExtraTreesRegressor", parents_during_fit="['age_at_seq_reported_years', 'msi_type', 'pre_therapy_group', 'primary_site']"];
age_at_seq_reported_years [causal_mechanism="AdditiveNoiseModel using ExtraTreesRegressor", parents_during_fit="['exon_number', 'metastic_site', 'tumor_size']"];
ethnicity [causal_mechanism="Empirical Distribution", parents_during_fit="[]"];
exon_number [causal_mechanism="AdditiveNoiseModel using ExtraTreesRegressor", parents_during_fit="['mutated_genes', 'variant_type']"];
gender [causal_mechanism="Empirical Distribution", parents_during_fit="[]"];
metastic_site [causal_mechanism="Discrete AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['race', 'treatment_response']"];
mitotic_rate [causal_mechanism="AdditiveNoiseModel using ExtraTreesRegressor", parents_during_fit="['msi_score', 'pre_therapy_group', 'tmb_nonsynonymous']"];
msi_score [causal_mechanism="AdditiveNoiseModel using Pipeline", parents_during_fit="['msi_type', 'primary_site_group', 'recurrence_status']"];
msi_type [causal_mechanism="Discrete AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['gender', 'race']"];
mutated_genes [causal_mechanism="Discrete AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['primary_site']"];
os_months [causal_mechanism="AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['msi_type', 'recurrence_status', 'sample_type', 'tmb_nonsynonymous']"];
os_status [causal_mechanism="AdditiveNoiseModel using ExtraTreesRegressor", parents_during_fit="['sample_coverage', 'stage_at_diagnosis']"];
ped_ind [causal_mechanism="Discrete AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['pre_therapy_group', 'recurrence_status', 'treatment_response']"];
pre_therapy_group [causal_mechanism="Discrete AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['primary_site_group', 'recurrence_free_months', 'tumor_size']"];
primary_site [causal_mechanism="Discrete AdditiveNoiseModel using Pipeline", parents_during_fit="['primary_site_group']"];
primary_site_group [causal_mechanism="AdditiveNoiseModel using LassoCV", parents_during_fit="['ethnicity', 'gender']"];
race [causal_mechanism="Empirical Distribution", parents_during_fit="[]"];
recurrence_free_months [causal_mechanism="Discrete AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['gender', 'mutated_genes', 'recurrence_status', 'stage_at_diagnosis']"];
recurrence_status [causal_mechanism="Discrete AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['sample_type', 'treatment', 'tumor_size']"];
sample_type [causal_mechanism="Discrete AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['primary_site']"];
stage_at_diagnosis [causal_mechanism="AdditiveNoiseModel using LassoCV", parents_during_fit="['sample_type']"];
tmb_nonsynonymous [causal_mechanism="AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['ped_ind', 'tumor_purity']"];
treatment [causal_mechanism="Empirical Distribution", parents_during_fit="[]"];
treatment_duration_days [causal_mechanism="AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['race', 'treatment_response']"];
treatment_response [causal_mechanism="Discrete AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['treatment']"];
tumor_size [causal_mechanism="AdditiveNoiseModel using HistGradientBoostingRegressor", parents_during_fit="['mutated_genes', 'variant_type']"];
variant_type [causal_mechanism="Discrete AdditiveNoiseModel using LinearRegression", parents_during_fit="['primary_site']"];
sample_coverage -> os_status [key=0, weight=0.13433858510850502];
tumor_purity -> tmb_nonsynonymous [key=0, weight=0.9644347982962264];
age_at_seq_reported_years -> age_at_diagnosis [key=0, weight=57.348307510333605];
ethnicity -> primary_site_group [key=0, weight=0.0006799112359550585];
exon_number -> age_at_seq_reported_years [key=0, weight=13.122911877876799];
gender -> primary_site_group [key=0, weight="-0.00029581644640235363"];
gender -> msi_type [key=0, weight=0.0017816749174917503];
gender -> recurrence_free_months [key=0, weight=30.847849656250002];
metastic_site -> age_at_seq_reported_years [key=0, weight=15.649401918594787];
msi_score -> tumor_purity [key=0, weight=250.2487705693146];
msi_score -> mitotic_rate [key=0, weight=136.10731004067313];
msi_score -> sample_coverage [key=0, weight=36816.374001265176];
msi_type -> msi_score [key=0, weight=0.5287420764207248];
msi_type -> age_at_diagnosis [key=0, weight=0.5557737015599684];
msi_type -> os_months [key=0, weight=125.78806804832345];
mutated_genes -> exon_number [key=0, weight=368.3172074417764];
mutated_genes -> tumor_size [key=0, weight=8.665513308834896];
mutated_genes -> recurrence_free_months [key=0, weight=85.94555009177215];
ped_ind -> tmb_nonsynonymous [key=0, weight=2.03802928809007];
pre_therapy_group -> age_at_diagnosis [key=0, weight=2.322684152643949];
pre_therapy_group -> mitotic_rate [key=0, weight=88.93312960017678];
pre_therapy_group -> ped_ind [key=0, weight="5.4129856115108855e-05"];
primary_site -> variant_type [key=0, weight=0.11790170833333341];
primary_site -> age_at_diagnosis [key=0, weight=9.822207445066331];
primary_site -> sample_type [key=0, weight=0.2465248];
primary_site -> mutated_genes [key=0, weight=10.465865030172417];
primary_site_group -> msi_score [key=0, weight=0.06967331811780841];
primary_site_group -> pre_therapy_group [key=0, weight=0.030999676282051258];
primary_site_group -> primary_site [key=0, weight=0.9845537999999999];
race -> msi_type [key=0, weight=0.003491260416666664];
race -> metastic_site [key=0, weight=0.4449928712121215];
race -> treatment_duration_days [key=0, weight=29468.904980131327];
recurrence_free_months -> pre_therapy_group [key=0, weight=0.1784311180555556];
recurrence_status -> msi_score [key=0, weight=0.021425598488431355];
recurrence_status -> os_months [key=0, weight=87.95376414401305];
recurrence_status -> recurrence_free_months [key=0, weight=186.10031009090906];
recurrence_status -> ped_ind [key=0, weight=0.01965148373983739];
sample_type -> os_months [key=0, weight=257.6472591823245];
sample_type -> recurrence_status [key=0, weight=0.02598459151785715];
sample_type -> stage_at_diagnosis [key=0, weight=0.04155148301971028];
stage_at_diagnosis -> recurrence_free_months [key=0, weight=62.28977031000001];
stage_at_diagnosis -> os_status [key=0, weight=0.08604322332309948];
tmb_nonsynonymous -> os_months [key=0, weight=292.1522791074078];
tmb_nonsynonymous -> mitotic_rate [key=0, weight=160.0638937948789];
treatment -> treatment_response [key=0, weight=1.3566088055555554];
treatment -> recurrence_status [key=0, weight=0.023405822368421043];
treatment_response -> metastic_site [key=0, weight=0.511185568181818];
treatment_response -> ped_ind [key=0, weight=0.2053785];
treatment_response -> treatment_duration_days [key=0, weight=159493.07776373829];
tumor_size -> recurrence_status [key=0, weight=0.03569069537815126];
tumor_size -> pre_therapy_group [key=0, weight=0.09068181333333335];
tumor_size -> age_at_seq_reported_years [key=0, weight=35.89281028217471];
variant_type -> exon_number [key=0, weight=183.03526291396932];
variant_type -> tumor_size [key=0, weight=2.248312286017367];
}
```

INTERPRETING THE GRAPH FOR THE MODEL:
1. **msi_score** is primarily driven by:
   - `msi_type` (Weight 0.53)
   - `primary_site_group` (Weight 0.07)
   - `recurrence_status` (Weight 0.02)

2. **tmb_nonsynonymous** is primarily driven by:
   - `ped_ind` (Weight 2.04)
   - `tumor_purity` (Weight 0.96)

INPUTS (order is FIXED - matches env.py):
1. msi_score
2. tmb_nonsynonymous
3. age_at_diagnosis (Static)
4. gender_encoded (0=Female, 1=Male)
5. stage_encoded (0=Localized, 1=Metastatic)
6. primary_site_group_encoded (0=Gastric, 1=Small Bowel, 2=Other)
7. race_encoded (0=White, 1=Black, 2=Asian, 3=Other)
8. recurrence_encoded (0=No, 1=Recurrence)
9. tumor_purity (Static)
10. msi_type_encoded (0=Stable, 1=Indeterminate, 2=DNR)
11. sample_type_encoded (0=Primary, 1=Metastasis)
12. tumor_size (Static)
13. mitotic_rate (Static)
14. sample_coverage (Static)
15. treatment_duration_days (Control)
16. recurrence_free_months (Control)

OUTPUT:
(d_msi_score__dt, d_tmb_nonsynonymous__dt)
"""
  else:
    raise NotImplementedError

def get_skeleton_code(env_name):
  if env_name == 'Dataset-CBIO':
    return """class StateDifferential(nn.Module):
  def __init__(self):
    super(StateDifferential, self).__init__()
    # TODO: Fill in the code here
    #
    # SCM-INFORMED ARCHITECTURE for Digital Twin
    #
    # REQUIRED STRUCTURE based on Graph:
    # 1. Shared encoder (process all 16 inputs) -> Latent H
    #
    # 2. MSI Pathway (d_msi_score__dt):
    #    - Must combine Latent H with specific SCM parents: [msi_type_encoded, primary_site_group_encoded, recurrence_encoded]
    #
    # 3. TMB Pathway (d_tmb_nonsynonymous__dt):
    #    - Must combine Latent H with specific SCM parent: [tumor_purity]
    #
    # Use Small MLPs (e.g., 64->32) with Dropout(0.3).

  def forward(self,
              msi_score: torch.Tensor,
              tmb_nonsynonymous: torch.Tensor,
              age_at_diagnosis: torch.Tensor,
              gender_encoded: torch.Tensor,
              stage_encoded: torch.Tensor,
              primary_site_group_encoded: torch.Tensor,
              race_encoded: torch.Tensor,
              recurrence_encoded: torch.Tensor,
              tumor_purity: torch.Tensor,
              msi_type_encoded: torch.Tensor,
              sample_type_encoded: torch.Tensor,
              tumor_size: torch.Tensor,
              mitotic_rate: torch.Tensor,
              sample_coverage: torch.Tensor,
              treatment_duration_days: torch.Tensor,
              recurrence_free_months: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: Fill in the code here
    # 1. Concatenate all inputs (16 dim)
    # 2. Forward pass through shared encoder
    # 3. Forward pass through SCM-specific heads
    # 4. Clamp derivatives [-1, 1]
    return (d_msi_score__dt, d_tmb_nonsynonymous__dt)"""
  else:
    raise NotImplementedError

def first_task_prompt(env_name, generations=20, current_iteration=0):
  system_description = get_system_description(env_name)
  skeleton_code = get_skeleton_code(env_name)
  return f"""
You will get a system description to code a differential equation simulator for.

System Description:```
{system_description}
```

Skeleton Code:
```python
{skeleton_code}
```"""