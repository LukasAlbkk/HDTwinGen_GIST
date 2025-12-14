# 🔄 Integração DoWhy + HDTwinGen: Evolução Temporal Contrafactual

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Fluxo do Pipeline](#fluxo-do-pipeline)
3. [Modificações Necessárias](#modificações-necessárias)
4. [Exemplos de Implementação](#exemplos-de-implementação)
5. [Configuração](#configuração)
6. [Resumo Técnico](#resumo-técnico)

---

## 🎯 Visão Geral

Este documento descreve como integrar **DoWhy** (biblioteca de inferência causal) ao framework **HDTwinGen** para criar um pipeline que:

1. **Inicia com dataset estático** (apenas tempo t0)
2. **Framework treina** modelo inicial
3. **DoWhy gera cenários contrafactuais** criando evolução temporal
4. **Framework treina** com dados expandidos (agora longitudinais)
5. **Repete** o ciclo, expandindo progressivamente as sequências temporais

### Vantagens

✅ Dataset inicial pode ser **estático** (não precisa ter dados longitudinais)  
✅ DoWhy cria evolução temporal de forma **causalmente fundamentada**  
✅ Framework aprende **progressivamente** com sequências cada vez mais longas  
✅ Modelo melhora ao ver **trajetórias temporais** mais complexas  
✅ Altamente **configurável** (frequência de chamadas DoWhy, variáveis, etc.)

---

## 🔄 Fluxo do Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ ITERAÇÃO 0: Dataset Inicial (pode ser estático!)           │
├─────────────────────────────────────────────────────────────┤
│ Paciente X: [t0] tumor_size=5cm, treatment=0mg            │
│ Paciente Y: [t0] tumor_size=3cm, treatment=0mg            │
│ Status: 100 pacientes × 1 timestep                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │ Framework treina modelo         │
        │ (Geração 0 do NSDT)            │
        │ Val Loss: 0.450                │
        └─────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ DoWhy: Inferência Causal + Cenários Contrafactuais         │
├─────────────────────────────────────────────────────────────┤
│ • Identifica relações causais no dataset                   │
│ • Gera intervenções: "E se tratamento fosse 30% maior?"   │
│ • Estima outcomes contrafactuais baseado em causalidade    │
├─────────────────────────────────────────────────────────────┤
│ Paciente X: [t0] original                                   │
│             [t1] tumor_size=4.5cm, treatment=400mg ← DoWhy │
│ Paciente Y: [t0] original                                   │
│             [t1] tumor_size=2.8cm, treatment=200mg ← DoWhy │
│ Status: 100 pacientes × 2 timesteps                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ ITERAÇÃO 1: Dataset Expandido (agora longitudinal!)        │
├─────────────────────────────────────────────────────────────┤
│ Framework treina com sequências temporais t0→t1             │
│ (Gerações 1-3 do NSDT)                                     │
│ Val Loss: 0.380 ← Melhorou!                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ DoWhy: Nova Expansão Temporal                               │
├─────────────────────────────────────────────────────────────┤
│ • Usa trajetória t0→t1 como contexto                       │
│ • Gera t2 baseado na evolução observada                    │
├─────────────────────────────────────────────────────────────┤
│ Paciente X: [t0] original                                   │
│             [t1] DoWhy (iteração 1)                        │
│             [t2] tumor_size=4cm, treatment=600mg ← DoWhy   │
│ Status: 100 pacientes × 3 timesteps                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
            (Loop continua até convergência...)
```

### Exemplo Numérico Completo

```python
# ITERAÇÃO 0 (dataset estático inicial)
Dataset inicial: 100 pacientes × 1 timestep (t0)
    ↓
Framework treina modelo (Geração 0)
    Val Loss: 0.450
    ↓
DoWhy gera t1 para cada paciente
    Intervenção: treatment += 30%
    Estima: tumor_size, msi_score, tmb
    ↓
Dataset evoluído: 100 pacientes × 2 timesteps (t0→t1)

# ITERAÇÃO 1 (agora com sequências temporais!)
Framework treina modelo (Gerações 1-3)
    Val Loss: 0.380 ← 15.6% melhor!
    ↓
DoWhy gera t2 baseado em t0→t1
    Intervenção: treatment += 40%
    Contexto: usa tendência t0→t1
    ↓
Dataset evoluído: 100 pacientes × 3 timesteps (t0→t1→t2)

# ITERAÇÃO 2
Framework treina modelo (Gerações 4-6)
    Val Loss: 0.325 ← 14.5% melhor!
    ↓
DoWhy gera t3 baseado em t0→t1→t2
    ↓
Dataset evoluído: 100 pacientes × 4 timesteps (t0→t1→t2→t3)

...e assim por diante
```

---

## 🔧 Modificações Necessárias
1. Modificar load_data para aceitar dataset inicial estático
Arquivo: env.py

def load_data_with_dowhy_evolution(initial_csv_path, iteration=0, evolved_csv_path=None):
    """
    Carrega dados que podem ser estáticos inicialmente e evoluem com DoWhy.
    
    Args:
        initial_csv_path: CSV inicial (pode ter apenas t0)
        iteration: Número da iteração atual
        evolved_csv_path: CSV com dados evoluídos pelo DoWhy (se iteration > 0)
    """
    
    if iteration == 0:
        # PRIMEIRA ITERAÇÃO: Dados estáticos
        df = pd.read_csv(initial_csv_path)
        # df tem: patient_id, tumor_size, treatment, age, gender, ...
        # Mas NÃO tem sequências temporais ainda!
        
        # Adiciona coluna temporal
        df['time_step'] = 0  # Todos são t0
        df['order'] = 0
        
    else:
        # ITERAÇÕES SEGUINTES: Dados evoluídos pelo DoWhy
        df = pd.read_csv(evolved_csv_path)
        # df agora tem:
        # patient_id, time_step, tumor_size, treatment, ...
        # Com múltiplos timesteps por paciente!
        
    # Ordena temporalmente
    df = df.sort_values(['patient_id', 'time_step']).reset_index(drop=True)
    
    # Resto do processamento igual...
    state_cols = ['msi_score', 'tmb_nonsynonymous']
    # ...
    
    return train_data, val_data, test_data, ''

2. Criar módulo de integração com DoWhy
Arquivo: dowhy_integration.py (criar novo)

import pandas as pd
import dowhy
from dowhy import CausalModel

def generate_counterfactual_evolution(
    current_df,
    iteration,
    treatment_vars=['treatment_duration_days'],
    outcome_vars=['tumor_size', 'msi_score', 'tmb_nonsynonymous'],
    static_vars=['age_at_diagnosis', 'gender_encoded']
):
    """
    Usa DoWhy para gerar próximo timestep contrafactual.
    
    Args:
        current_df: DataFrame atual (pode ter t0, t0→t1, etc.)
        iteration: Iteração atual
        
    Returns:
        evolved_df: DataFrame com novo timestep adicionado
    """
    
    # 1. Para cada paciente, pega último timestep
    latest_timesteps = current_df.groupby('patient_id').tail(1).copy()
    
    # 2. Cria modelo causal com DoWhy
    causal_graph = """
    digraph {
        treatment_duration_days -> tumor_size;
        treatment_duration_days -> msi_score;
        treatment_duration_days -> tmb_nonsynonymous;
        age_at_diagnosis -> tumor_size;
        stage_encoded -> tumor_size;
        tumor_size -> tmb_nonsynonymous;
    }
    """
    
    model = CausalModel(
        data=latest_timesteps,
        treatment=treatment_vars,
        outcome=outcome_vars,
        graph=causal_graph
    )
    
    # 3. Identifica efeito causal
    identified_estimand = model.identify_effect()
    
    # 4. Gera cenários contrafactuais
    new_timesteps = []
    
    for _, patient_row in latest_timesteps.iterrows():
        patient_id = patient_row['patient_id']
        current_time = patient_row['time_step']
        
        # Cenário: "E se aumentássemos tratamento em 30%?"
        new_treatment = patient_row['treatment_duration_days'] * 1.3
        
        # DoWhy estima outcomes contrafactuais
        counterfactual = model.refute_estimate(
            identified_estimand,
            method_name="add_unobserved_common_cause",
            # Intervenção: mudar tratamento
            confounders_effect_on_treatment="binary_flip",
            confounders_effect_on_outcome="linear"
        )
        
        # Cria novo registro temporal
        new_row = patient_row.copy()
        new_row['time_step'] = current_time + 1
        new_row['order'] = current_time + 1
        new_row['treatment_duration_days'] = new_treatment
        
        # Aplica efeitos causais estimados
        # (simplificado - na prática usar DoWhy counterfactual)
        new_row['tumor_size'] *= 0.95  # Reduz 5% com tratamento
        new_row['msi_score'] *= 1.02   # Aumenta 2%
        
        new_timesteps.append(new_row)
    
    # 5. Combina dados originais + novos timesteps
    new_df = pd.DataFrame(new_timesteps)
    evolved_df = pd.concat([current_df, new_df], ignore_index=True)
    
    # 6. Ordena temporalmente
    evolved_df = evolved_df.sort_values(['patient_id', 'time_step'])
    
    return evolved_df

3. Modificar loop principal em agents.py
Localização: agents.py - método NSDT._run (linha ~380)

def _run(self, state=''):
    self.messages = [self.system_message]
    
    # ═══════════════════════════════════════════════════════════
    # ITERAÇÃO 0: Dataset Inicial (pode ser estático)
    # ═══════════════════════════════════════════════════════════
    
    generation_id = 0
    iteration_id = 0  # ← NOVO: Controla iterações DoWhy
    
    # Treina geração 0 com dados iniciais
    initial_prompt = prompts.first_task_prompt(env_name=self.env.env_name, ...)
    self.messages.append({"role": "user", "content": initial_prompt, ...})
    
    response_messages = self.get_llm_response_with_retries(self.messages, n=1)
    generation_dicts = []
    
    for response_message in response_messages:
        function_return_message, code_dict, has_success = \
            process_function_call_and_return_message(...)
        generation_dicts.append({'code_dict': code_dict, ...})
    
    generation_dicts = sorted(generation_dicts, key=lambda d: d['fitness'])
    history_best_generation.append(deepcopy(generation_dicts[0]))
    
    # ═══════════════════════════════════════════════════════════
    # ★ APÓS GERAÇÃO 0: CHAMAR DOWHY ★
    # ═══════════════════════════════════════════════════════════
    
    if self.config.run.use_dowhy_evolution:
        from dowhy_integration import generate_counterfactual_evolution
        
        self.logger.info(f"[DoWhy Iteration {iteration_id}] Gerando evolução contrafactual...")
        
        # Pega dados atuais do environment
        current_states, current_actions, current_static = self.env.train_data
        
        # Reconstrói DataFrame para DoWhy processar
        current_df = reconstruct_dataframe_from_tensors(
            states=current_states,
            actions=current_actions,
            static=current_static
        )
        
        # ★ DoWhy gera próximo timestep ★
        evolved_df = generate_counterfactual_evolution(
            current_df=current_df,
            iteration=iteration_id
        )
        
        # Salva CSV evoluído
        evolved_csv_path = f'{self.folder_path}/evolved_data_iter{iteration_id}.csv'
        evolved_df.to_csv(evolved_csv_path, index=False)
        
        # Recarrega dados evoluídos
        from libs.datasets.env import load_data_with_dowhy_evolution
        self.env.train_data, self.env.val_data, self.env.test_data, _ = \
            load_data_with_dowhy_evolution(
                initial_csv_path=None,
                iteration=iteration_id + 1,
                evolved_csv_path=evolved_csv_path
            )
        
        iteration_id += 1
        
        self.logger.info(f"[DoWhy] Dados evoluídos:")
        self.logger.info(f"  Shape: {self.env.train_data[0].shape}")
        self.logger.info(f"  Timesteps por paciente: {iteration_id + 1}")
    
    # ═══════════════════════════════════════════════════════════
    # GERAÇÕES 1-N: Treina com dados evoluídos
    # ═══════════════════════════════════════════════════════════
    
    for generation_id in range(1, self.config.run.generations):
        
        # Gera reflexão
        reflection_prompt = self.generate_reflection_prompt_with_group(...)
        messages_i = deepcopy(self.messages)
        messages_i.append({"role": "user", "content": reflection_prompt, ...})
        
        # LLM gera novo código
        response_messages = self.get_llm_response_with_retries(messages_i, n=1)
        
        for response_message in response_messages:
            # Treina com dados evoluídos (agora longitudinais!)
            function_return_message, code_dict, has_success = \
                process_function_call_and_return_message(
                    response_message["function_call"],
                    self.simulator_code_dict,
                    env=self.env,  # ← Usa dados com t0→t1
                    ...
                )
            generation_dicts.append({'code_dict': code_dict, ...})
        
        # Ordena e mantém melhores
        generation_dicts = sorted(generation_dicts, key=lambda d: d['fitness'])
        
        # ★ A CADA N GERAÇÕES: CHAMA DOWHY NOVAMENTE ★
        if generation_id % self.config.run.dowhy_evolution_frequency == 0:
            
            self.logger.info(f"[DoWhy Iteration {iteration_id}] Expandindo sequências temporais...")
            
            # Reconstrói DataFrame atual
            current_df = reconstruct_dataframe_from_tensors(
                states=self.env.train_data[0],
                actions=self.env.train_data[1],
                static=self.env.train_data[2]
            )
            
            # DoWhy gera t2 baseado em t0→t1
            evolved_df = generate_counterfactual_evolution(
                current_df=current_df,
                iteration=iteration_id
            )
            
            # Salva e recarrega
            evolved_csv_path = f'{self.folder_path}/evolved_data_iter{iteration_id}.csv'
            evolved_df.to_csv(evolved_csv_path, index=False)
            
            self.env.train_data, self.env.val_data, self.env.test_data, _ = \
                load_data_with_dowhy_evolution(
                    initial_csv_path=None,
                    iteration=iteration_id + 1,
                    evolved_csv_path=evolved_csv_path
                )
            
            iteration_id += 1
            
            self.logger.info(f"[DoWhy] Agora com {iteration_id + 1} timesteps por paciente")
        
        # Early stopping
        if patience_counter >= self.config.run.nsdt_patience:
            break
    
    # Retorna melhor modelo
    best_program = generation_dicts[0]['code_dict']
    return best_program['test_loss']


4. Configuração YAML
Arquivo: cbio_digital_twin.yaml

run:
  # Configuração DoWhy
  use_dowhy_evolution: true
  dowhy_evolution_frequency: 3  # A cada 3 gerações, chama DoWhy
  
  dowhy:
    treatment_vars: ['treatment_duration_days']
    outcome_vars: ['tumor_size', 'msi_score', 'tmb_nonsynonymous']
    causal_graph_path: 'config/causal_graph.dot'  # Opcional


Fluxo Completo Resumido

# ITERAÇÃO 0 (dataset estático inicial)
Dataset inicial: 100 pacientes × 1 timestep (t0)
↓
Framework treina modelo (Geração 0)
↓
DoWhy gera t1 para cada paciente (cenários contrafactuais)
↓
Dataset evoluído: 100 pacientes × 2 timesteps (t0→t1)

# ITERAÇÃO 1 (agora com sequências!)
Framework treina modelo (Gerações 1-3)
↓
DoWhy gera t2 baseado em t0→t1
↓
Dataset evoluído: 100 pacientes × 3 timesteps (t0→t1→t2)

# ITERAÇÃO 2
Framework treina modelo (Gerações 4-6)
↓
DoWhy gera t3 baseado em t0→t1→t2
↓
Dataset evoluído: 100 pacientes × 4 timesteps (t0→t1→t2→t3)

...e assim por diante


Resumo das Mudanças

Arquivo	Modificação	Propósito
env.py	Adicionar load_data_with_dowhy_evolution()	Carregar dados estáticos/evoluídos
dowhy_integration.py	Criar novo arquivo	Lógica DoWhy para gerar contrafactuais
agents.py	Modificar _run() linha ~380	Chamar DoWhy entre gerações
cbio_digital_twin.yaml	Adicionar seção dowhy	Configurar frequência e parâmetros

✅ Vantagens dessa Abordagem
Dataset inicial pode ser estático (apenas t0)
DoWhy cria evolução temporal de forma causal
Framework aprende progressivamente com mais timesteps
Modelo melhora ao ver trajetórias mais longas
Flexível: Ajusta frequência de chamadas DoWhy
Quer que eu implemente alguma dessas partes em código completo?

me de tudo que disse na mensagem anterior em um .md

GitHub Copilot
Sorry, your request failed. Please try again. Request id: c89a402f-f083-4ac5-9814-3af91c1cf6de

Reason: Please check your firewall rules and network connection then try again. Error Code: net::ERR_CONNECTION_RESET.