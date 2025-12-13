# Features 


O modelo recebe **12 features** a cada passo de tempo:
- **4 features DINÂMICAS** (STATE): mudam a cada dia → modelo prevê como elas vão mudar
- **7 features ESTÁTICAS**: não mudam durante a simulação → personalizam o comportamento
- **1 feature CONTROLE**: você decide (tratamento acumulado)

---

## 12 FEATURES 

```python
x = torch.cat([
    # DINÂMICAS
    tumor_size,           
    msi_score,           
    tmb_nonsynonymous,   
    mitotic_rate,         

    #ESTÁTICAS 
    age_at_diagnosis,     
    stage_encoded,        
    treatment_encoded,    
    has_kit_mutation,     
    tumor_purity,         
    site_small_intestine, 
    site_stomach,         

    # VOCÊ CONTROLA 
    treatment_duration   
], dim=-1)
```

**Resultado:** vetor de 12 números → `x.shape = [1, 12]`

---

## MODELO

### Passo 1: Concatenação (linha 40-44)
```
Exemplo de um paciente no dia 100:
┌────────────────────────────────────────────────────────────┐
│ [13.6, 5.07, 0.13, 50.0, 68.0, 1.0, 0.0, 1.0, 90.0, 0.0, 1.0, 100.0] │
│   ↑     ↑     ↑     ↑     ↑     ↑    ↑    ↑     ↑     ↑    ↑     ↑    │
│   │     │     │     │     │     │    │    │     │     │    │     │    │
│  tumor msi  tmb  mitotic age  stage treat KIT  purity SI  stomach days│
└────────────────────────────────────────────────────────────┘
         12 números formam um VETOR de contexto completo
```

### Passo 2: Processamento pela MLP (linhas 16-26, 49)
```
x (12 números)
    ↓
[Linear 12→256]  ← Aprende combinações das 12 features
    ↓
[LayerNorm + LeakyReLU + Dropout]
    ↓
[Linear 256→128] ← Refina as interações
    ↓
[LayerNorm + LeakyReLU + Dropout]
    ↓
[Linear 128→4]   ← Produz 4 derivadas
    ↓
[d_tumor, d_msi, d_tmb, d_mitotic]
```

### Passo 3: Treatment Gate (linhas 27-32, 53-54)
```
x (12 números)
    ↓
[Linear 12→64]   ← Aprende sensibilidade ao tratamento
    ↓
[LeakyReLU + Dropout]
    ↓
[Linear 64→1]    ← Produz intensidade γ
    ↓
[Softplus]       ← γ sempre ≥ 0 (monotônico)
    ↓
d_tumor = d_tumor - γ * treatment_duration
```

**IMPORTANTE:** O γ é **personalizado** para cada paciente baseado nas 12 features!

---

