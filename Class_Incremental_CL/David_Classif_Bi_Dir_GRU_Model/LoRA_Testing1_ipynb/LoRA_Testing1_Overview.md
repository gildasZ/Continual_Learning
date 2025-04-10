# LoRA Testing Overview

This document summarizes experiments on LoRA (Low-Rank Adaptation) implementations for Continual Learning (CL) using a Bi-Directional GRU model with attention. The goal is to balance adaptation to new classes and preservation of old knowledge, while mitigating catastrophic forgetting (especially for Class 0). All experiments incorporate knowledge distillation with varying LoRA strategies.

---

## LoRA v1: Initial Implementation

### Overview
- **Version**: LoRA v1
- **Notebook**: `CL_Classif_Bi_Dir_GRU_Model_v1.ipynb`
- **Description**: Initial LoRA implementation for fine-tuning, without similarity considerations.
- **Strategy**: 
  - Fixed single LoRA adapter added for fine-tuning.
  - Full training (all parameters trainable except specific frozen layers).
- **Details**: 
  - Single LoRA adapter trained alongside the full model.
  - No dynamic similarity-based adjustments.
- **Characteristics**: 
  - Simple, no dynamic allocation.
  - Flexible but computationally costly, prone to forgetting.

*Note*: Similarity-based strategies (Period 2+ scenarios) are not applicable here.

---

## LoRA DynEx_CLoRA (Initial Attempt)

### Overview
- **Notebook**: `CIL_Classif_Bi_Dir_GRU_Model_DynEx_CLoRA-init_idea_test.ipynb`
- **Description**: First attempt at DynEx_CLoRA (Dynamic Expansion with Class-wise LoRA) using similarity-based LoRA allocation.
- **Strategy**: 
  - Two thresholds (`tau_low`, `tau_high`) determine LoRA strategy:
    1. **New LoRA**: If max similarity ≤ `tau_low` or no existing features.
    2. **Update Last LoRA**: If max similarity > `tau_high`.
    3. **Joint Update**: If `tau_low` < max similarity ≤ `tau_high`.
- *Note*: Abandoned due to suboptimal performance.

---

## LoRA DynEx_CLoRA (Improvements)

![alt text](../img/overview/DynEx_CLoRA(Improvements).png)

### Overview
- **Versions**: LoRA v2 to v10
- **Description**: Iterative refinements of DynEx_CLoRA, focusing on similarity-based LoRA allocation.

### Strategy (Period 2+ Scenarios)
1. **Calculate Similarity**: Cosine similarity between new and old class features:
![alt text](../img/overview/Calculate%20Similarity.png)
   - *New Class*: Current period distribution.
   - *Existing Class*: Previous period distribution.
2. **Old Classes (excl. Class 0)**: Update related networks based on self-similarity.
3. **New Classes vs. Class 0**: Add LoRA adapter if similarity with Class 0 is high.
4. **New Classes vs. Old Classes**: Update related networks or associate with existing LoRA based on similarity.

### Logic Implementation

#### Logic 1: Adapt to Distribution Shifts
- **Old Classes (excl. Class 0)**:
  - If self-similarity < threshold: Unfreeze related network (`attention_fc` or LoRA).
- **New Classes**:
  - If similarity with Class 0 ≥ threshold: Add new LoRA, unfreeze.
  - If similarity with other old classes ≥ threshold: Unfreeze related network.
  - If similarity with Class 0 < threshold: Associate with existing LoRA of similar old class.

#### Logic 2: Preserve Old Knowledge
- **Old Classes (excl. Class 0)**:
  - If self-similarity ≥ threshold: Unfreeze related network.
- **New Classes**:
  - If similarity with Class 0 ≥ threshold: Add new LoRA, unfreeze.
  - If similarity with other old classes < threshold: Unfreeze related network.
  - If similarity with Class 0 < threshold: Associate with existing LoRA of similar old class.

### Parameter Testing (v2–v10)
| Version | Threshold | Logic | Rank |
|---------|-----------|-------|------|
| v2      | 0.5       | 1     | 4    |
| v3      | 0.5       | 2     | 4    |
| v4      | 0.0       | 2     | 4    |
| v5      | 0.0       | 2     | 8    |
| v6      | 0.3       | 2     | 4    |
| v7      | 0.5       | 1     | 8    |
| v8      | 0.0       | 1     | 4    |
| v9      | 0.3       | 1     | 4    |
| v10     | 0.3       | 1     | 8    |

---

## Current Conclusion
- **Best Logic**: Logic 1 slightly outperforms Logic 2.
- **Optimal Threshold**: 0.0
- **Best Rank**: 4.
- **Efficiency**: 1.5x faster training per period vs. full training.
- **Forgetting**: Slightly mitigates Class 0 forgetting, stable despite minor new-class trade-offs.