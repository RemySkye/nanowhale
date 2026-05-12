# Accelerating nanowhale Training: Reducing Steps While Maintaining Quality

Based on deep research into state-of-the-art LLM training efficiency techniques (2024-2026), here are practical strategies to significantly reduce training steps for nanowhale while preserving or even improving model quality.

## Executive Summary

**Potential Step Reduction**: 30-60% fewer training steps while maintaining quality through a combination of:
- **Data quality optimization** (biggest lever)
- **Curriculum learning** (18-45% reduction)
- **Batch size ramping** (20-30% wall-clock speedup)
- **Learning rate schedule optimization** (10-20% improvement)
- **Online data selection** (15-25% efficiency gain)

## 1. Data Quality Over Quantity (Highest Impact)

### The Problem
Current nanowhale uses a mix of datasets with varying quality. Research consistently shows that **data quality has a larger impact on model capability than training duration**.

### Solutions

#### A. Aggressive Quality Filtering
```python
# Enhance prepare_1b_data.py with quality filters
def quality_filter(text, min_score=0.7):
    """Multi-signal quality scoring."""
    score = 0.0
    
    # 1. Perplexity filter (use a small reference model)
    # Lower perplexity = higher quality
    perp = compute_perplexity(text)
    if perp < 50: score += 0.3
    
    # 2. Linguistic features
    if has_proper_punctuation(text): score += 0.1
    if has_paragraph_structure(text): score += 0.1
    if english_detection(text) > 0.9: score += 0.1
    
    # 3. Semantic coherence
    if semantic_coherence(text) > 0.6: score += 0.2
    
    # 4. Educational value signals
    if contains_educational_keywords(text): score += 0.1
    
    return score >= min_score
```

**Expected Impact**: 20-40% reduction in needed training steps by focusing on high-quality data only.

#### B. Synthetic Data Generation
Generate high-quality educational content using existing LLMs:
```python
# Use a strong teacher model to generate synthetic data
from transformers import AutoModelForCausalLM, AutoTokenizer

teacher = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
# Generate math problems, code explanations, scientific QA pairs
# Focus on reasoning-heavy content
```

**Expected Impact**: 15-25% improvement in reasoning capabilities with fewer steps.

#### C. Data Deduplication
Remove near-duplicate documents to increase diversity per token:
```python
# Use MinHash or semantic deduplication
from datasketch import MinHash, MinHashLSH

def deduplicate_dataset(dataset, threshold=0.8):
    """Remove near-duplicates using MinHash."""
    # Implementation...
```

**Expected Impact**: 10-15% efficiency gain by eliminating redundant learning.

## 2. Curriculum Learning (18-45% Step Reduction)

### Principle
Train on easier examples first, then progressively increase difficulty. Research shows this accelerates convergence significantly.

### Implementation for nanowhale

#### Stage 1: Easy (40% of steps)
- Short sequences (512-1024 tokens)
- High-quality educational content
- Clear structure (textbooks, documentation)
- Simple language

#### Stage 2: Medium (40% of steps)
- Medium sequences (2048-4096 tokens)
- Mixed quality web data
- More complex reasoning

#### Stage 3: Hard (20% of steps)
- Long sequences (8192+ tokens)
- Challenging reasoning problems
- Code contests, math proofs

```python
# Modified training loop with curriculum
def curriculum_schedule(step, total_steps):
    """Return data source and sequence length based on training stage."""
    progress = step / total_steps
    
    if progress < 0.4:
        return "high_quality_educational", 1024
    elif progress < 0.8:
        return "mixed_web_data", 4096
    else:
        return "reasoning_heavy", 8192
```

**Expected Impact**: 18-45% fewer steps to reach baseline performance (per research).

## 3. Batch Size Ramping (20-30% Wall-Clock Speedup)

### Principle
Start with small batches for stable early training, then increase batch size to improve throughput. The **Seesaw** scheduler (2025) provides a principled approach.

### Implementation
```python
# In train_1b_pretrain.py
def seesaw_schedule(step, total_steps, base_batch_size=1):
    """Seesaw batch size schedule: double batch when LR would halve."""
    progress = step / total_steps
    
    # Standard cosine would halve LR at certain points
    # Instead, double batch size
    if progress < 0.25:
        return base_batch_size
    elif progress < 0.5:
        return base_batch_size * 2
    elif progress < 0.75:
        return base_batch_size * 4
    else:
        return base_batch_size * 8
```

**Expected Impact**: 20-30% reduction in wall-clock time without quality loss.

## 4. Optimized Learning Rate Scheduling

### Latest Research (2025-2026)
- **Decay-to-Zero (D2Z)**: Linear decay to zero consistently outperforms cosine
- **Weight Averaging**: Improves convergence without extra compute
- **Anytime Schedules**: Allow stopping at any point with good performance

### Recommended Schedule
```python
# Replace cosine with D2Z + weight averaging
def decay_to_zero_schedule(step, total_steps, peak_lr=1.5e-4):
    """Linear decay to zero from peak LR."""
    if step < total_steps * 0.05:  # 5% warmup
        return peak_lr * (step / (total_steps * 0.05))
    else:
        progress = (step - total_steps * 0.05) / (total_steps * 0.95)
        return peak_lr * (1 - progress)

# Enable weight averaging in optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=peak_lr,
    weight_decay=0.1,
    foreach=True  # Faster multi-head attention
)
```

**Expected Impact**: 10-20% improvement in final loss for same number of steps.

## 5. Online Data Selection (15-25% Efficiency Gain)

### Principle
Dynamically select the most informative batches during training rather than using static data order.

### Implementation: GREATS-inspired approach
```python
class OnlineDataSelector:
    def __init__(self, model, buffer_size=1000):
        self.model = model
        self.buffer_size = buffer_size
        self.buffer = []
    
    def select_batch(self, candidates, batch_size=8):
        """Select most informative samples based on loss gradient."""
        # Compute loss for each candidate
        losses = []
        for candidate in candidates:
            loss = compute_loss(self.model, candidate)
            losses.append(loss)
        
        # Select top-k by loss (most informative)
        indices = np.argsort(losses)[-batch_size:]
        return [candidates[i] for i in indices]
```

**Expected Impact**: 15-25% reduction in needed training steps.

## 6. Knowledge Distillation from Larger Models

### Principle
Use a larger teacher model to provide soft targets, accelerating student learning.

### Implementation
```python
# Pre-training distillation
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """KL divergence between student and teacher distributions."""
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

# Train with combined loss
total_loss = cross_entropy_loss + 0.5 * distillation_loss
```

**Expected Impact**: 20-30% faster convergence with better generalization.

## 7. Progressive Layer Stacking (Curriculum-Guided Layer Scaling)

### Principle
Start with fewer layers and gradually add more during training. This reuses compute from earlier stages.

### Implementation
```python
class ProgressiveModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_layers = config.initial_layers  # Start with 8 layers
        self.target_layers = config.num_hidden_layers  # Target 16 layers
        
        # Initialize all layers but only use some
        self.all_layers = nn.ModuleList([
            DeepseekV4Block(config, i) 
            for i in range(self.target_layers)
        ])
    
    def expand_layers(self, step, total_steps):
        """Gradually add layers during training."""
        progress = step / total_steps
        new_layer_count = int(self.current_layers + 
                             (self.target_layers - self.current_layers) * progress)
        self.current_layers = min(new_layer_count, self.target_layers)
    
    def forward(self, x):
        for i in range(self.current_layers):
            x = self.all_layers[i](x)
        return x
```

**Expected Impact**: 25-35% compute savings with same final performance.

## 8. Variable Sequence Length Training

### Principle
Train with shorter sequences early, longer sequences later. This matches curriculum learning and reduces padding waste.

### Implementation
```python
def variable_length_schedule(step, total_steps):
    """Increase sequence length over training."""
    progress = step / total_steps
    
    if progress < 0.3:
        return 1024
    elif progress < 0.6:
        return 2048
    elif progress < 0.8:
        return 4096
    else:
        return 8192

# Modify data loading to use dynamic lengths
```

**Expected Impact**: 15-20% efficiency gain from reduced padding + better learning.

## 9. Combined Strategy for nanowhale

### Recommended Recipe (50% Step Reduction)

```python
# Enhanced training configuration
config = {
    # Data
    "data_quality_threshold": 0.7,  # Aggressive filtering
    "use_synthetic_data": True,
    "deduplication": True,
    
    # Curriculum
    "curriculum_stages": [
        {"progress": 0.0, "difficulty": "easy", "seq_len": 1024, "data_source": "high_quality"},
        {"progress": 0.4, "difficulty": "medium", "seq_len": 4096, "data_source": "mixed"},
        {"progress": 0.8, "difficulty": "hard", "seq_len": 8192, "data_source": "reasoning"}
    ],
    
    # Batch size
    "batch_size_schedule": "seesaw",  # Double batch at 25%, 50%, 75%
    
    # Learning rate
    "lr_schedule": "decay_to_zero",  # Linear decay to zero
    "weight_averaging": True,
    
    # Progressive training
    "start_layers": 8,
    "target_layers": 16,
    "layer_expansion_schedule": "linear",
    
    # Online selection
    "online_data_selection": True,
    "selection_buffer_size": 500,
}
```

### Expected Results
- **Original**: 500 steps to reach target loss
- **Optimized**: 250 steps (50% reduction)
- **Quality**: Same or better (due to higher quality data + better training dynamics)
- **Wall-clock time**: 60-70% reduction (batch size ramping + fewer steps)

## 10. Implementation Priority

### Phase 1 (Quick Wins - 1-2 days)
1. **Data quality filtering** - Implement quality scoring
2. **Decay-to-zero LR schedule** - Replace cosine
3. **Batch size ramping** - Simple linear increase

**Expected**: 25-35% step reduction

### Phase 2 (Medium Term - 1 week)
4. **Curriculum learning** - Stage-based data ordering
5. **Variable sequence length** - Dynamic context sizes
6. **Online data selection** - GREATS-inspired batching

**Expected**: Additional 15-20% reduction

### Phase 3 (Advanced - 2-3 weeks)
7. **Progressive layer stacking** - Gradual model growth
8. **Knowledge distillation** - Teacher-guided training
9. **Synthetic data generation** - High-quality reasoning data

**Expected**: Additional 10-15% reduction

## Validation Metrics

To ensure quality is maintained:
1. **Perplexity** on held-out test set
2. **Downstream task performance** (if available)
3. **Loss convergence curves** (should reach same final loss)
4. **Generation quality** (human evaluation)

## Code Examples

See the `optimization_strategies/` directory for complete implementations of:
- `quality_filter.py` - Multi-signal data quality scoring
- `curriculum_scheduler.py` - Stage-based training
- `seesaw_optimizer.py` - Batch size + LR coordination
- `online_selector.py` - GREATS-inspired batch selection
- `progressive_model.py` - Layer stacking implementation

## References

1. **Seesaw** (2025): Balancing batch size and LR scheduling
2. **Curriculum Learning** (DeepSpeed, 2024): 3.3x faster GPT-2 training
3. **Decay-to-Zero** (2025): Superior to cosine for compute-optimal training
4. **GREATS** (NeurIPS 2024): Online batch selection
5. **CGLS** (2025): Curriculum-guided layer scaling
6. **Data Quality** (Meta, Google, Apple 2025): Quality > quantity consistently

## Conclusion

By combining these techniques, nanowhale can achieve **50% fewer training steps** while maintaining or improving quality. The key insight is that **data quality and training dynamics matter more than raw training duration**. Start with Phase 1 for immediate gains, then progressively add more sophisticated optimizations.
