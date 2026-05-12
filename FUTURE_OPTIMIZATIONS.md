# Future Optimization Ideas for nanowhale

This document contains advanced optimization strategies that can be implemented in future iterations. They are organized by impact, effort, and risk.

## 🎯 High-Impact, Medium-High Effort

### 1. Progressive Layer Stacking
**Idea**: Start with 8 layers, gradually add more up to 16 during training.

**Why**:
- Reuses compute from earlier stages
- 25-35% compute savings
- Same final quality with fewer total FLOPs

**Implementation**:
```python
class ProgressiveModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.current_layers = config.initial_layers  # 8
        self.target_layers = config.num_hidden_layers  # 16
        self.all_layers = nn.ModuleList([DeepseekV4Block(config, i) for i in range(self.target_layers)])
    
    def expand_layers(self, step, total_steps):
        progress = step / total_steps
        new_count = int(self.current_layers + (self.target_layers - self.current_layers) * progress)
        self.current_layers = min(new_count, self.target_layers)
    
    def forward(self, x):
        for i in range(self.current_layers):
            x = self.all_layers[i](x)
        return x
```

**Expected**: 30% fewer total FLOPs, same final quality

---

### 2. Online Hard Example Mining (GREATS-inspired)
**Idea**: Dynamically select batches with highest loss (most informative samples).

**Why**:
- Focuses learning on difficult examples
- 15-25% fewer training steps needed
- Better sample efficiency

**Implementation**:
```python
class OnlineDataSelector:
    def __init__(self, model, buffer_size=1000):
        self.model = model
        self.buffer = []
    
    def select_batch(self, candidates, batch_size=8):
        # Compute loss for each candidate
        losses = []
        for candidate in candidates:
            loss = compute_loss(self.model, candidate)
            losses.append(loss)
        
        # Select top-k by loss (most informative)
        indices = np.argsort(losses)[-batch_size:]
        return [candidates[i] for i in indices]
```

**Expected**: 15-25% reduction in training steps

---

### 3. Knowledge Distillation from Larger Models
**Idea**: Use a larger teacher model (e.g., Llama-3-8B) to provide soft targets.

**Why**:
- Transfers knowledge from larger model
- 20-30% faster convergence
- Better generalization

**Implementation**:
```python
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """KL divergence between student and teacher distributions."""
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

# Train with combined loss
total_loss = cross_entropy_loss + 0.5 * distillation_loss
```

**Expected**: 20-30% faster convergence, better final quality

---

## 🚀 Medium-Impact, Low-Medium Effort

### 4. Dynamic Sequence Packing
**Idea**: Pack multiple short sequences into one batch to reduce padding waste.

**Why**:
- Current approach pads to longest sequence
- Packing can reduce padding by 30-50%
- 20-30% throughput improvement

**Implementation**:
```python
def pack_sequences(sequences, max_length=4096):
    """Bin-packing: group sequences to minimize padding."""
    sorted_seqs = sorted(sequences, key=len, reverse=True)
    packed = []
    current_pack = []
    current_len = 0
    
    for seq in sorted_seqs:
        if current_len + len(seq) <= max_length:
            current_pack.append(seq)
            current_len += len(seq)
        else:
            if current_pack:
                packed.append(current_pack)
            current_pack = [seq]
            current_len = len(seq)
    
    if current_pack:
        packed.append(current_pack)
    
    return packed
```

**Expected**: 20-30% throughput improvement

---

### 5. Multi-Task Learning Head
**Idea**: Add auxiliary prediction heads (next sentence prediction, span prediction).

**Why**:
- Multi-task learning improves representations
- Acts as regularization
- 3-8% better downstream performance

**Implementation**:
```python
class MultiTaskHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.nsp_head = nn.Linear(hidden_size, 2)  # Next sentence prediction
        self.span_head = nn.Linear(hidden_size, 2)  # Span prediction
    
    def forward(self, hidden_states, labels=None):
        # Use [CLS] token for NSP
        cls_hidden = hidden_states[:, 0, :]
        nsp_logits = self.nsp_head(cls_hidden)
        
        # Span prediction on random tokens
        span_logits = self.span_head(hidden_states)
        
        loss = 0
        if labels is not None:
            nsp_loss = F.cross_entropy(nsp_logits, labels['nsp_labels'])
            span_loss = F.cross_entropy(span_logits.view(-1, 2), labels['span_labels'].view(-1))
            loss = nsp_loss * 0.1 + span_loss * 0.1  # Weight auxiliary losses
        
        return loss
```

**Expected**: 3-8% better downstream performance

---

### 6. Contrastive Learning Objective
**Idea**: Add a contrastive loss that pulls similar contexts together.

**Why**:
- Improves representation learning
- Especially beneficial for long-context
- Better semantic understanding

**Implementation**:
```python
def contrastive_loss(hidden_states, temperature=0.1):
    """InfoNCE loss on context representations."""
    # Use last token as context representation
    context_repr = hidden_states[:, -1, :]  # [batch, hidden]
    
    # Normalize
    context_repr = F.normalize(context_repr, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = context_repr @ context_repr.T / temperature
    
    # InfoNCE loss
    labels = torch.arange(len(sim_matrix), device=sim_matrix.device)
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss * 0.1  # Weight auxiliary loss
```

**Expected**: Better long-context understanding

---

## 💡 Experimental/Creative

### 7. Self-Correction Training
**Idea**: Generate multiple samples, select best, retrain on those.

**Why**:
- Model learns from its own high-quality outputs
- Iterative improvement
- No external data needed

**Implementation**:
```python
def self_correction_step(model, tokenizer, prompt, n_samples=5):
    """Generate candidates, filter by quality, retrain."""
    candidates = []
    for _ in range(n_samples):
        output = model.generate(prompt, max_new_tokens=100, do_sample=True)
        text = tokenizer.decode(output[0])
        quality = assess_quality(text)  # Custom quality metric
        candidates.append((text, quality))
    
    # Select best candidate
    best_text = max(candidates, key=lambda x: x[1])[0]
    return best_text
```

**Expected**: Improved generation quality over time

---

### 8. Mixture-of-Depths
**Idea**: Skip computation for "easy" tokens using a confidence threshold.

**Why**:
- Reduces computation for tokens model is confident about
- 20-40% reduction in inference cost
- Some training speedup

**Implementation**:
```python
class AdaptiveBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.original_block = DeepseekV4Block(config, layer_idx)
        self.confidence_head = nn.Linear(config.hidden_size, 1)
        self.skip_threshold = 0.8
    
    def forward(self, x, cache=None):
        # Predict confidence
        confidence = torch.sigmoid(self.confidence_head(x[:, -1, :]))
        
        # Skip if confident
        if confidence > self.skip_threshold and cache is not None:
            return cache, cache  # Return cached output
        
        # Otherwise compute
        output, new_cache = self.original_block(x)
        return output, new_cache
```

**Expected**: 20-40% inference speedup

---

### 9. Dynamic Vocabulary Expansion
**Idea**: Start with smaller vocab, gradually add rare tokens.

**Why**:
- Reduces embedding size early
- Focuses learning on frequent tokens
- 10-15% faster early training

**Implementation**:
```python
class VocabularyCurriculum:
    def __init__(self, base_vocab_size=50000, final_vocab_size=129280):
        self.base_size = base_vocab_size
        self.final_size = final_vocab_size
    
    def get_active_vocab(self, step, total_steps):
        """Expand vocabulary over training."""
        progress = step / total_steps
        current_size = int(self.base_size + (self.final_size - self.base_size) * progress)
        return current_size
```

**Expected**: 10-15% faster early training

---

## 📊 Prioritization Matrix

| Idea | Impact | Effort | Risk | Priority |
|------|--------|--------|------|----------|
| Progressive Layer Stacking | High | High | Medium | **P2** |
| Online Hard Example Mining | High | High | Medium | **P2** |
| Knowledge Distillation | High | High | Low | **P2** |
| Dynamic Sequence Packing | Medium | Medium | Low | **P1** |
| Multi-Task Learning | Medium | Medium | Low | **P1** |
| Contrastive Learning | Medium | Medium | Low | **P1** |
| Self-Correction Training | Medium | High | Medium | **P3** |
| Mixture-of-Depths | Medium | High | High | **P3** |
| Dynamic Vocabulary | Medium | Medium | Low | **P2** |

**Priority Levels**:
- **P0**: Implement now (easy, high impact) - Already done!
- **P1**: Implement soon (medium effort, good impact)
- **P2**: Plan for next iteration (higher effort, high impact)
- **P3**: Experimental (uncertain ROI)

---

## 🎯 Recommended Implementation Order

### Phase 1 (Next Week)
1. **Dynamic Sequence Packing** - Easy win, immediate throughput improvement
2. **Multi-Task Learning** - Simple to add, good regularization
3. **Contrastive Learning** - Improves representations

### Phase 2 (Next Month)
4. **Progressive Layer Stacking** - Significant compute savings
5. **Online Hard Example Mining** - Better sample efficiency
6. **Knowledge Distillation** - Quality improvement

### Phase 3 (Future)
7. **Dynamic Vocabulary Expansion** - Early training speedup
8. **Self-Correction Training** - Iterative improvement
9. **Mixture-of-Depths** - Inference optimization

---

## 📚 Research References

1. **Progressive Layer Stacking**: "Curriculum-Guided Layer Scaling" (2025)
2. **Online Hard Example Mining**: "GREATS: Online Selection of High-Quality Data" (NeurIPS 2024)
3. **Knowledge Distillation**: "Pre-training Distillation for Large Language Models" (ACL 2025)
4. **Dynamic Sequence Packing**: "Faster LLM Training with Variable Sequence Length Curriculum" (NeurIPS 2024)
5. **Multi-Task Learning**: "Scaling Laws for Multi-Task Pre-training" (2025)
6. **Contrastive Learning**: "Contrastive Pre-training for LLMs" (2025)
7. **Mixture-of-Depths**: "AdaPonderLM: Gated Pondering Language Models" (2026)
8. **Dynamic Vocabulary**: "Scaling LLM Pre-training with Vocabulary Curriculum" (2025)

---

## 💡 Implementation Tips

### For Progressive Layer Stacking:
- Ensure weight initialization is smooth when adding layers
- Use layer copying from existing layers
- Gradually unfreeze new layers

### For Online Hard Example Mining:
- Maintain a buffer of recent losses
- Use Thompson sampling for exploration vs exploitation
- Don't overfit to hard examples

### For Knowledge Distillation:
- Use temperature scaling (2.0-4.0)
- Balance CE and KL loss (0.5-0.8 weight on CE)
- Consider using multiple teachers

### For Dynamic Sequence Packing:
- Use First-Fit Decreasing bin packing algorithm
- Maintain attention mask for packed sequences
- Be careful with position IDs

---

## 🚀 Quick Start for Any Idea

```python
# Template for implementing new optimization
class NewOptimization:
    def __init__(self, config):
        self.config = config
    
    def apply(self, model, step, total_steps):
        # Implementation here
        pass
    
    def on_train_batch(self, batch, loss):
        # Modify training dynamics
        return modified_loss
```

---

## 📈 Measuring Success

For each optimization, track:
1. **Training Loss** - Should converge faster
2. **Validation Perplexity** - Should be lower
3. **Throughput** - Tokens/sec should increase
4. **Memory Usage** - Should stay within limits
5. **Final Quality** - Downstream task performance

Use these metrics to decide whether to keep or discard each optimization.

---

## 🔧 Integration Checklist

Before implementing any advanced optimization:
- [ ] Baseline model is stable
- [ ] All P0 optimizations are applied
- [ ] Validation pipeline is set up
- [ ] Metrics tracking is in place
- [ ] A/B testing framework exists
- [ ] Rollback plan is ready

---

This document serves as a roadmap for future improvements to the nanowhale training pipeline. Start with P1 items for quick wins, then move to P2 for more significant gains.
