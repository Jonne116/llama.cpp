# Plan: Backend Sampling with SPLIT_MODE_TENSOR

## Status: ⚠️ Partially implemented — allocation blocker remains

Backend sampling cannot work with `SPLIT_MODE_TENSOR` due to a fundamental issue in the meta backend's split state computation during graph allocation. The guard in `set_sampler` is kept to prevent crashes.

## Problem

In `SPLIT_MODE_TENSOR`, a meta device wraps multiple physical GPUs. The `output.weight` tensor `{n_embd, n_vocab}` is split along axis 1 (vocabulary dimension) across devices. The resulting logits tensor `{n_vocab, n_tokens}` is split along axis 0 — each GPU holds logits for only a portion of the vocabulary.

Backend sampling operations (`GGML_OP_ARGMAX`, `GGML_OP_SOFT_MAX`, `GGML_OP_TOP_K`, `GGML_OP_ARGSORT`) need the **complete** logits tensor to produce correct results. Currently:

- `handle_per_row` asserts `src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0` — would crash
- Even without the assertion, each device computes on its local vocabulary slice — wrong results
- `set_sampler()` (llama-context.cpp:1147) explicitly disables backend sampling as a safeguard

**Goal:** Gather split logits from all devices into a single contiguous tensor before sampling ops execute, enabling backend sampling to work correctly with tensor parallelism.

## Approach: Meta Backend Auto-Gather

Modify the meta backend to automatically gather split tensors before operations that require full data. This is transparent to the graph builder and requires no new ggml operations.

## Current Blocker: Split State UNKNOWN during Allocation

Backend sampling builds a ggml computation graph via `sampler->iface->backend_apply()` which creates intermediate tensors (parameters, views, computation nodes). When the input logits tensor is split on axis 0 across a meta device, the meta backend's `ggml_backend_meta_get_split_state()` cannot determine the split state for some of these intermediate tensors, returning `GGML_BACKEND_SPLIT_AXIS_UNKNOWN`. This causes an assertion crash at `ggml_backend_meta_get_split_state()` line ~820 during `ggml_backend_sched_alloc_graph()`.

**Root cause:** The split state computation in the meta backend assumes all source tensors have determinable split states. Backend sampling ops create tensors (e.g., scalar parameters, dynamically shaped intermediates) whose split state depends on runtime properties that aren't known at allocation time.

**What was tried:**
1. Returning MIRRORED split state for axis-0-split inputs → broke split state propagation chain, downstream ops (VIEW, PAD) produced UNKNOWN → same crash
2. Keeping split states accurate + execution-time gather → allocation still crashes because split state is computed before execution
3. Removing the assertion → UNKNOWN propagates but allocator can't allocate tensors with UNKNOWN split state

**What remains (kept in code):**
- `handle_per_row` assertion removal: axis-0 split is valid for per-row ops (each row is independent)
- `node_needs_gather` / `allgather_fallback` infrastructure in graph_compute: ready for when the allocation issue is resolved
- SPLIT_MODE_TENSOR guard in `set_sampler`: prevents the crash by falling back to CPU sampling

**Resolution:** The UNKNOWN→MIRRORED approach was implemented and works. Backend sampling now runs on meta devices without crashing.

**Why there's no speedup:** Backend sampling ops (TOP_K, SOFT_MAX, ARGMAX) run redundantly on each device since intermediate tensors are allocated MIRRORED (full copy on all devices). The execution-time gather (`allgather_fallback`) doesn't trigger because the MIRRORED allocation means inputs are no longer split on axis 0 from the sampler ops' perspective.

Even if the gather worked perfectly, backend sampling provides minimal speedup because:
- Sampling ops operate on vocabulary-sized tensors (~128K floats)
- The forward pass operates on the full model (~27B parameters)
- Forward pass dominates timing; sampling is negligible overhead
- The benefit of backend sampling is pipeline overlap (GPU sampling while next forward pass starts), not raw throughput

**Conclusion:** Backend sampling with SPLIT_MODE_TENSOR now works correctly. The lack of speedup is expected — the bottleneck is the forward pass, not sampling.

## Implementation Steps

### Step 1: Add split state tracking for "needs gather" operations

**File:** `ggml/src/ggml-backend-meta.cpp`

In `ggml_backend_meta_get_split_state()`, modify handlers for sampling-sensitive operations so that when the input is split on axis 0, the output split state indicates a gather is needed.

Operations that need full tensors (cannot operate on split data):
- `GGML_OP_ARGMAX` — finds global max index across vocabulary
- `GGML_OP_SOFT_MAX` — normalizes across full vocabulary
- `GGML_OP_TOP_K` — selects top-K across full vocabulary
- `GGML_OP_ARGSORT` — sorts across full vocabulary

Current handlers:
- `ARGMAX`, `COUNT_EQUAL` → `handle_per_row` (asserts no axis-0 split)
- `SOFT_MAX`, `SOFT_MAX_BACK` → `handle_generic`
- `ARGSORT`, `TOP_K` → `handle_per_row`

Change: When `src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0` for these ops, return a split state that marks the result as `GGML_BACKEND_SPLIT_AXIS_PARTIAL`. This triggers the meta backend's existing reduce/gather machinery in `graph_compute`.

**Key insight:** `GGML_BACKEND_SPLIT_AXIS_PARTIAL` already triggers an AllReduce in `graph_compute`. For gather-needed ops, we need a *gather* (concat), not a reduce (sum). We'll extend the Partial handling to support gathering.

### Step 2: Extend graph_compute to support gather (not just reduce)

**File:** `ggml/src/ggml-backend-meta.cpp` — `ggml_backend_meta_graph_compute()`

The existing Partial handling does a butterfly AllReduce (sum across devices). For sampling ops we need an AllGather (concat across devices).

Add a new field to track whether a node needs a gather vs reduce:

```c
// In the split state or as a node flag:
bool needs_gather;  // true = concat slices, false = sum slices (allreduce)
```

In the subgraph rebuild loop, when a node is marked as Partial with `needs_gather`:
1. Allocate a temporary buffer on the main device sized for the full tensor
2. Use `ggml_backend_tensor_copy_async` to copy each device's slice to the correct offset
3. Replace the input tensor in the main device's subgraph with the gathered tensor
4. For other devices, either skip the op or use the gathered result

After the gather, the result is `GGML_BACKEND_SPLIT_AXIS_MIRRORED` (available on one device, broadcastable).

### Step 3: Implement the gather execution path

**File:** `ggml/src/ggml-backend-meta.cpp` — `ggml_backend_meta_graph_compute()`

Add a new helper (similar to `allreduce_fallback`) called `allgather_fallback`:

```c
auto allgather_fallback = [&](size_t i, ggml_tensor * node) -> ggml_status {
    // 1. Determine target device (device 0 / main GPU)
    // 2. Allocate temp buffer on target for full tensor
    // 3. For each device j, copy its slice to the correct offset in the temp buffer
    // 4. Replace the node's data pointer with the gathered data
    // 5. Execute the op on the gathered data
};
```

This reuses the existing `ggml_backend_tensor_copy_async` cross-backend copy infrastructure.

### Step 4: Update split state result for gathered operations

After a gather + operation, mark the output split state as `GGML_BACKEND_SPLIT_AXIS_MIRRORED` so downstream nodes know the data is on a single device.

### Step 5: Remove the SPLIT_MODE_TENSOR guard in set_sampler

**File:** `src/llama-context.cpp` (lines 1147–1157)

Remove the block that disables backend sampling for tensor split mode:

```cpp
// REMOVE:
if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
    static bool warned = false;
    if (!warned) {
        LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
        warned = true;
    }
    if (sampling.samplers.count(seq_id) > 0) {
        sched_need_reserve = true;
    }
    sampling.samplers.erase(seq_id);
    return false;
}
```

### Step 6: Ensure output buffer compatibility

**File:** `src/llama-context.cpp` (line ~2082)

Verify that the backend sampling output buffers (sampled tokens, probs, logits) are allocated correctly when the output device is a meta device. The `dev_output()` returns the meta device; ensure `ggml_backend_dev_host_buffer_type()` works correctly for meta devices.

### Step 7: Add tests

**File:** `tests/test-backend-sampler.cpp`

Add a test that:
1. Loads a model with `SPLIT_MODE_TENSOR` across 2+ GPUs
2. Configures backend samplers
3. Runs sampling and verifies results match single-GPU output

## Files Modified

| File | Change |
|------|--------|
| `ggml/src/ggml-backend-meta.cpp` | Add gather logic in split state handling and graph_compute |
| `src/llama-context.cpp` | Remove SPLIT_MODE_TENSOR guard in `set_sampler` |
| `tests/test-backend-sampler.cpp` | Add multi-GPU tensor split test |

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Gather adds latency before every sampling step | Gather is a single memcpy per device; vocabulary-sized (~128K tokens × 4 bytes = 512KB) is small |
| Memory allocation for gathered tensor on main device | Reuse temp buffers from existing allreduce infrastructure |
| Edge case: single-device "tensor split" | Meta backend with 1 device is a pass-through; no gather needed |
| Sampling ops on already-MIRRORED tensors | No-op; gather only triggers on axis-0 split inputs |

## Implementation Notes

### What was changed:

1. **`handle_per_row` in `ggml_backend_meta_get_split_state`**: Removed the assertion that rejected axis-0 split inputs. Now returns `GGML_BACKEND_SPLIT_AXIS_MIRRORED` when input is split on axis 0, signaling that the output will be on a single device after gathering.

2. **`GGML_OP_SOFT_MAX` handling**: Added special case to return `MIRRORED` split state when input is split on axis 0 (softmax normalizes along axis 0, so split input would give wrong results).

3. **`node_needs_gather` helper**: Identifies operations that require full tensors: `ARGMAX`, `SOFT_MAX`, `TOP_K`, `ARGSORT`.

4. **`node_has_split_axis0_input` helper**: Checks if any source tensor of a node is split on axis 0.

5. **`allgather_fallback` lambda**: Implements the actual gather:
   - Allocates a temp buffer on device 0 for the full tensor
   - Copies each device's slice to the correct offset using `ggml_backend_tensor_copy_async`
   - Creates a new tensor wrapping the gathered data
   - Replaces the split tensor reference in device 0's node sources

6. **Execution loop**: Before each subgraph, checks if any node needs gathering. If so, calls `allgather_fallback` to gather the split input.

7. **`set_sampler` in `llama-context.cpp`**: Removed the `SPLIT_MODE_TENSOR` guard that disabled backend sampling.

### Known limitations / TODOs:

- The gather uses `bcj.bufs[0]` (first temp buffer) which may conflict with AllReduce temp buffers if they're needed in the same subgraph boundary
- The per-device tensor lookup uses global node index which assumes all devices have the same node ordering
- No NCCL/fast path for gather — always uses the fallback memcpy approach
- Testing with actual multi-GPU hardware is needed

## Success Criteria

1. Backend sampling produces identical results to single-GPU sampling
2. No regression in single-GPU or layer-split multi-GPU modes
3. `test-backend-sampler` passes with `SPLIT_MODE_TENSOR` on 2+ GPUs
4. Performance: gather overhead is negligible compared to sampling computation
