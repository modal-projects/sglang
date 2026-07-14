# Inkling Architecture Guide

The Inkling model is a decoder-only transformer family with several non-standard pieces: relative attention as the position mechanism, short causal convolution layers, shared-sink MoE routing, and optional communication layouts for sharded sconv execution.

## Overview

At a high level, Inkling follows the standard decoder-only, pre-norm transformer pattern:

```text
token and multimodal embeddings
  -> repeated decoder blocks
  -> final RMSNorm
  -> LM head
```

Each decoder block has an attention sublayer followed by an MLP or MoE sublayer. Unlike a plain transformer block, the Inkling block includes sconv layers around those sublayers:

```text
hidden_states
  -> RMSNorm -> attention -> attention sconv -> residual add
  -> RMSNorm -> dense MLP or MoE -> MLP sconv -> residual add
```

The main architectural differences from a conventional decoder transformer are:

* Inkling uses relative attention for position information instead of absolute position embeddings or RoPE.
* Attention layers are a configurable mix of full causal attention and sliding-window attention.
* Decoder blocks include short per-channel causal convolution layers, called sconv layers.
* Most feed-forward layers are routed MoE layers with a shared expert sink.

## MoE And Shared Expert Sink

A common sigmoid-gated top-k MoE router scores the routed experts, applies a sigmoid before top-k, normalizes over the selected routed experts, and combines only those expert outputs. A gate bias can be used for expert selection only:

```python
scores = router(x)                         # [num_experts]
routed_scores = sigmoid(scores)            # [num_experts]

# Selection path: gate bias can affect which experts are chosen.
selection_scores = routed_scores + gate_bias
expert_ids = top_k(selection_scores, k)    # [k]

# Weighting path: selected weights come from unbiased routed scores.
weights = routed_scores[expert_ids]
weights = weights / sum(weights)           # [k]

y = 0
for i in range(k):
    y += weights[i] * expert[expert_ids[i]](x)
```

Many shared-expert MoE architectures add an always-on shared path next to the routed path:

```python
routed_y = top_k_moe(x)

shared_y = 0
for j in range(num_shared_experts):
    shared_y += shared_expert[j](x)

y = routed_y + shared_y
```

In that baseline, the shared experts are not selected by the router and do not compete with routed experts for router probability mass. They are a parallel dense expert path.

Inkling uses shared experts differently. The router scores both routed experts and shared experts. Top-k selection is performed only over the routed experts. After the routed experts are selected, the selected routed scores are concatenated with the shared-expert scores and normalized together.

Conceptually, each token computes separate selection scores and mixture weights:

```python
scores = router(x)                         # [num_routed + num_shared]
routed_scores = scores[:num_routed]
shared_scores = scores[num_routed:]

# Selection path: sigmoid first; gate bias is used only here.
selection_scores = sigmoid(routed_scores) + gate_bias
routed_ids = top_k(selection_scores, k)

# Weighting path: use original scores, without gate bias.
active_scores = concat(routed_scores[routed_ids], shared_scores)
active_weights = normalize(logsigmoid(active_scores))

routed_weights = active_weights[:k]
shared_weights = active_weights[k:]

output = 0
for i in range(k):
    output += routed_weights[i] * routed_expert[routed_ids[i]](x)
for j in range(num_shared):
    output += shared_weights[j] * shared_expert[j](x)
```

The shared experts therefore compete for probability mass with the selected routed experts, but they are not candidates in the top-k routed selection. This is why they are described as a shared expert sink: they are shared across tokens like the usual shared-expert path, but their per-token weights come from the same normalization as the selected routed experts.

The shared sink experts are dense MLP experts. They can be computed in a batched form, then multiplied by the per-token shared weights and summed into the routed expert output.

## Sconv

Sconv is a short causal convolution over hidden states. For a hidden state at token position `t`, sconv reads the current hidden state and the previous `W - 1` hidden states from the same tensor stream. Current Inkling configurations use `W = 4`, so each sconv reads this four-token window:

```text
[x[t - 3], x[t - 2], x[t - 1], x[t]]
```

Here, a stream means one sequence of tensors inside the layer, such as K, V, the attention output, or the MLP/MoE output.

For each hidden channel, sconv has its own `W` weights. The same channel is read across the last `W` token positions:

```python
# One hidden channel c at token position t.
window = [x[t - 3, c], x[t - 2, c], x[t - 1, c], x[t, c]]
conv = sum(weight[c, i] * window[i] for i in range(W))
y[t, c] = x[t, c] + conv
```

Inkling places sconv in four logical locations in each decoder layer:

* K stream sconv, before K is written to the attention cache.
* V stream sconv, before V is written to the attention cache.
* Attention output sconv, before the attention residual add.
* MLP/MoE output sconv, before the MLP residual add.

Every sconv stream needs a cache of the previous `W - 1` hidden states for each active request. The cache is keyed by request slot and sconv stream. Its logical shape is:

```text
[num_request_slots, W - 1, stream_width]
```

The stream width depends on which sconv stream is being cached:

* K/V sconv uses the tensor-parallel local K/V hidden width for the relevant attention mode.
* Attention-output and MLP-output sconv use the residual-stream hidden width, or a hidden shard of that width when the reduce-scatter sharded sconv optimization is enabled.

The cache must be updated with the final `W - 1` input hidden states after a prefill pass, or shifted by one token after a decode pass. Implementations usually use separate prefill and decode kernels because prefill processes ragged multi-token chunks while decode processes one token per active request.

## Relative Attention

Inkling uses relative attention as its position mechanism. It does not rely on absolute position embeddings or RoPE to inject token order into the residual stream. Instead, each attention layer adds a learned relative-position term to the pre-softmax attention logits.

Relative attention takes Q, K, V, and an additional relative feature `R` for each query token and attention head. It uses `R` only to modify the pre-softmax attention logits. It does not add relative value vectors to the value path.

For each query token, the relative feature is projected to a per-head vector of relative logits. The index `d` below is a causal distance bucket from `0` to `rel_extent - 1`:

```text
rel_logits[t, h, d] = R[t, h] @ rel_projection[:, d]
```

During attention, the pre-softmax attention logit for query position `i`, key position `j`, and head `h` is:

```text
attention_logit(i, j, h) =
  attention_scale * dot(q[i, h], k[j, h])
  + rel_bias(i, j, h)
```

The relative-bias term is selected from the query's `rel_logits` by relative distance:

```text
rel_bias(i, j, h) = rel_logits[i, h, i - j]
```

The relative term is defined only when `0 <= i - j < rel_extent`. Future positions are masked by causal attention. For positions outside the relative extent, the relative contribution is zero in full attention rather than clipped into the furthest distance bucket. In sliding-window attention, positions outside the local window are masked by the attention backend.

This mechanism is related to Shaw et al., "Self-Attention with Relative Position Representations" (<https://arxiv.org/pdf/1803.02155>), in that both use relative position information inside attention rather than absolute position embeddings. The main difference is that Inkling generates a query-conditioned scalar logit from the separate `R` feature, while Shaw et al. use learned relative key vectors and can also add learned relative value vectors. Inkling also uses causal nonnegative distances with a finite extent rather than signed clipped distance buckets.

## Attention Sublayer Structure

The attention sublayer wraps the relative-attention operation with projection, sconv, normalization, and output projection:

```text
normalized hidden states
  -> Q/K/V/R projection
  -> K sconv and V sconv
  -> Q RMSNorm and K RMSNorm
  -> R projection to relative logits
  -> relative attention(Q, K, V, relative logits)
  -> output projection
```

K and V sconv happen after the Q/K/V/R projection and before attention. Q and K RMSNorm happen after K/V sconv and before the attention logits are computed. The R projection produces the relative logits consumed by the relative-attention operation described above.

Full-attention layers can apply a length-dependent log scaling factor to Q and to the relative logits; this gives the model a trained way to adjust logit magnitude for long contexts. Local attention layers do not use that long-context scaling path.

## Attention Layout

Inkling uses both sliding-window attention layers and full attention layers in the same decoder stack. Both attention modes are relative attention. Local layers use sliding-window relative attention, which limits each query to a recent context window. Full layers use full causal relative attention over the available prefix.

The model config chooses which layers are local and which are full-attention layers. The standard Inkling layout uses a repeating group of five sliding-window layers followed by one full-attention layer. The pattern starts with sliding-window attention, so layer 0 is sliding-window and layer 5 is the first full-attention layer. Equivalently, full-attention layers are the layers where `layer_id % 6 == 5`.

## Relative-Attention Kernels

Inkling relative attention needs an attention backend that can add the relative pre-softmax logit term inside the attention computation. A stock attention backend that only accepts Q, K, V, masks, and scalar scale factors is not sufficient.

The supported paths use FlashAttention-4. There are two equivalent ways to integrate the learned relative pre-softmax logit term into FA4.

### Score-Mod Kernel

The score-mod path uses FA4 score modification support. It computes the relative-logit addition inside the attention score callback. For each attention tile, the callback receives the query index, key index, head index, and sequence-length metadata. It computes:

```text
delta = query_position - key_position
```

If `delta` is in range, it loads `rel_logits[query_position, head, delta]` and adds it to the content logit. If `delta` is out of range, it adds zero. Causal masking and sliding-window masking remain the responsibility of the attention backend.

This path works with FA4 score-mod support. The tradeoff is that bias indexing happens in the score callback for the attention tile.

### Sheared-Bias Kernel

The sheared-bias path inserts a preprocessing step between relative-logit projection and FA4 attention:

```text
R projection
  -> rel_logits[query_position, head, relative_distance]
  -> shear rel_logits into column-aligned bias
  -> specialized FA4(Q, K, V, sheared_bias)
```

Before shearing, the relative logits are indexed by distance from the query:

```text
rel_logits[i, h, d]
```

After shearing, each query row is rearranged into the same column coordinate system used by the attention tile:

```text
sheared_bias[i, h, j] = rel_logits[i, h, i - j]
```

when `j` is in the valid causal or sliding-window attention band and `i - j` is within the relative extent. Values outside the valid band are filled with the padding values expected by the attention kernel. The stored sheared row is also padded to match the kernel's tile alignment.

The difference from score-mod is where the indexing happens. Score-mod computes `i - j` inside the attention score callback. Sheared-bias computes a tiled, column-aligned bias layout ahead of time, so the attention kernel can add bias using regular tile loads. That is more specialized, but it can be faster because the inner attention kernel performs regular tile loads instead of per-score callback indexing.

The sheared-bias path requires a Inkling-specialized FA4 fork. It is not supported by stock FA4. The relative extent and padding conventions must match that fork's kernel contract. Implementations commonly require the relative extent to be aligned to the kernel tile size. Local attention also needs the sheared layout to encode which positions outside the local band should be masked.

## Optional: Reduce-Scatter Sharded Sconv

The reduce-scatter sharded sconv optimization applies to the attention-output and MLP/MoE-output sconv layers, which run immediately before their outputs are added to the residual stream. Because sconv is channelwise, each hidden channel is independent. That means these sconv layers can run on a shard of the hidden dimension without changing model math.

Without this optimization, each tensor-parallel attention rank materializes the full residual stream `[T, H]`. Attention-output and MLP-output sconv then run over all `H` channels on every rank, and their full-width sconv caches are also allocated on every rank. That replicates both residual-stream sconv compute and residual-stream sconv cache memory across the tensor-parallel group.

With the optimization, attention produces a hidden-sharded residual stream `[T, H / P]` across the attention tensor-parallel group. The attention output projection uses reduce-scatter instead of an all-reduce, so each rank receives only its local hidden shard. Attention-output sconv then runs over that local hidden shard. MLP-output sconv can use the same hidden-sharded layout after the MLP output has been converted back to the attention layout.

The attention-to-MLP boundary looks like this:

```text
replicated residual-stream path:
  attention output projection
    -> all_reduce over tensor-parallel ranks
    -> each rank has [T, H]
    -> attention-output sconv, cache [slots, W - 1, H]
    -> residual add in [T, H]
    -> keep [T, H], or scatter tokens to [T / P, H] for EP MoE routing/dispatch
    -> MLP/MoE RMSNorm

hidden-sharded residual-stream path:
  attention output projection
    -> reduce_scatter over hidden dimension
    -> rank r has [T, H / P]
    -> attention-output sconv, cache [slots, W - 1, H / P]
    -> residual add in [T, H / P]
    -> all_gather to [T, H] or all_to_all to [T / P, H]
    -> MLP/MoE RMSNorm
```

The layout step before MLP/MoE RMSNorm is not part of the model math. It is a runtime communication step that puts the residual stream in the layout expected by the MLP/MoE implementation. A replicated dense MLP can consume `[T, H]`; an expert-parallel MoE path can consume token-sharded activations `[T / P, H]` for routing and dispatch; a hidden-sharded attention stream starts as `[T, H / P]` and must be converted to one of those layouts first.

The same idea applies to the MLP-output sconv. The MLP output is converted back to the hidden-sharded attention layout, MLP-output sconv runs on `[T, H / P]`, and the cache for that stream is `[slots, W - 1, H / P]`. A full hidden stream is reconstructed only when a later consumer needs `[T, H]`.

The optimization changes communication and tensor layout, not the model function. It is valid because attention-output and MLP-output sconv do not mix channels. The sconv weights and cache for those streams are sharded along the hidden dimension, so their local cache shape becomes:

```text
[num_request_slots, W - 1, H / P]
```

Implementations that combine this with expert-parallel MoE layouts need explicit layout transitions:

* hidden-sharded attention layout: `[T, H / P]`.
* token-sharded EP MoE activation layout: `[T / P, H]`.
* full replicated layout: `[T, H]`.

The hidden-sharded and token-sharded layouts are related by an all-to-all transpose. A full replicated layout requires an all-gather over the hidden dimension. These layout transitions are implementation details, but the cache shape and sconv weight sharding must match the chosen residual-stream layout.

## Multimodal Input Embeddings

Inkling handles image and audio inputs by converting them into token-position embeddings with the same hidden width as text token embeddings. The prompt contains media placeholder token spans. The image and audio towers produce one embedding per media placeholder position, and those embeddings replace the ordinary token embeddings at those positions. After this replacement, the text, image, and audio embeddings form a single sequence consumed by the same decoder stack.

Both multimodal towers are lightweight per-token embedding modules. They do not run a separate deep image or audio transformer before the language model. Their job is to map each preprocessed image patch token or audio token into the decoder hidden space.

### Image Embeddings

The image tower consumes preprocessed image or video patches shaped like:

```text
[num_patches, temporal_patch_size, patch_size, patch_size, channels]
```

The image encoder is a lightweight HMLP patch encoder applied independently to each preprocessed image patch token. It progressively folds local time/space dimensions into the channel dimension, applies linear layers, and uses RMSNorm plus GELU between intermediate layers:

```text
patch pixels
  -> fold time/space into channels
  -> linear
  -> RMSNorm
  -> GELU
  -> ... repeated ...
  -> final linear to decoder hidden width
  -> optional final RMSNorm
```

The final output is one decoder-width embedding per image patch token. These embeddings occupy the image placeholder span in the model input sequence.

### Audio Embeddings

The audio tower consumes quantized dMel features shaped like:

```text
[num_audio_tokens, num_mel_bins]
```

The audio encoder is a lightweight per-token embedding module. For each audio token, every mel bin contributes one discrete value. Inkling uses a separate embedding range for each mel bin, embeds each mel-bin value, and sums the mel-bin embeddings to produce one decoder-width audio embedding:

```python
# One audio token with num_mel_bins quantized dMel values.
audio_embedding = 0
for mel_bin in range(num_mel_bins):
    index = mel_bin * mel_vocab_size + dmel_value[mel_bin]
    audio_embedding += mel_embedding[index]
```

An optional RMSNorm can be applied to the summed audio embedding. The result is one decoder-width embedding per audio token, and those embeddings occupy the audio placeholder span in the model input sequence.
