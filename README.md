# brainllm
Brain-inspired Qwen 4-bit: every layer gets live neurotransmitter levels with noise, PID homeostasis and push/pull prompts, with no weight edits.
# NeuroQwen

Qwen (4-bit) with a simulated neurotransmitter system. Every transformer layer has its own live vector of six chemicals (dopamine, serotonin, norepinephrine, acetylcholine, glutamate, GABA). Each level sits at 1.0 when balanced, and the model's behaviour changes as the levels drift.

## How it works

On every generated token, for every layer:

1. **Noise:** random kicks perturb each level.
2. **Homeostasis:** a PID controller pulls each level back to 1.0.
3. **Cross-talk:** opposing chemicals inhibit each other (glutamate vs GABA, dopamine vs serotonin).
4. **Diffusion:** neighbouring layers leak chemistry into each other.
5. **Push and pull prompts:**
   - *Push:* the environment generates a random prompt that shoves a chemical.
   - *Pull:* the controller generates a counter-prompt against the biggest deviation.

## What each chemical does

| Chemical | Effect |
|---|---|
| Glutamate / GABA | Scales MLP output (excite vs inhibit) |
| Acetylcholine | Scales attention queries (attention gain) |
| Dopamine | Boosts attention output and logit confidence |
| Serotonin | Damps attention output and narrows top-p |
| Norepinephrine | Adds activation noise and raises temperature |

Chemicals act with different strength by depth (for example dopamine is stronger in late layers, acetylcholine in early ones), and each layer has its own sensitivity.

## Weights

The 4-bit weights are never modified. Scaling a linear layer's output by `g` is equivalent to using weights `g * W`, so each layer runs with live effective weights `W_eff = g_layer(t) * W` without writing to the quantized tensors. Remove the hooks and you get plain Qwen back.

## Usage

1. Open `neuro_qwen.ipynb` in Google Colab.
2. Runtime > Change runtime type > T4 GPU.
3. Run all.

Included: a per-layer live weight table (`live_weights`), layer-by-time heatmaps (`plot_layers`), targeted layer pushes, and ablations (no control, no noise, high noise).

## Tuning

Hook strengths and chemistry constants are heuristic starting points. If output becomes gibberish, lower them. If nothing changes, raise them. Default model is `Qwen/Qwen2.5-1.5B-Instruct`; change `MODEL` to try others.

## Status

Experimental. This is a metaphor-driven research toy, not a validated model of the brain.

## Files

- `neuro_qwen.ipynb`: Colab notebook
- `neuro_qwen.py`: same code as a script
