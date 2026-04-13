# OneTrans PyTorch

PyTorch reproduction of the core OneTrans architecture from:
`https://arxiv.org/html/2510.26104v3`

## Scope

This project reproduces the model architecture described in the paper:

- unified non-sequential and sequential tokenization
- mixed shared/token-specific causal attention
- mixed shared/token-specific FFN
- pyramid token-pruning stack
- a simple ranking head on top of the backbone

This implementation focuses on the research architecture. Production-only
optimizations such as FlashAttention-2 kernels, distributed training, and full
cross-request KV caching are left as extension points instead of kernel-level
reproduction.

## Structure

- `src/model_module/model/schema.py`: model and input schema
- `src/model_module/model/tokenizer.py`: NS/S tokenization modules
- `src/model_module/model/layers.py`: RMSNorm, mixed attention, mixed FFN, blocks
- `src/model_module/model/onetrans.py`: backbone and ranking model
- `run/pipeline/training/demo_forward.py`: minimal dummy forward example

## Input Format

The model expects preprocessed tensors after feature embedding/bucketization.

`non_sequential`:
- dict of `{feature_name: tensor[B, D]}`

`sequential`:
- dict of `{sequence_name: {'values': tensor[B, L, D], 'mask': bool[B, L], 'timestamps': tensor[B, L]}}`

For `timestamp_aware` merge mode, `timestamps` is required.

## Example

```bash
python run/pipeline/training/demo_forward.py
```

## Notes

- The paper does not fully specify the final task head; this repo uses a simple
  MLP head over the final non-sequential tokens for end-to-end testing.
- The default configuration follows the paper's OneTransS spirit: 6 layers,
  hidden size 256, 4 heads, Auto-Split tokenizer, timestamp-aware fusion.
