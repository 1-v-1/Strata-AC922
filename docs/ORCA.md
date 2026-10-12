# OrcaRouter IQ3_XXS compatibility

Validated on Linux with an RTX 5090 (32 GB), Ryzen 9 9950X3D and 128 GB RAM on 2026-09-27.

The target is `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`, **IQ3_XXS** (two shards, 85.20 GB total).
It has Strata's Qwen4Exp geometry, but ordinary quantization also compresses the hyperconnection,
small attention and PLE projections which Strata's kernels read as BF16.

`tools/iq_pack.py --compat-bf16` expands only those projections, rounding to BF16 using nearest-even.
This adds rounding relative to the GGUF's dequantized weights; it does not recover the original BF16
checkpoint or its quality. Expert weights, native attention weights, embeddings and the 28.8 GB PLE
lookup table remain unchanged. The latter stays on disk. The pack records its conversions in
`compat-bf16.json`.

The native loader retains this pack's converted PLE key. Its existing native Q2_0 key path remains
available to the original model. Hugging Face snapshot symlinks are supported; all split shards must
finish downloading before packing. Pass the snapshot filename, not its hash-named blob target.

## Preparation

Build Strata normally using the pinned llama.cpp dependency. Python needs numpy, regex and gguf-py
from that dependency (`STRATA_GGUF_PY` can point to its `gguf-py` directory).

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf \
  --out packs/orca-iq3_xxs --compat-bf16
```

Use this model's own tokenizer exported into the pack. Do not share another model's `dense.bin`,
and do not rename the Orca files to impersonate one of setup's GSQ-RCO models.

Strata's persistent server also uses the MTP runtime (optional: without `--mtp` it drafts by lookup only), prepared by
the existing tools:

```sh
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

This uses the original model's draft head, as setup does for Swift. The target model verifies its
proposals; draft acceptance and performance must be measured for this fine-tune.

## Local server

Save the following as `strata-orca-iq3_xxs.json` at the repository root, replacing both `/path/to/`
entries with the **same first shard**. Orca's PLE table is in shard 1. Start with a 32K context and
512-token prefill chunks; the expert cache sizes itself to available VRAM. The expert arena alone
needs about 49.8 GiB of available system RAM, plus draft/runtime buffers and other applications.

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/orca-iq3_xxs",
    "--native", "/path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf",
    "--ple-gguf", "/path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "512", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "32768", "--kv", "int8"
  ],
  "cwd": ".",
  "tokenizer": "packs/orca-iq3_xxs/tokenizer",
  "model_name": "orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs",
  "log": "strata-orca-iq3_xxs.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

Run from the repository root:

```sh
.venv/bin/python -m serve.server --engine strata --config strata-orca-iq3_xxs.json --port 8080
```

The web interface is at `http://127.0.0.1:8080`; API clients use `http://127.0.0.1:8080/v1`.
Vision is not configured in this text-only example. Performance of the original GSQ-RCO model
does not establish this fine-tune's speed or accuracy.

## Reading those weights as FP16 at load (opt-in)

`STRATA_FP16=load` with `STRATA_FP16_GATES=1` makes the engine read the 483 whitelisted tensors from the
shards at load and keep them as FP16, instead of the BF16 copy `--compat-bf16` wrote into the pack. Nothing
is re-packed: the pack's rows stay `Bf16InF32` and the switch replaces them at load. The two variables are
off by default, so the default path stays byte-identical.

This model stores those tensors in five forms, and the loader now decodes each of them:

| group | tensors | stored as |
|---|---:|---|
| the projections read as FP16 activations | 219 | Q6_K 97, Q8_0 97, BF16 24, IQ3_XXS 1 |
| the gates (`ffn_gate_inp`, `hc_attn_inject`, `hc_ffn_inject`, `ssm_alpha`, `ssm_beta`) | 264 | F32 96, Q6_K 96, IQ3_XXS 72 |

GSQ-RCO's files keep all 483 pre-widened to BF16, so this model is the first place the Q6_K and IQ3_XXS
forms appear. Before this change `build()` rejected the first Q6_K tensor it met, so the switch could not be
used on this pack at all.

Checked without a GPU, on bytes both sides read:

- `dequant_iq3_xxs_test`: 4,096 synthetic blocks (all-zero, all-ones, random scale, grid indices and sign
  words), every stored fp16 bit compared against a reference that re-derives the encoding from a flat index
  -> 0 mismatches. Perturbing the reference's grid index by one gives 446,942, so the check is not empty.
- the first 64 blocks of two tensors from the real shards, against a pure-Python reference that re-derives
  each encoding from a flat index. `strata-dequant` and the reference print the same line for both:
  `blk.0.ssm_alpha.weight` (IQ3_XXS, 6,272 B) `sum=-3.786504358 sumabs=201.6124055`, first eight
  `-0.00596792 -0.00835508 0.00119358 0.0185005 0.00596792 -0.0131294 0.00119358 -0.0185005`;
  `output_hc_down.weight` (Q6_K, 13,440 B) `sum=-4.664942384 sumabs=339.1567574`, first eight
  `0.00388813 -0 0.0116644 0.031105 0.0155525 0.0116644 -0.0330491 0.0272169`.
- `LoadConverted::build` on those same two synthetic files: one FP16 tensor each, 32,768 bytes, 0 subnormal,
  0 flushed. Before the change it returned "output_hc_down.weight is Q6_K in the GGUF" and refused to start.
- speed and quality with this pack are **not** measured here. The +14-17% prefill figure in `IBM_AC922.md`
  was measured on UD-Q4_K_XL on 4x Tesla V100; this fine-tune's numbers are its own.

## Scope and validation

- IQ3_XXS is the target of this change; other Orca quantizations are not validated.
- IQ3_M additionally uses Q5_0 expert down matrices, which the native GPU expert path does not support.
- The installer model menu is unchanged. This is an explicit local packing workflow.
- Focused conversion and split-file tests: `.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py`.
- All eight packing tests, 17 server tests and the GPU gated-residual parity check passed.
- The complete model packed successfully: 460 tensors converted, 1.39 GiB of converted BF16 weights;
  1.43 GiB total packed dense weights. The 48 expert layers remain IQ3_XXS gate/up and IQ4_NL down.
- Real server checks passed: arithmetic, Python code, translation, a longer explanation, multi-turn context
  reuse (25 prompt tokens reused), streaming cancellation and a subsequent correct response. The engine
  PID stayed unchanged. These are smoke tests, not a quality benchmark or a full-context stress test.
- With the configuration above, a 111-token explanation generated at 77.7 tokens/s (engine decode timing);
  total request time was 1.77 seconds. This single short measurement is not a general throughput claim.
