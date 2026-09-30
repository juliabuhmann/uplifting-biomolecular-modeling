"""Layout flags for the converted OpenFold3 weights in the AF3 haiku layout (sokrypton/alphafold3 `global_config.of3_weights`).
Set ONCE before building the model:  from xfold import of3; of3.OF3 = True
Mirrors the 7 code-path differences of the JAX fork (model.py, modules.py, evoformer.py, atom_cross_attention.py, diffusion_head.py x2, diffusion_transformer.py).

OPENBIND: which OpenFold3 checkpoint layout OF3 refers to (the JAX fork's `global_config.of3_openbind`, juliabuhmann/alphafold3
add_openbind_porter). OpenFold3 >= 0.5.0 ("openbind", of3-ob-2025-06-30-174k) reverted two of preview-2's divergences from AlphaFold 3:
the diffusion transformer's pair LayerNorm is run once on the transformer with one pair-logits Linear per super block (AF3's own layout,
xfold's non-OF3 branch), and the column-wise pair attention's bias is computed from z[q, k] as AF3 does (no transpose). Every other OF3
difference (residue alphabet, element one-hot shift, atom cross-attention masks, bond symmetrisation, Fourier buffers, 833-channel
single conditioning, GAP template slots) applies to both. Ignored unless OF3 is True. The variant is read from the converted
parameters themselves (params.detect_variant: openbind's shared `transformer/pair_input_layer_norm/scale` record vs preview-2's
per-block stack), or from the `of3_variant` marker the fork's converter writes beside them (variant_from_dir)."""
import os

OF3 = False
OPENBIND = False

VARIANT_MARKER = "of3_variant"                  # the fork converter's marker file: "openbind" | "p2"
VARIANTS = ("p2", "openbind")


def per_block_pair_bias() -> bool:
    """True when the diffusion transformer carries a pair LayerNorm + logits Linear per block (OpenFold3 preview-2); False for the
    AlphaFold 3 layout (one shared LayerNorm, one Linear per super block) — AF3's own weights and OpenFold3 openbind alike."""
    return bool(OF3) and not bool(OPENBIND)


def column_bias_transposed() -> bool:
    """True when the column-wise pair attention's bias is Linear(z[k, q]) (OpenFold3 preview-2); False for AF3 and openbind (z[q, k])."""
    return bool(OF3) and not bool(OPENBIND)


def variant_from_dir(params_dir) -> "str | None":
    """The `of3_variant` marker beside the converted parameters ("openbind" | "p2"), None when absent or unreadable."""
    try:
        with open(os.path.join(str(params_dir), VARIANT_MARKER), encoding="utf-8") as f:
            word = f.read().strip()
    except OSError:
        return None
    return word if word in VARIANTS else None


def set_variant(word: str) -> None:
    """Set OPENBIND from a variant word ("openbind" | "p2"); must run BEFORE the model is constructed."""
    if word not in VARIANTS:
        raise ValueError(f"unknown OF3 variant {word!r} (one of {VARIANTS})")
    global OPENBIND
    OPENBIND = word == "openbind"
