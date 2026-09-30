"""The two OpenFold3 weight layouts of the port (xfold/of3.py OF3 + OPENBIND): the variant read off the converted records
(xfold/params.py detect_variant / check_layout_flags), the diffusion transformer's parameter scopes under each layout (the record
spellings the fork's converter writes — preview-2's per-block stack vs openbind's shared LayerNorm + per-super-block Linear, the same as
AlphaFold 3's own), the column-wise pair attention's bias transpose (preview-2 only), and the row-pair adapter's per-block bias rows
against the module's own pair logits under BOTH layouts. CPU, random weights; needs torch + einops (+ triton for xfold.fastnn's import);
skipped otherwise."""
import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("triton")

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
KIT = os.path.join(OPT, "forward", "af3t", "af3_torch")
if KIT not in sys.path:
    sys.path.insert(0, KIT)
if not torch.cuda.is_available():                        # xfold.fastnn's @triton.autotune probes the driver at import (test_tp_xfold_cpu does the same)
    import triton
    triton.autotune = lambda *a, **kw: (lambda fn: fn)
    triton.heuristics = lambda *a, **kw: (lambda fn: fn)

from xfold import of3                                     # noqa: E402
from xfold import params as P                             # noqa: E402

FOURIER = "diffuser/~/diffusion_head/fourier_embedding_weight"   # the OF3-layout marker detect_layout reads
TR = "diffuser/~/diffusion_head/transformer/"
# the converter's record names (juliabuhmann/alphafold3 add_openbind_porter of3_weight_converter.py, as read back from both converted files)
P2_RECORDS = {TR + "__layer_stack_no_per_layer/__layer_stack_no_per_layer/pair_input_layer_norm/scale": (6, 4, 128),
              TR + "__layer_stack_no_per_layer/__layer_stack_no_per_layer/pair_logits_projection/weights": (6, 4, 128, 16)}
OB_RECORDS = {TR + "pair_input_layer_norm/scale": (128,),
              TR + "__layer_stack_with_per_layer/pair_logits_projection/weights": (6, 128, 4, 16)}


@pytest.fixture
def flags():
    saved = (of3.OF3, of3.OPENBIND)
    yield
    of3.OF3, of3.OPENBIND = saved


def test_detect_variant_on_the_records():
    assert P.detect_variant({FOURIER: 0, **{k: 0 for k in P2_RECORDS}}) == "p2"
    assert P.detect_variant({FOURIER: 0, **{k: 0 for k in OB_RECORDS}}) == "openbind"
    for bad in ({FOURIER: 0}, {FOURIER: 0, **{k: 0 for k in P2_RECORDS}, **{k: 0 for k in OB_RECORDS}}):
        with pytest.raises(ValueError):
            P.detect_variant(bad)


def test_flag_helpers_and_marker(tmp_path, flags):
    of3.OF3, of3.OPENBIND = True, False
    assert of3.per_block_pair_bias() and of3.column_bias_transposed()
    of3.set_variant("openbind")
    assert of3.OPENBIND and not of3.per_block_pair_bias() and not of3.column_bias_transposed()
    of3.OF3 = False
    assert not of3.per_block_pair_bias() and not of3.column_bias_transposed()     # AlphaFold 3's own weights: neither, whatever OPENBIND says
    with pytest.raises(ValueError):
        of3.set_variant("p3")
    assert of3.variant_from_dir(tmp_path) is None
    (tmp_path / of3.VARIANT_MARKER).write_text("openbind\n")
    assert of3.variant_from_dir(tmp_path) == "openbind"


def test_check_layout_flags_refuses_a_mismatch(flags):
    p2 = {FOURIER: 0, **{k: 0 for k in P2_RECORDS}}; ob = {FOURIER: 0, **{k: 0 for k in OB_RECORDS}}
    of3.OF3, of3.OPENBIND = True, False
    assert P.check_layout_flags(p2) == "p2"
    with pytest.raises(ValueError, match="openbind"):
        P.check_layout_flags(ob)
    of3.OPENBIND = True
    assert P.check_layout_flags(ob) == "openbind"
    with pytest.raises(ValueError, match="'p2'"):
        P.check_layout_flags(p2)
    of3.OF3 = False
    with pytest.raises(ValueError, match="of3.OF3"):
        P.check_layout_flags(ob)


def _transformer_records(openbind):
    """The translation keys the port expects for a DiffusionTransformer built under the flag, as the loader flattens them."""
    from xfold.nn.diffusion_transformer import DiffusionTransformer
    of3.OF3, of3.OPENBIND = True, openbind
    tr = DiffusionTransformer()                                             # the model's dims: 24 blocks of 4 per super block, c_pair 128, 16 heads
    flat = P._process_translations_dict({"transformer": P.DiffusionTransformerParams(tr)}, _key_prefix=TR[:-len("transformer/")])
    return tr, flat


@pytest.mark.parametrize("openbind,expect,absent", [(False, P2_RECORDS, OB_RECORDS), (True, OB_RECORDS, P2_RECORDS)])
def test_transformer_scopes_follow_the_variant(flags, openbind, expect, absent):
    tr, flat = _transformer_records(openbind)
    assert tr.per_block_pair_bias == (not openbind) and tr.of3
    for k in expect:                                                        # the record spellings (shapes are the converter's; the loader stacks the lists)
        assert k in flat, (k, sorted(x for x in flat if "pair_" in x))
    for k in absent:
        assert k not in flat
    if openbind:                                                            # the per-super-block Linear: 6 x Linear(128 -> 4*16)
        assert len(tr.pair_logits_projection) == 6 and tr.pair_logits_projection[0].weight.shape == (64, 128)
        assert not isinstance(tr.pair_input_layer_norm, torch.nn.ModuleList)
    else:
        assert len(tr.pair_logits_projection) == 24 and tr.pair_logits_projection[0].weight.shape == (16, 128)
        assert len(tr.pair_input_layer_norm) == 24


def _randomise(mod):
    with torch.no_grad():
        for prm in mod.parameters():
            prm.copy_(torch.randn_like(prm) * 0.1)


@pytest.mark.parametrize("openbind", [False, True])
def test_rowpair_block_bias_matches_the_module_under_both_layouts(flags, openbind):
    """rowpair_xfold.dit_block_fns(...).bias on rows [r0, r1) == the module's own pair_logits_for_block, rows sliced, under each layout."""
    pytest.importorskip("opt_core.testing")
    from xfold.nn.diffusion_transformer import DiffusionTransformer
    of3.OF3, of3.OPENBIND = True, openbind
    torch.manual_seed(0)
    tr = DiffusionTransformer(c_act=32, c_single_cond=16, c_pair_cond=24, num_head=4, num_blocks=8, super_block_size=4).eval()
    _randomise(tr)
    spec = importlib.util.spec_from_file_location("rowpair_xfold_variant_test", os.path.join(OPT, "af3_torch_opt", "rowpair_xfold.py"))
    rpx = importlib.util.module_from_spec(spec); spec.loader.exec_module(rpx)
    N = 12; z = torch.randn(N, N, 24)
    mask = torch.ones(N)
    r0, r1 = 3, 9
    for i in range(tr.num_blocks):
        fns = rpx.dit_block_fns(tr, i, mask)
        got = fns.bias(z[r0:r1])                                            # [rows, N, H]
        want = tr.pair_logits_for_block(z, i, {}).permute(1, 2, 0)[r0:r1]   # [H, N, N] -> [N, N, H], my rows
        assert got.shape == (r1 - r0, N, 4)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_column_attention_bias_transpose_is_preview2_only(flags):
    """GridSelfAttention(transpose=True): under openbind the forward equals the AlphaFold 3-layout forward (bias from z[q, k]); under
    preview-2 it differs (bias from the transposed pair) — the module reads xfold.of3 at forward time."""
    from xfold.nn.attention import GridSelfAttention
    torch.manual_seed(1)
    of3.OF3, of3.OPENBIND = True, False
    mod = GridSelfAttention(c_pair=16, num_head=2, transpose=True).eval()
    _randomise(mod)
    N = 6; pair = torch.randn(N, N, 16); mask = torch.ones(N, N)
    with torch.no_grad():
        y_p2 = mod(pair.clone(), mask)
        of3.OPENBIND = True
        y_ob = mod(pair.clone(), mask)
        of3.OF3, of3.OPENBIND = False, False
        y_af3 = mod(pair.clone(), mask)
    torch.testing.assert_close(y_ob, y_af3)
    assert not torch.allclose(y_p2, y_af3, atol=1e-6)
