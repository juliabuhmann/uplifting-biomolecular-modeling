"""The memory-reach row-blocked variants (commits 8d50458..783410c) vs the whole-tensor statements they replace, CPU fp32:
DistogramHead.forward, ConfidenceHead._forward_rows, DiffusionTransformer._precompute_pair_logits_rows (both pair-bias layouts),
Evoformer._relative_encoding's row path, SingleTemplateEmbedding.construct_input_rows; and template_dedupe.forward's release of each
distinct slot's embedding after its last use (same adds, same order as a plain per-slot sum).

The row path is forced by monkeypatching the module's *_MIN_TOKENS constant to 0 and its *_BLOCK_GIB constant tiny (BLOCK_GIB below:
blocks clamp to the 64-row minimum, several blocks at N not a multiple of 64); the whole path runs with *_MIN_TOKENS huge. Random
(seeded) weights and inputs: the equivalence is row path == whole path of the SAME module. Needs torch + einops (xfold imports triton:
skipped where it does not import). Every comparison records its max abs diff in MAXDIFF (printed with `pytest -s`)."""
import dataclasses
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
KIT = os.path.join(OPT, "forward", "af3t", "af3_torch")                 # the xfold package's parent
if KIT not in sys.path:
    sys.path.insert(0, KIT)
try:
    import xfold.nn.head  # noqa: F401  (imports triton through xfold.fastnn)
except ImportError as e:                                                 # pragma: no cover
    pytest.skip(f"xfold does not import here: {e}", allow_module_level=True)

ATOL = RTOL = 1e-5
BLOCK_GIB = 1e-4          # tiny transient budget: every variant's block size clamps to its 64-row minimum
WHOLE = 10 ** 9           # a *_MIN_TOKENS no test input reaches: the whole-tensor statements
N = 150                   # tokens: not a multiple of 64 (blocks 64, 64, 22)
N_CONF = 100              # tokens for the confidence head (4 pairformer blocks on CPU; blocks 64, 36)
DENSE_ATOMS = 24          # xfold.constants.atom_types.DENSE_ATOM_NUM (asserted below)
MAXDIFF = {}


def _close(name, got, ref):
    d = float((got.double() - ref.double()).abs().max()) if got.numel() else 0.0
    MAXDIFF[name] = d
    print(f"[memreach] {name}: max abs diff {d:.3e}")
    assert got.shape == ref.shape and got.dtype == ref.dtype, (name, got.shape, ref.shape, got.dtype, ref.dtype)
    torch.testing.assert_close(got, ref, atol=ATOL, rtol=RTOL, msg=lambda m: f"{name}: {m}")


def _calls(module):
    """A list that grows by one per forward call of `module` (proves which path ran: one call whole, one per row block on the row path)."""
    seen = []
    module.register_forward_hook(lambda mod, inp, out: seen.append(tuple(inp[0].shape)))
    return seen


def _n_blocks(n, rows=64):
    return -(-n // rows)


def _randomise(module, seed, scale=0.1):
    """Seeded random parameters (the inits of LayerNorm / zero-init Linears are degenerate)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return module.eval()


def _chains(n, n_chains, g):
    """Sorted chain ids 1..n_chains, every chain non-empty."""
    cuts = torch.sort(torch.randperm(n - 1, generator=g)[: n_chains - 1] + 1).values
    asym = torch.zeros(n, dtype=torch.int64)
    asym[cuts] = 1
    return torch.cumsum(asym, 0) + 1


def _token_features(n, seed):
    """token_index / residue_index / asym_id / entity_id / sym_id: 4 chains, chains 1 and 3 one entity (sym 1, 2), residue gaps."""
    g = torch.Generator().manual_seed(seed)
    asym = _chains(n, 4, g)
    entity = torch.tensor([0, 1, 2, 1, 3])[asym]
    sym = torch.tensor([0, 1, 1, 2, 1])[asym]
    resid = torch.zeros(n, dtype=torch.int64)
    for c in asym.unique():
        m = asym == c
        resid[m] = torch.cumsum(torch.randint(1, 3, (int(m.sum()),), generator=g), 0)
    return types.SimpleNamespace(token_index=torch.arange(1, n + 1, dtype=torch.int32), residue_index=resid.to(torch.int32),
                                 asym_id=asym.to(torch.int32), entity_id=entity.to(torch.int32), sym_id=sym.to(torch.int32))


@pytest.fixture(autouse=True)
def _cpu_fp32():
    torch.set_default_dtype(torch.float32)
    yield


# 1. DistogramHead.forward
def test_distogram_head_rows_equals_whole(monkeypatch):
    from xfold.nn import head
    m = _randomise(head.DistogramHead(), seed=1)
    g = torch.Generator().manual_seed(11)
    mask = torch.ones(N, dtype=torch.bool)
    mask[torch.randperm(N, generator=g)[:7]] = False
    batch = types.SimpleNamespace(token_features=types.SimpleNamespace(mask=mask))
    pair = torch.randn(N, N, 128, generator=g)
    pair0 = pair.clone()
    monkeypatch.setattr(head, "ROWS_BLOCK_GIB", BLOCK_GIB)
    assert head._block_rows(N, N * m.num_bins * 4 * 3) == 64
    seen = _calls(m.half_logits)
    with torch.no_grad():
        monkeypatch.setattr(head, "ROWS_MIN_TOKENS", WHOLE)
        ref = m(batch, {"pair": pair})
        monkeypatch.setattr(head, "ROWS_MIN_TOKENS", 0)
        got = m(batch, {"pair": pair})
    assert len(seen) == 1 + _n_blocks(N), seen                              # whole: one call; rows: one per 64-row block
    assert torch.equal(pair, pair0)
    assert torch.equal(got["bin_edges"], ref["bin_edges"])
    _close("distogram.contact_probs", got["contact_probs"], ref["contact_probs"])


# 2. ConfidenceHead.forward vs ._forward_rows
CONF_KEYS = ("full_pde", "full_pae", "tmscore_adjusted_pae_global", "tmscore_adjusted_pae_interface", "average_pde",
             "predicted_lddt", "predicted_experimentally_resolved")


@pytest.fixture(scope="module")
def confidence_outputs():
    from xfold.constants import atom_types
    from xfold.nn import head
    from xfold.nn.atom_layout import GatherInfo
    assert atom_types.DENSE_ATOM_NUM == DENSE_ATOMS
    n = N_CONF
    m = _randomise(head.ConfidenceHead(), seed=2)
    g = torch.Generator().manual_seed(22)
    seq_mask = torch.ones(n, dtype=torch.bool)
    seq_mask[torch.randperm(n, generator=g)[:5]] = False
    asym = _chains(n, 4, g).to(torch.int32)
    pos = torch.randn(n, DENSE_ATOMS, 3, generator=g) * 8.0
    slot = torch.randint(0, 4, (n,), generator=g)                                     # one atom per token in the flattened [N*24] layout
    gmask = torch.ones(n, dtype=torch.bool)
    gmask[torch.randperm(n, generator=g)[:3]] = False
    pb = GatherInfo(gather_idxs=(torch.arange(n) * DENSE_ATOMS + slot).to(torch.int64), gather_mask=gmask,
                    input_shape=torch.tensor((n, DENSE_ATOMS)))
    emb = {"pair": torch.randn(n, n, 128, generator=g), "single": torch.randn(n, 384, generator=g),
           "target_feat": torch.randn(n, 447, generator=g)}
    emb0 = {k: v.clone() for k, v in emb.items()}
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(head, "ROWS_BLOCK_GIB", BLOCK_GIB)
        assert head._block_rows(n, n * m.c_pair * 4 * 3) == 64
        seen_pde, seen_pae = _calls(m.left_half_distance_logits), _calls(m.pae_logits)
        with torch.no_grad():
            mp.setattr(head, "ROWS_MIN_TOKENS", WHOLE)
            ref = m(pos.clone(), emb, seq_mask, pb, asym)
            pair_after_whole = torch.equal(emb["pair"], emb0["pair"])
            mp.setattr(head, "ROWS_MIN_TOKENS", 0)
            got = m(pos.clone(), emb, seq_mask, pb, asym)
            pair_after_rows = torch.equal(emb["pair"], emb0["pair"])
        assert len(seen_pde) == len(seen_pae) == 1 + _n_blocks(n), (seen_pde, seen_pae)
    finally:
        mp.undo()
    return dict(ref=ref, got=got, emb=emb, emb0=emb0, pair_after_whole=pair_after_whole, pair_after_rows=pair_after_rows)


@pytest.mark.parametrize("key", CONF_KEYS)
def test_confidence_head_rows_equals_whole(confidence_outputs, key):
    ref, got = confidence_outputs["ref"], confidence_outputs["got"]
    assert set(got) == set(ref)
    _close(f"confidence.{key}", got[key], ref[key])


def test_confidence_head_leaves_embeddings_unmodified(confidence_outputs):
    c = confidence_outputs
    assert c["pair_after_whole"], "forward() modified embeddings['pair']"
    assert c["pair_after_rows"], "_forward_rows() modified embeddings['pair']"
    for k in ("single", "target_feat"):
        assert torch.equal(c["emb"][k], c["emb0"][k]), k


# 3. DiffusionTransformer.precompute_pair_logits vs _precompute_pair_logits_rows, both layouts
@pytest.mark.parametrize("of3_flag,openbind", [(True, False), (True, True), (False, False)], ids=["of3-p2-per-block", "of3-openbind", "af3"])
def test_dit_pair_logits_rows_equals_whole(monkeypatch, of3_flag, openbind):
    from xfold import of3
    from xfold.nn import diffusion_transformer as dt
    monkeypatch.setattr(of3, "OF3", of3_flag)
    monkeypatch.setattr(of3, "OPENBIND", openbind)
    m = _randomise(dt.DiffusionTransformer(c_act=64, c_single_cond=32, c_pair_cond=128, num_head=4, num_blocks=8, super_block_size=4), seed=3)
    assert m.per_block_pair_bias == (of3_flag and not openbind)
    pc = torch.randn(N, N, 128, generator=torch.Generator().manual_seed(33))
    pc0 = pc.clone()
    monkeypatch.setattr(dt, "ROWS_BLOCK_GIB", BLOCK_GIB)
    seen = _calls(m.pair_logits_projection[0])
    with torch.no_grad():
        monkeypatch.setattr(dt, "ROWS_MIN_TOKENS", WHOLE)
        ref = m.precompute_pair_logits(pc)
        monkeypatch.setattr(dt, "ROWS_MIN_TOKENS", 0)
        got = m.precompute_pair_logits(pc)
    assert len(seen) == 1 + _n_blocks(N), seen
    assert torch.equal(pc, pc0)
    assert ref.shape == (8, 4, N, N)
    _close(f"dit.pair_logits[{'per_block' if m.per_block_pair_bias else 'super_block'}{'-openbind' if openbind else ''}]", got, ref)


# 4. Evoformer._relative_encoding
def test_evoformer_relative_encoding_rows_equals_whole(monkeypatch):
    from xfold import alphafold3 as af
    holder = _randomise(torch.nn.Module(), seed=4)
    holder.position_activations = torch.nn.Linear(139, 128, bias=False)
    _randomise(holder, seed=4)
    batch = types.SimpleNamespace(token_features=_token_features(N, seed=44))
    pair = torch.randn(N, N, 128, generator=torch.Generator().manual_seed(45))
    monkeypatch.setattr(af, "REL_ROWS_BLOCK_GIB", BLOCK_GIB)
    seen = _calls(holder.position_activations)
    with torch.no_grad():
        monkeypatch.setattr(af, "REL_ROWS_MIN_TOKENS", WHOLE)
        ref = af.Evoformer._relative_encoding(holder, batch, pair.clone())
        monkeypatch.setattr(af, "REL_ROWS_MIN_TOKENS", 0)
        got = af.Evoformer._relative_encoding(holder, batch, pair.clone())
    assert len(seen) == 1 + _n_blocks(N), seen
    _close("evoformer.relative_encoding", got, ref)


# 5. SingleTemplateEmbedding.construct_input vs construct_input_rows
@pytest.mark.parametrize("of3_flag", [False, True], ids=["af3", "of3"])
@pytest.mark.parametrize("empty_slot", [False, True], ids=["templated", "empty-slot"])
def test_template_construct_input_rows_equals_whole(monkeypatch, of3_flag, empty_slot):
    from xfold import of3
    from xfold.nn import template
    monkeypatch.setattr(of3, "OF3", of3_flag)
    m = _randomise(template.SingleTemplateEmbedding(num_channels=64), seed=5)
    g = torch.Generator().manual_seed(55)
    aatype = torch.randint(0, 20, (N,), generator=g)
    aatype[torch.randperm(N, generator=g)[:6]] = 20
    aatype[torch.randperm(N, generator=g)[:4]] = 21
    pos = torch.randn(N, DENSE_ATOMS, 3, generator=g) * 6.0
    amask = torch.rand(N, DENSE_ATOMS, generator=g) < 0.7
    amask[:, :3] = torch.rand(N, 3, generator=g) < 0.9                                 # backbone mostly present
    if empty_slot:
        amask[:] = False
    asym = _chains(N, 3, g)
    multichain = (asym[:, None] == asym[None, :]).to(torch.float32)
    q = torch.randn(N, N, 128, generator=g)
    q0 = q.clone()

    def tmpl():           # construct_input multiplies atom_positions in place: a fresh copy per call
        return types.SimpleNamespace(aatype=aatype.to(torch.int32).clone(), atom_positions=pos.clone(), atom_mask=amask.clone())

    seen = _calls(m.template_pair_embedding_8)
    with torch.no_grad():
        ref = m.construct_input(q, tmpl(), multichain)
        got = m.construct_input_rows(q, tmpl(), multichain)              # module default budget: one block
        monkeypatch.setattr(template, "ROWS_BLOCK_GIB", BLOCK_GIB)
        got_small = m.construct_input_rows(q, tmpl(), multichain)
    assert len(seen) == 1 + 1 + _n_blocks(N), seen                          # whole, rows at the default budget (one block), 64-row blocks
    assert torch.equal(q, q0)
    n_rows = max(64, min(N, int(BLOCK_GIB * 2 ** 30 // (N * 128 * 4 * 4)) // 64 * 64))
    assert n_rows == 64
    tag = f"{'of3' if of3_flag else 'af3'}{'-empty' if empty_slot else ''}"
    _close(f"template.construct_input[{tag}, one block]", got, ref)
    _close(f"template.construct_input[{tag}, 64-row blocks]", got_small, ref)


def test_template_forward_takes_rows_path_from_constant(monkeypatch):
    """SingleTemplateEmbedding.forward dispatches on ROWS_MIN_TOKENS (construct_input_rows at/above it, construct_input below)."""
    from xfold.nn import template
    m = template.SingleTemplateEmbedding(num_channels=8)
    m.template_embedding_iteration = torch.nn.ModuleList()
    seen = []
    monkeypatch.setattr(m, "construct_input", lambda q, t, mc: (seen.append("whole"), torch.zeros(*q.shape[:2], 8))[1])
    monkeypatch.setattr(m, "construct_input_rows", lambda q, t, mc: (seen.append("rows"), torch.zeros(*q.shape[:2], 8))[1])
    q = torch.zeros(5, 5, 128)
    monkeypatch.setattr(template, "ROWS_MIN_TOKENS", 6)
    m(q, None, None, None)
    monkeypatch.setattr(template, "ROWS_MIN_TOKENS", 5)
    m(q, None, None, None)
    assert seen == ["whole", "rows"]


# 6. template_dedupe.forward: release after last use == a plain per-slot sum in slot order
@dataclasses.dataclass
class _Templates:
    aatype: torch.Tensor
    atom_positions: torch.Tensor

    def __getitem__(self, t):
        return _Templates(self.aatype[t], self.atom_positions[t])


@pytest.mark.parametrize("slot_values", [(0, 1, 0, 2, 1, 1), (0, 0, 0, 0), (0, 1, 2, 3), (3, 3, 1, 3)])
def test_template_dedupe_forward_equals_reference_sum(slot_values):
    from af3_torch_opt import template_dedupe as td
    n, c, T = 13, 6, len(slot_values)
    g = torch.Generator().manual_seed(66)
    base_pos = torch.randn(max(slot_values) + 1, n, 4, 3, generator=g)
    tm = _Templates(aatype=torch.tensor(slot_values, dtype=torch.int32)[:, None].expand(T, n).contiguous(),
                    atom_positions=base_pos[list(slot_values)].contiguous())
    calls = []

    def single_template_embedding(query, t, pad, mc):              # deterministic per slot content
        calls.append(int(t.aatype[0]))
        k = int(t.aatype[0])
        return torch.sin(query[..., :1] * (k + 1) + t.atom_positions.sum() + torch.arange(c, dtype=query.dtype))

    out_lin = _randomise(torch.nn.Linear(c, 5), seed=6)
    stub = types.SimpleNamespace(num_channels=c, single_template_embedding=single_template_embedding, output_linear=out_lin)
    q = torch.randn(n, n, 3, generator=g)
    td.begin_item()
    with torch.no_grad():
        got = td.forward(stub, q, tm, None, None)
        ref_sum = torch.zeros(n, n, c)
        for t in range(T):                                          # the stock TemplateEmbedding.forward: every slot, slot order
            ref_sum += single_template_embedding(q, tm[t], None, None)
        ref = out_lin(torch.relu(ref_sum / (1e-7 + T)))
    distinct = list(dict.fromkeys(slot_values))
    assert calls[: len(distinct)] == distinct and len(calls) == len(distinct) + T   # dedupe: one call per distinct slot (then T reference calls)
    MAXDIFF[f"template_dedupe{slot_values}"] = float((got - ref).abs().max())
    assert torch.equal(got, ref), f"not bitwise: max abs diff {MAXDIFF[f'template_dedupe{slot_values}']:.3e}"
    td.take()


def teardown_module(module):
    if MAXDIFF:
        print("\n[memreach] max abs diffs:\n" + "\n".join(f"  {k}: {v:.3e}" for k, v in MAXDIFF.items()))
