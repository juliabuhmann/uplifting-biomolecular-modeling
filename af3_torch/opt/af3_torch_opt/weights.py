"""`run.sh install --weights DIR [--fetch]` — the weights step of the install: the converted checkpoint under DIR checked against stock/PINS.json,
and, with --fetch, made when DIR has none.

The kit's weights are OpenFold3 parameters converted to this port's layout — stock/PINS.json `variants.<v>.converted` (of3_ported_weights.bin.zst,
one spelling for every variant; a directory holds ONE of them) with `variants.<v>.conventions` (of3_conventions.json, optional) beside it. Two
variants: `p2` (OpenFold3-preview2, the default) and `ob` (OpenFold3 openbind, >= 0.5.0); the model process reads the variant off the converted
records themselves (xfold/params.py detect_variant), so `--variant` only says WHICH public checkpoint --fetch downloads and converts. The step judges
DIR with the kit's ONE digest judge, stock/check_pins.py `digest_report` (the words `run.sh check` prints too):
  pinned   — DIR holds the pinned bytes of either variant: WEIGHTS OK naming it, exit 0;
  unpinned — DIR holds another parameters file in that layout (*.bin.zst | *.bin): WEIGHTS UNPINNED, exit 0 (it runs; the kit's tests and
             timings cover the pinned checkpoints only — the same rule every verb applies);
  missing  — no parameters file under DIR: WEIGHTS MISSING with the two ways to obtain it, exit 1 — unless --fetch: then the variant's public
             checkpoint (`variants.<v>.checkpoint`: url, sha256) is downloaded into DIR/checkpoint/ (a copy already there with the pinned digest
             is kept), converted into DIR by the reference fork's own converter (`variants.<v>.converter`: AF3_TORCH_JAX_REPO's
             convert_of3_weights.py on AF3_TORCH_JAX_PY with the checkout's src/ first on PYTHONPATH — the converter of the pinned commit, whatever
             wheel the JAX environment carries; it needs that environment's torch (CPU) and zstandard), and DIR is judged again:
             WEIGHTS OK | WEIGHTS UNPINNED as above, or WEIGHTS FETCH FAILED / WEIGHTS CONVERT FAILED by name, exit 1. Nothing under DIR is deleted.
Without --fetch nothing is downloaded. DIR is then the value of AF3_TORCH_PARAMS_DIR (README.md 'Setup').

    python -m af3_torch_opt.weights DIR [--fetch] [--variant p2|ob]        (exit: 0 ok · 1 missing / failed · 2 usage)
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, List, Optional

from . import stack

PREFIX = "[af3-torch-opt install]"
EXIT_OK, EXIT_MISSING, EXIT_USAGE = 0, 1, 2
DEFAULT_VARIANT = "p2"


def check(directory: str, pins: Optional[dict] = None, judge: Optional[Callable[[dict], dict]] = None, out=sys.stdout, fetch: bool = False,
          opener=urllib.request.urlopen, variant: str = DEFAULT_VARIANT) -> int:
    """Judge the checkpoint under `directory` against the pins and print ONE verdict line. `pins` defaults to stock/PINS.json, `judge` to
    stock/check_pins.py digest_report (it reads AF3_TORCH_PARAMS_DIR, which this step sets to `directory` for the judgement). `fetch`: a
    directory without a checkpoint gets one (fetch_and_convert, of `variant`) and is judged again. The verdict names the variant whose pinned
    bytes the directory holds; `variant` (p2 | ob) names the one --fetch obtains and the one a MISSING line describes."""
    pins = stack.pins() if pins is None else pins
    judge = stack.pins_tool().digest_report if judge is None else judge
    if variant not in pins["variants"]:
        print(f"{PREFIX} WEIGHTS USAGE: --variant {variant} is not one of {', '.join(pins['variants'])} (stock/PINS.json variants)", file=out, flush=True)
        return EXIT_USAGE
    d = os.path.abspath(directory)
    os.environ["AF3_TORCH_PARAMS_DIR"] = d
    rep = judge(pins)
    held = rep.get("variant") or next((v for v, r in (rep.get("variants") or {}).items() if r.get("pinned")), None)   # the variant whose pinned bytes DIR holds, if any
    spec = pins["variants"][held or variant]; pinned, conv = spec["converted"], spec.get("conventions")
    vrep = (rep.get("variants") or {}).get(held or variant) or {}
    crep = vrep.get("conventions") or {}
    conv_words = ""
    if conv:
        conv_words = (f"; {conv['file']} beside it " + ("has the pinned digest" if crep.get("ok") else "is present with another digest (reported, not judged)" if crep.get("present")
                      else "is absent (optional: the model steps pass it only when present)"))
    verdict = rep.get("verdict")
    if verdict == "pinned":
        print(f"{PREFIX} WEIGHTS OK: {rep['file']} is the pinned {spec['name']} checkpoint (sha256 {rep['sha256'][:16]}…, {pinned['bytes']} bytes){conv_words} "
              f"— export AF3_TORCH_PARAMS_DIR={d}", file=out, flush=True)
        return EXIT_OK
    if verdict == "unpinned":
        others = "; ".join(f"{pins['variants'][v]['name']} {pins['variants'][v]['converted']['sha256'][:16]}…, {pins['variants'][v]['converted']['bytes']} bytes" for v in pins["variants"])
        print(f"{PREFIX} WEIGHTS UNPINNED: {rep['file']} (sha256 {rep['sha256'][:16]}…) is not a pinned {pinned['file']} ({others}) "
              f"— it runs: {stack.UNPINNED_WORDS}{conv_words} — export AF3_TORCH_PARAMS_DIR={d}", file=out, flush=True)
        return EXIT_OK
    if fetch:
        rc = fetch_and_convert(d, pins, out=out, opener=opener, variant=variant)
        return rc if rc != EXIT_OK else check(d, pins, judge, out, fetch=False, variant=variant)
    ck = spec.get("checkpoint") or {}
    vflag = "" if variant == DEFAULT_VARIANT else f" --variant {variant}"
    print(f"{PREFIX} WEIGHTS MISSING: no parameters file (*.bin.zst | *.bin) under {d}; nothing was downloaded. {pinned['file']} ({pinned['bytes']} bytes, "
          f"sha256 {pinned['sha256'][:16]}…) is the {spec['name']} checkpoint converted to this port's layout — {spec['source']}. Either re-run this step with --fetch "
          f"(`run.sh install --weights {d} --fetch{vflag}`: the public checkpoint{' (%d MB)' % (ck['bytes'] // 10**6) if ck.get('bytes') else ''} is downloaded and converted here on the CPU; "
          f"--variant {' | '.join(pins['variants'])} picks the checkpoint, default {DEFAULT_VARIANT}), "
          f"or put a converted {pinned['file']} {'(and ' + conv['file'] + ') ' if conv else ''}in {d} and re-run it (STOCK.md 'weights').", file=out, flush=True)
    return EXIT_MISSING


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def converter_env(repo: Optional[str]) -> dict:
    """The converter's environment: the JAX-process environment (stack.model_process_env(jax=True)) with the checkout's src/ FIRST on PYTHONPATH
    when it exists — the converter script imports alphafold3.model.of3_weight_converter, and this makes it the pinned commit's own module rather
    than whatever alphafold3-open wheel the JAX venv carries (a wheel built from the parent pin lacks the openbind layout). Pure Python (numpy,
    torch, zstandard): the checkout's compiled extension is never needed for the conversion."""
    env = stack.model_process_env(jax=True)
    src = os.path.join(repo, "src") if repo else None
    if src and os.path.isdir(os.path.join(src, "alphafold3")):
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def fetch_and_convert(d: str, pins: dict, out=sys.stdout, opener=urllib.request.urlopen, runner=subprocess.run, variant: str = DEFAULT_VARIANT) -> int:
    """--fetch: the variant's public checkpoint (PINS variants.<variant>.checkpoint) into <d>/checkpoint/ — kept when already there with the
    pinned digest, else downloaded and digest-checked — then the reference fork's converter (PINS variants.<variant>.converter) run on the JAX
    interpreter with the JAX-process environment (converter_env), writing the converted parameters into <d>. ONE line per stage (FETCH,
    CONVERT); EXIT_OK when the converter returned 0 and wrote the file (the caller judges its digest), EXIT_MISSING otherwise with the reason by
    name. Deletes nothing."""
    spec = pins["variants"][variant]; ck, cv, pinned = spec["checkpoint"], spec["converter"], spec["converted"]
    jax_py, repo = stack.jax_python(), stack.jax_repo()
    script = os.path.join(repo or "", cv["script"])
    if not (jax_py and os.path.isfile(jax_py) and os.access(jax_py, os.X_OK)):
        print(f"{PREFIX} WEIGHTS CONVERT FAILED: AF3_TORCH_JAX_PY ({jax_py or 'not set'}) is not an executable interpreter — the converter runs on the JAX environment (README.md 'Variables')", file=out, flush=True)
        return EXIT_MISSING
    if not (repo and os.path.isfile(script)):
        print(f"{PREFIX} WEIGHTS CONVERT FAILED: {cv['script']} not found under AF3_TORCH_JAX_REPO ({repo or 'not set'}) — the reference fork checkout of the install (README.md 'Variables')", file=out, flush=True)
        return EXIT_MISSING
    ckdir = os.path.join(d, "checkpoint"); path = os.path.join(ckdir, ck["file"])
    os.makedirs(ckdir, exist_ok=True)
    if os.path.isfile(path) and _sha256(path) == ck["sha256"]:
        print(f"{PREFIX} FETCH kept file={path} sha256={ck['sha256'][:16]}… (already present with the pinned digest)", file=out, flush=True)
    else:
        t0 = time.time(); part = path + ".part"
        try:
            with opener(ck["url"]) as r, open(part, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 22), b""):
                    f.write(chunk)
        except (OSError, urllib.error.URLError) as e:
            print(f"{PREFIX} WEIGHTS FETCH FAILED url={ck['url']} ({e.__class__.__name__}: {e}) — re-run with network access, or download that file into {ckdir} yourself and re-run", file=out, flush=True)
            return EXIT_MISSING
        got = _sha256(part)
        if got != ck["sha256"]:
            print(f"{PREFIX} WEIGHTS FETCH FAILED url={ck['url']} sha256={got[:16]}… is not the pinned {ck['sha256'][:16]}… ({os.path.getsize(part)} bytes; kept as {part} for inspection, not converted)", file=out, flush=True)
            return EXIT_MISSING
        os.replace(part, path)
        print(f"{PREFIX} FETCH downloaded url={ck['url']} file={path} bytes={os.path.getsize(path)} sha256={got[:16]}… (pinned) wall_s={time.time() - t0:.0f}", file=out, flush=True)
    cmd = [jax_py, script, "--of3_checkpoint", path, "--output_dir", d]
    t0 = time.time()
    env = converter_env(repo)
    print(f"{PREFIX} CONVERT running {' '.join(cmd)} (cwd {repo}; the fork's converter of variant {variant} ({spec['name']}), CPU"
          f"{'; PYTHONPATH=' + env['PYTHONPATH'].split(os.pathsep)[0] if 'PYTHONPATH' in env else ''})", file=out, flush=True)
    rc = runner(cmd, cwd=repo, env=env).returncode
    made = os.path.join(d, pinned["file"])
    if rc != 0 or not os.path.isfile(made):
        print(f"{PREFIX} WEIGHTS CONVERT FAILED rc={rc} — {cv['script']} did not leave {made} (its transcript is above)", file=out, flush=True)
        return EXIT_MISSING
    print(f"{PREFIX} CONVERT done file={made} bytes={os.path.getsize(made)} wall_s={time.time() - t0:.0f}", file=out, flush=True)
    return EXIT_OK


USAGE = "usage: python -m af3_torch_opt.weights DIR [--fetch] [--variant p2|ob]   (run.sh install --weights DIR [--fetch] [--variant p2|ob])"


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fetch = "--fetch" in argv; argv = [x for x in argv if x != "--fetch"]
    variant = DEFAULT_VARIANT
    rest = []
    k = 0
    while k < len(argv):
        a = argv[k]
        if a == "--variant" and k + 1 < len(argv) and not argv[k + 1].startswith("-"):
            variant = argv[k + 1]; k += 2; continue
        if a.startswith("--variant="):
            variant = a.split("=", 1)[1]; k += 1; continue
        rest.append(a); k += 1
    if len(rest) != 1 or rest[0].startswith("-") or not variant:
        print(USAGE, file=sys.stderr)
        return EXIT_USAGE
    return check(rest[0], fetch=fetch, variant=variant)


if __name__ == "__main__":
    sys.exit(main())
