"""Dump benchmark texts + their G2P phonemes into iOSDemo/iOSDemo/Resources/.

The iOS app uses precomputed phonemes from this JSON instead of running G2P
on-device. Run after editing TEXTS in benchmark.py:

    uv run python dump_benchmark_data.py

Also copies af_heart.bin and vocab.json so the app has everything it needs
in one folder. Models (mlpackages) are NOT copied — drop those into
iOSDemo/iOSDemo/Models/ manually.
"""
import json
import pathlib

import numpy as np
from huggingface_hub import hf_hub_download

from kokoro.pipeline import KPipeline

from benchmark import TEXTS, phonemize_for_benchmark


HERE = pathlib.Path(__file__).parent
RESOURCES = HERE / "iOSDemo" / "iOSDemo" / "Resources"
RESOURCES.mkdir(parents=True, exist_ok=True)

VOICE = "af_heart"
LANG = "a"


def main():
    # Phonemize texts via Kokoro G2P
    pipe = KPipeline(lang_code=LANG, model=False)
    cases = []
    for i, text in enumerate(TEXTS):
        phonemes = phonemize_for_benchmark(pipe, text)
        cases.append({
            "id": i,
            "text": text,
            "phonemes": phonemes,
            "n_phonemes": len(phonemes),
        })
        print(f"  case {i}: T_enc={len(phonemes):3d}  '{text[:60]}{'...' if len(text) > 60 else ''}'")

    # Voice pack [510, 256] flat float32 → bin
    voice = pipe.load_voice(VOICE).cpu().numpy().astype(np.float32)
    assert voice.shape == (510, 1, 256), f"unexpected voice shape {voice.shape}"
    voice = voice.reshape(510, 256)
    voice_path = RESOURCES / f"{VOICE}.bin"
    voice_path.write_bytes(voice.tobytes())
    print(f"  voice {VOICE}: {voice.shape} → {voice_path.name} ({voice_path.stat().st_size} bytes)")

    # Vocab → vocab.json (from HF config, no weights needed)
    config_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="config.json")
    config = json.loads(pathlib.Path(config_path).read_text())
    vocab = config["vocab"]
    vocab_path = RESOURCES / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))
    print(f"  vocab: {len(vocab)} entries → {vocab_path.name}")

    # Benchmark cases → benchmark_data.json
    data = {
        "voice": VOICE,
        "lang": LANG,
        "sample_rate": 24000,
        "cases": cases,
    }
    out = RESOURCES / "benchmark_data.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  cases: {len(cases)} → {out.name}")

    print(f"\nWrote to {RESOURCES.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
