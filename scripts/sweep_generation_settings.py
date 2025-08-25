#!/usr/bin/env python3
"""
Sweep Generation Settings für T5/FLAN-Checkpoints

Beschreibung
- Führt GPU-beschleunigte Text-Generierung mit mehreren Decoding-Settings aus.
- Unterstützt HF-Baseline (t5-base) und Lightning-Checkpoints (.ckpt).
- Handhabt Lightning-Präfixe (model.model./model.) und erkennt t5-base vs. google/flan-t5-base anhand des Pfads.
- Sucht nach einem lokalen Tokenizer nahe dem Checkpoint; Fallback auf HF-Tokenizer.
- Speichert Ergebnisse als JSON unter evaluation/results/generation_sweep/.

Schnellstart (Beispiele)
1) Baseline + 3 Checkpoints mit wissenschaftlichen Prompts (UTF-8) und kurzen Outputs
     export PROJECT_ROOT=$PWD
     python -u scripts/sweep_generation_settings.py \
         --include_baseline \
         --ckpt "clean_restart_logs/train/runs/2025-08-06_18-44-33/checkpoints/best/step-665000-val_ppl-1.39532.ckpt" \
                     "pretraining_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule/train/runs/2025-08-13_23-20-56/checkpoints/steps/step-step=640000.ckpt" \
                     "flan_t5_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule/train/runs/2025-08-08_16-11-27/checkpoints/best/step-237500-val_ppl-1.36184.ckpt" \
         --prompts_file scientific_prompts_en.txt \
         --max_new 24

2) Einzelnen Checkpoint testen, zusätzliche Prompts inline
     python -u scripts/sweep_generation_settings.py \
         --ckpt "/pfad/zum/checkpoint.ckpt" \
         --prompt "What is the central dogma of molecular biology?" \
         --prompt "Explain the difference between precision and recall." \
         --max_new 32

3) GPU/CPU steuern
     CUDA_VISIBLE_DEVICES=0 python -u scripts/sweep_generation_settings.py ...   # spezifische GPU
     python -u scripts/sweep_generation_settings.py --device cpu ...            # CPU erzwingen

Wichtige Parameter
- --include_baseline           Testet zusätzlich die HF-Referenz t5-base.
- --ckpt <paths...>            Eine oder mehrere .ckpt-Pfade (Lightning).
- --prompts_file <pfad>        Textdatei mit einem Prompt pro Zeile (UTF-8).
- --prompt "..."               Zusätzliche Prompts direkt auf der CLI (mehrfach möglich).
- --max_new <int>              Maximal neue Tokens pro Antwort (Default 24).
- --device <cuda|cpu>          Automatik (cuda, falls verfügbar) oder explizit setzen.
- --out_dir <pfad>             Zielordner für JSON-Ergebnisse (Default evaluation/results/generation_sweep).

Ausgabe
- JSON-Datei pro Modell/Run unter evaluation/results/generation_sweep/
    Struktur:
    {
        "model": "<kurzname>",
        "source": "<quelle>",
        "meta": { "base_id": "...", "tokenizer_source": "...", "missing_keys": 0, "unexpected_keys": 0,
                             "emb_vocab": 32128, "tok_vocab": 32000 },
        "results": { "<template>": { "<setting>": [ { "prompt": "...", "output": "...", "seconds": 0.12 }, ... ] } }
    }

Hinweise & Troubleshooting
- Prompts-Datei muss UTF-8 sein. Bei Decode-Fehlern (UnicodeDecodeError):
    iconv -f latin1 -t utf-8 scientific_prompts_en.txt -o scientific_prompts_en.utf8.txt
    und dann --prompts_file scientific_prompts_en.utf8.txt verwenden.
- Out-of-Memory: --max_new reduzieren (z. B. 16), Beams klein halten (num_beams=1).
- Für FLAN-Checkpoints ggf. FLAN-typische Formulierungen/Prompts nutzen (Templates "qa" oder "t5_answer").
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


def strip_prefixes(state_dict: dict) -> Tuple[dict, int, int]:
    clean = {}
    c_mm = c_m = 0
    for k, v in state_dict.items():
        if k.startswith('model.model.'):
            clean[k[len('model.model.'):]] = v
            c_mm += 1
        elif k.startswith('model.'):
            clean[k[len('model.'):]] = v
            c_m += 1
        else:
            clean[k] = v
    return clean, c_mm, c_m


def infer_vocab_size_from_weights(sd: dict) -> Optional[int]:
    if 'shared.weight' in sd:
        w = sd['shared.weight']
        if hasattr(w, 'shape') and len(w.shape) == 2:
            return w.shape[0]
    for k, v in sd.items():
        if k.endswith('shared.weight') and hasattr(v, 'shape') and len(v.shape) == 2:
            return v.shape[0]
    return None


def infer_base_model_id(path: Union[str, Path]) -> str:
    p = str(path).lower()
    if 'flan' in p:
        return 'google/flan-t5-base'
    return 't5-base'


def resolve_tokenizer_source(ckpt_path: Path, fallback: str) -> Union[str, Path]:
    candidates = []
    for cand in [ckpt_path.parent,
                 ckpt_path.parent.parent if ckpt_path.parent.parent else None,
                 ckpt_path.parent.parent.parent if ckpt_path.parent.parent and ckpt_path.parent.parent.parent else None]:
        if cand:
            candidates.append(cand)
            for sub in ['tokenizer', 'spm', 'sentencepiece', 'hf_tokenizer']:
                candidates.append(cand / sub)
    token_files = {'tokenizer.json', 'spiece.model', 'sentencepiece.bpe.model', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json'}
    for c in candidates:
        try:
            if c and c.exists() and c.is_dir():
                files = {x.name for x in c.iterdir() if x.is_file()}
                if files & token_files:
                    return c
        except Exception:
            continue
    return fallback


def load_model_from_ckpt(ckpt: Path, device: str):
    # Ensure project root on path for unpickler
    project_root = os.environ.get('PROJECT_ROOT')
    if not project_root:
        project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        ckpt_obj = torch.load(ckpt, map_location='cpu', weights_only=True)
    except Exception:
        ckpt_obj = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd_raw = ckpt_obj['state_dict']
    sd, c_mm, c_m = strip_prefixes(sd_raw)

    vocab_from_weights = infer_vocab_size_from_weights(sd)
    base_id = infer_base_model_id(ckpt)

    tok_src = resolve_tokenizer_source(ckpt, base_id)
    try:
        tokenizer = T5Tokenizer.from_pretrained(tok_src, local_files_only=isinstance(tok_src, Path))
        tok_src_label = str(tok_src)
    except Exception:
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        tok_src_label = 't5-base (fallback)'

    config = T5Config.from_pretrained(base_id)
    if vocab_from_weights is not None and config.vocab_size != vocab_from_weights:
        config.vocab_size = vocab_from_weights
    if getattr(config, 'eos_token_id', None) is None and tokenizer.eos_token_id is not None:
        config.eos_token_id = tokenizer.eos_token_id
    if getattr(config, 'pad_token_id', None) is None and tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id
    if getattr(config, 'decoder_start_token_id', None) is None and tokenizer.pad_token_id is not None:
        config.decoder_start_token_id = tokenizer.pad_token_id

    model = T5ForConditionalGeneration(config)
    res = model.load_state_dict(sd, strict=False)

    model.to(device).eval()
    return model, tokenizer, {
        'base_id': base_id,
        'tokenizer_source': tok_src_label,
        'missing_keys': len(getattr(res, 'missing_keys', []) or []),
        'unexpected_keys': len(getattr(res, 'unexpected_keys', []) or []),
        'emb_vocab': int(model.get_input_embeddings().weight.shape[0]),
        'tok_vocab': int(tokenizer.vocab_size or 0),
    }


def load_hf_model(name: str, device: str):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
    return mdl, tok, {
        'base_id': name,
        'tokenizer_source': name,
        'missing_keys': 0,
        'unexpected_keys': 0,
        'emb_vocab': int(mdl.get_input_embeddings().weight.shape[0]),
        'tok_vocab': int(tok.vocab_size or 0),
    }


@dataclass
class GenConfig:
    name: str
    params: Dict[str, Any]


def make_sweep() -> List[GenConfig]:
    cfgs: List[GenConfig] = []
    # Greedy variants
    for ngram in [2, 3, 4]:
        for rep in [1.0, 1.2, 1.5]:
            cfgs.append(GenConfig(name=f"greedy_ng{ngram}_rp{rep}", params={
                'num_beams': 1,
                'do_sample': False,
                'no_repeat_ngram_size': ngram,
                'repetition_penalty': rep,
            }))
    # Beam search variants
    for beams in [2, 4]:
        for lp in [1.0, 1.2]:
            for ngram in [2, 3]:
                cfgs.append(GenConfig(name=f"beam{beams}_lp{lp}_ng{ngram}", params={
                    'num_beams': beams,
                    'do_sample': False,
                    'length_penalty': lp,
                    'no_repeat_ngram_size': ngram,
                }))
    # Sampling variants
    for temp in [0.7, 1.0]:
        for rep in [1.0, 1.2]:
            cfgs.append(GenConfig(name=f"sample_t{temp}_rp{rep}", params={
                'num_beams': 1,
                'do_sample': True,
                'temperature': temp,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': rep,
            }))
    return cfgs


PROMPT_TEMPLATES = {
    'plain': "{p}",
    't5_answer': "answer: {p}",
    'qa': "Question: {p}\nAnswer:",
}

DEFAULT_PROMPTS = [
    "The capital of Germany is",
    "What is 2 + 2?",
    "Frankfurt is a city in",
    "Machine learning is the study of",
]


def run_sweep(model, tokenizer, device: str, prompts: List[str], out_max_new: int = 24):
    sweep = make_sweep()
    results: Dict[str, Any] = {}
    for tname, ttemplate in PROMPT_TEMPLATES.items():
        results[tname] = {}
        for cfg in sweep:
            cfg_result = []
            for p in prompts:
                text = ttemplate.format(p=p)
                ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
                gen_kwargs = dict(max_new_tokens=out_max_new)
                gen_kwargs.update(cfg.params)
                # Remove unsupported flags to avoid warnings
                if not gen_kwargs.get('do_sample', False):
                    for k in ['top_k', 'top_p', 'temperature']:
                        gen_kwargs.pop(k, None)
                start = time.time()
                with torch.no_grad():
                    out = model.generate(ids, **gen_kwargs)
                dur = time.time() - start
                out_text = tokenizer.decode(out[0], skip_special_tokens=True)
                cfg_result.append({
                    'prompt': text,
                    'output': out_text,
                    'seconds': dur,
                })
            results[tname][cfg.name] = cfg_result
    return results


def main():
    ap = argparse.ArgumentParser(description="Sweep decoding settings for T5 checkpoints and save outputs")
    ap.add_argument('--ckpt', nargs='*', help='Paths to .ckpt files')
    ap.add_argument('--include_baseline', action='store_true', help='Also test t5-base baseline')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_dir', default='evaluation/results/generation_sweep')
    ap.add_argument('--max_new', type=int, default=24)
    ap.add_argument('--prompts_file', type=str, default=None, help='Path to a text file with one prompt per line')
    ap.add_argument('--prompt', action='append', default=None, help='Add an extra prompt (can be passed multiple times)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    jobs: List[Tuple[str, str]] = []  # (label, type)
    if args.include_baseline:
        jobs.append(('t5-base', 'hf'))
    if args.ckpt:
        for c in args.ckpt:
            jobs.append((c, 'ckpt'))

    # Build prompts list
    prompts = list(DEFAULT_PROMPTS)
    if args.prompts_file:
        try:
            with open(args.prompts_file, 'r', encoding='utf-8') as f:
                file_prompts = [ln.strip() for ln in f.readlines() if ln.strip()]
                if file_prompts:
                    prompts = file_prompts
        except Exception as e:
            print(f"[WARN] Failed to read prompts file {args.prompts_file}: {e}")
    if args.prompt:
        for p in args.prompt:
            if p and p.strip():
                prompts.append(p.strip())

    for label, typ in jobs:
        print(f"\n=== Running sweep for {label} ({typ}) ===")
        if typ == 'hf':
            model, tok, meta = load_hf_model(label, args.device)
            model_label = 't5-base-baseline'
        else:
            model, tok, meta = load_model_from_ckpt(Path(label), args.device)
            # form a compact name
            p = Path(label)
            model_label = p.parent.parent.parent.name + '_' + p.parent.name if p.exists() else p.name

        print("meta:", meta)
        results = run_sweep(model, tok, args.device, prompts, out_max_new=args.max_new)

        ts = time.strftime('%Y%m%d_%H%M%S')
        out_json = Path(args.out_dir) / f"{model_label}_{ts}.json"
        with open(out_json, 'w') as f:
            json.dump({
                'model': model_label,
                'source': label,
                'meta': meta,
                'results': results,
            }, f, indent=2)
        print(f"Saved: {out_json}")


if __name__ == '__main__':
    main()
