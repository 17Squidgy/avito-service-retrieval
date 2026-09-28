"""Offline HyDE pilot on a fixed random development subset, never hidden labels.

Generate hypothetical service ads with Qwen, encode them with E5's passage
prefix, and measure incremental candidate recall. This is a retrieval pilot,
not evidence of a final-top-50 improvement until a selector is evaluated.
"""
import gc
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoModelForCausalLM, AutoTokenizer

from avito_retrieval.data import identifier, load_data, normalize, validation_corpus
from avito_retrieval.dense import DenseEncoder
from avito_retrieval.retrieval import topk

ROOT = Path(__file__).resolve().parents[1]
PROMPT = ("Напиши короткое описание объявления исполнителя, точно соответствующего поисковому запросу. "
          "Пиши по-русски, не более 35 слов. Назови услугу и подходящие работы. "
          "Не придумывай город, цены, рейтинг, опыт или гарантии. "
          "Верни только описание, без вступления и пояснений.")
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def main():
    torch.set_num_threads(6)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("Run the HyDE pilot with GPU access")
    payload = joblib.load(ROOT / "artifacts/dense/validation_candidates.joblib")
    indices = np.random.default_rng(913).choice(len(payload["splits"]["dev"]), 300, replace=False)
    queries = payload["splits"]["dev"].iloc[indices].reset_index(drop=True)
    cache_path = ROOT / "artifacts/hyde_generations.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    texts = [f"Запрос: {q.search_query}\nФильтры: {q.search_infm_params_text}" for q in queries.itertuples()]
    keys = [hashlib.sha256((REVISION + PROMPT + ":greedy72:" + text).encode()).hexdigest() for text in texts]
    missing = [i for i, key in enumerate(keys) if key not in cached]
    if missing:
        path = ROOT / "artifacts/models/qwen2.5-1.5b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float16).to(device).eval()
        for start in range(0, len(missing), 16):
            rows = missing[start:start+16]
            chats = [tokenizer.apply_chat_template([
                {"role": "system", "content": PROMPT}, {"role": "user", "content": texts[i]}],
                tokenize=False, add_generation_prompt=True) for i in rows]
            inputs = tokenizer(chats, padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=72, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
            generations = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            for i, generation in zip(rows, generations):
                cached[keys[i]] = {"query": texts[i], "hypothesis": generation.strip()}
            temporary = cache_path.with_suffix(".pending.json")
            temporary.write_text(json.dumps(cached, ensure_ascii=False, indent=2))
            temporary.replace(cache_path)
            print(f"Generated {min(start+16, len(missing))}/{len(missing)}", flush=True)
        del model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    hypotheses = [cached[key]["hypothesis"] for key in keys]
    train, _, items = load_data(ROOT / "data/raw")
    corpus = validation_corpus(train, items)
    assert np.array_equal(corpus.item_id.to_numpy(), payload["ids"])
    encoder = DenseEncoder(ROOT / "artifacts/models/multilingual-e5-small", ROOT / "artifacts/models/e5_cache")
    embeddings = encoder.items(corpus)
    generated = encoder.encode(["passage: " + text for text in hypotheses])
    original = encoder.queries(queries)
    blended = generated + original
    blended /= np.linalg.norm(blended, axis=1, keepdims=True)
    locations = corpus.item_location_id.map(identifier).to_numpy()
    base = [payload["results"]["dev"][i]["candidates"] for i in indices]
    wider_path = ROOT / "artifacts/pool300/validation_candidates.joblib"
    wider = joblib.load(wider_path) if wider_path.exists() else None
    wider_base = [wider["results"]["dev"][i]["candidates"] for i in indices] if wider else None
    report = {"queries": len(queries), "seed": 913, "model_revision": REVISION,
              "selection_note": "Random development subset; no reserved-test scoring.", "variants": {}}
    with threadpool_limits(limits=8):
        for name, vectors in (("hypothesis", generated), ("blend", blended)):
            recalls, baseline, increments, standalone, wider_increments = [], [], [], [], []
            for start in range(0, len(queries), 32):
                scores = vectors[start:start+32] @ embeddings.T
                for j, score in enumerate(scores):
                    i = start + j
                    same = (locations == identifier(queries.iloc[i].search_location_id))
                    extra = np.unique(np.concatenate([topk(score, 100), topk(score + .04*same, 100),
                                                       topk(np.where(same, score, -np.inf), min(100, same.sum()))]))
                    relevant = set(queries.iloc[i].relevant)
                    hits = set(payload["ids"][base[i]]) & relevant
                    additional = set(payload["ids"][extra]) & relevant
                    baseline.append(len(hits) / len(relevant))
                    recalls.append(len(hits | additional) / len(relevant))
                    increments.append(len(additional - hits) / len(relevant))
                    standalone.append(len(set(payload["ids"][topk(score+.04*same,50)]) & relevant)/len(relevant))
                    if wider_base is not None:
                        wider_hits = set(payload["ids"][wider_base[i]]) & relevant
                        wider_increments.append(len(additional - wider_hits) / len(relevant))
            report["variants"][name] = {"baseline_pool_recall": float(np.mean(baseline)),
                "expanded_pool_recall": float(np.mean(recalls)), "incremental_pool_recall": float(np.mean(increments)),
                "standalone_context_recall_at_50": float(np.mean(standalone)),
                "increment_over_pool300": float(np.mean(wider_increments)) if wider_increments else None}
    output = ROOT / "reports/round2/hyde_pilot.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
