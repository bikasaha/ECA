import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM

import json
import uuid
import time


# Auto-select device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[ECA] Using device: {DEVICE}")

# ---------------------------------------------------------
# Embedder wrapper
# ---------------------------------------------------------
class Embedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = DEVICE,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        return embs

# ---------------------------------------------------------
# Masked-LM local search
# ---------------------------------------------------------
class MlmLocalSearch:
    def __init__(
        self,
        mlm_name: str = "bert-base-uncased",
        embedder: Optional[Embedder] = None,
        topk: int = 15,
        steps: int = 25,
        beam: int = 5,
        max_positions_per_step: int = 6,
        device: str = DEVICE,
    ):
        """
        mlm_name: name of masked LM model (bert-base-uncased, roberta-base, etc.)
        topk:     top-k predictions per mask
        steps:    number of optimization steps
        beam:     beam width
        max_positions_per_step: max token positions to mask in each step
        """
        self.tokenizer = AutoTokenizer.from_pretrained(mlm_name,add_prefix_space=True)
        self.model = AutoModelForMaskedLM.from_pretrained(mlm_name).to(device)
        self.embed = embedder or Embedder()
        self.topk = topk
        self.steps = steps
        self.beam = beam
        self.max_positions_per_step = max_positions_per_step
        self.device = device

    @torch.no_grad()
    def _mlm_candidates(self, text: str, positions: List[int]) -> Dict[int, List[str]]:
        toks = self.tokenizer.tokenize(text)
        cand_map: Dict[int, List[str]] = {}
        for pos in positions:
            if pos < 0 or pos >= len(toks):
                continue
            masked = toks.copy()
            masked[pos] = self.tokenizer.mask_token
            ids = self.tokenizer.encode(
                masked,
                is_split_into_words=True,
                return_tensors="pt",
                
            ).to(self.device)

            logits = self.model(ids).logits[0]
            mask_idx = (ids[0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_idx) == 0:
                continue

            top_ids = torch.topk(logits[mask_idx[0]], k=self.topk).indices.tolist()
            words = [self.tokenizer.decode([i]).strip() for i in top_ids]
            words = [w for w in words if w.isalpha() and len(w) < 20]
            if words:
                cand_map[pos] = words[: self.topk]
        return cand_map


    def improve(
        self,
        target_query: str,
        seed_text: str,
        patience: int = 3,
        eps: float = 1e-4,
        log_fn=None,   
        ) -> Tuple[str, float]:
        qv_vec = self.embed.encode([target_query])[0]
        seed_vec = self.embed.encode([seed_text])[0]
        base_sim = float(seed_vec @ qv_vec)

        beams: List[Tuple[str, float]] = [(seed_text, base_sim)]

        best_txt, best_sim = seed_text, base_sim
        no_improve_steps = 0

        # optional: log initial state
        if log_fn is not None:
            log_fn(
                event="mlm_init",
                payload={
                    "seed_text": seed_text,
                    "seed_sim": best_sim,
                },
            )

        for step in range(self.steps):
            new_beams: List[Tuple[str, float]] = []

            for txt, _sim in beams:
                toks = self.tokenizer.tokenize(txt)
                if not toks:
                    continue
                positions = list(range(len(toks)))
                random.shuffle(positions)
                positions = positions[: min(self.max_positions_per_step, len(positions))]
                cand_map = self._mlm_candidates(txt, positions)

                for pos, words in cand_map.items():
                    for w in words:
                        ntoks = toks.copy()
                        ntoks[pos] = w
                        cand_txt = self.tokenizer.convert_tokens_to_string(ntoks)
                        cand_vec = self.embed.encode([cand_txt])[0]
                        sim = float(cand_vec @ qv_vec)
                        new_beams.append((cand_txt, sim))

            if not new_beams:
                break

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[: self.beam]

            # track global best
            cur_best_txt, cur_best_sim = beams[0]
            improved = False
            if cur_best_sim > best_sim + eps:
                best_sim = cur_best_sim
                best_txt = cur_best_txt
                no_improve_steps = 0
                improved = True
            else:
                no_improve_steps += 1

            # ---------- LOGGING PER ITERATION (STEP) ----------
            if log_fn is not None:
                log_fn(
                    event="mlm_step",
                    payload={
                        "step": step,
                        "num_candidates": len(new_beams),
                        "cur_best_text": cur_best_txt,
                        "cur_best_sim": cur_best_sim,
                        "global_best_text": best_txt,
                        "global_best_sim": best_sim,
                        "no_improve_steps": no_improve_steps,
                        "improved": improved,
                    },
                )
            # --------------------------------------------------

            # convergence: no improvement for "patience" steps
            if no_improve_steps >= patience:
                break

        # final log
        if log_fn is not None:
            log_fn(
                event="mlm_final",
                payload={
                    "best_text": best_txt,
                    "best_sim": best_sim,
                },
            )

        return best_txt, best_sim

# ---------------------------------------------------------
# High-level ECA wrapper
# ---------------------------------------------------------
def eca_attack(
    target_query: str,
    off_topic_seed: str,
    embedder: Optional[Embedder] = None,
    mlm_search: Optional[MlmLocalSearch] = None,
    verbose: bool = True,
    patience: int = 3,
    eps: float = 1e-4,
    log_path: Optional[str] = None,   
    run_id: Optional[str] = None,     
    ) -> Dict[str, object]:
    """
    Run ECA:
        - MLM-based local search attack.
        - Starts directly from off_topic_seed and uses masked-LM mutations
            to optimize similarity to target_query.
    Compute and report seed similarity as a baseline.
    """
    embedder = embedder or Embedder()
    mlm_search = mlm_search or MlmLocalSearch(embedder=embedder)

    # --------------- SETUP LOGGER ---------------
    if run_id is None:
        run_id = str(uuid.uuid4())

    def log_fn(event: str, payload: Dict):
        if log_path is None:
            return
        record = {
            "timestamp": time.time(),
            "run_id": run_id,
            "event": event,
            "target_query": target_query,
            "off_topic_seed": off_topic_seed,
            "payload": payload,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # --------------------------------------------

    # Similarity of raw off-topic seed
    qv_vec = embedder.encode([target_query])[0]
    seed_vec = embedder.encode([off_topic_seed])[0]
    seed_sim = float(seed_vec @ qv_vec)

    # log initial summary
    log_fn(
        event="init",
        payload={
            "seed_sim": seed_sim,
            "embedder_model": embedder.model_name,
        },
    )

    if verbose:
        print("\n[ECA] Target query:")
        print("     ", target_query)
        print("\n[ECA] Off-topic seed:")
        print("     ", off_topic_seed)
        print(f"[ECA] Seed similarity to target: {seed_sim:.4f}")

    # -----------------------------
    # Mode: MLM-based local search
    # -----------------------------
    mlm_txt, mlm_sim = mlm_search.improve(
        target_query=target_query,
        seed_text=off_topic_seed, 
        patience=patience,
        eps=eps,
        log_fn=log_fn,              
    )

    if verbose:
        print("\n[ECA] [MLM] MLM-refined collider (starting from seed):")
        print("     ", mlm_txt)
        print(f"     similarity: {mlm_sim:.4f}")

    return {
        "mode": "mlm",
        "target": target_query,
        "off_topic_seed": off_topic_seed,
        "seed_sim": seed_sim,
        "mlm_best_text": mlm_txt,
        "mlm_best_sim": mlm_sim,
        "embedder_model": embedder.model_name,
        "mlm_model": mlm_search.model.base_model_prefix
        if hasattr(mlm_search.model, "base_model_prefix")
        else type(mlm_search.model).__name__,
        "run_id": run_id,
        "log_path": log_path,
    }



if __name__ == "__main__":
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    mlm_search = MlmLocalSearch(mlm_name="bert-base-uncased", embedder=embedder)

    target = "The research focuses on developing robust techniques to detect advanced cyber attacks in large-scale computer networks."
    off_seed = "The orchestra performed a beautiful symphony in the old concert hall last night."

    log_path = "logs.jsonl"

    # MLM-based attack with logging
    result_mlm = eca_attack(
        target_query=target,
        off_topic_seed=off_seed,
        embedder=embedder,
        mlm_search=mlm_search,
        patience=5,
        eps=1e-3,
        log_path=log_path,  
    )
