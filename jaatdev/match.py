from typing import List, Tuple, Dict, Any, Optional, Generator
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import multiprocessing as mp
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, CrossEncoder

from .base import get_device_settings, get_sentence_splitter, get_optimal_attn_implementation
from .config import MODEL_CACHE, logger, GLOBAL_SETTINGS
from .utils import progress_bar
from .taxonomy import Taxonomy, load_default_taxonomy

def get_shared_model(model_name: str, device: str, token: Optional[str] = None) -> SentenceTransformer:
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    if model_name not in MODEL_CACHE:
        model = SentenceTransformer(model_name, device=device, token=token, model_kwargs={"dtype": torch_dtype})
        model.max_seq_length = 128
        model.tokenizer.model_max_length = 128
        MODEL_CACHE[model_name] = model
    return MODEL_CACHE[model_name]

def get_shared_cross_encoder(model_name: str, device: str, token: Optional[str] = None, attn_backend: str = "sdpa") -> CrossEncoder:
    key = f"ce_{model_name}"
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = CrossEncoder(model_name, device=device, token=token, model_kwargs={"attn_implementation": attn_backend, "dtype": torch_dtype})
    return MODEL_CACHE[key]

@torch.inference_mode()
def cross_encoder_predict(
    cross_encoder: CrossEncoder,
    pairs: List[Tuple[str, str]],
    batch_size: int = 1024,
    max_length: int = 72,
    device: str = "cuda"
) -> np.ndarray:
    """Direct batched execution of CE."""
    if not pairs:
        return np.array([], dtype=np.float32)

    hf_model = cross_encoder.model
    hf_tokenizer = cross_encoder.tokenizer

    lengths = [len(p[0]) + len(p[1]) for p in pairs]
    sort_order = np.argsort(lengths)

    sorted_q = [pairs[i][0] for i in sort_order]
    sorted_t = [pairs[i][1] for i in sort_order]

    all_scores = []
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for i in range(0, len(pairs), batch_size):
        b_q = sorted_q[i:i + batch_size]
        b_t = sorted_t[i:i + batch_size]

        encoded = hf_tokenizer(
            b_q,
            b_t,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            logits = hf_model(**encoded).logits
            if logits.shape[-1] == 2:
                probs = F.softmax(logits, dim=-1)[:, 1]
            elif logits.shape[-1] == 1:
                probs = torch.sigmoid(logits.squeeze(-1))
            else:
                probs = logits.squeeze(-1)

        all_scores.append(probs.cpu().to(torch.float32).numpy())

    concat_scores = np.concatenate(all_scores)
    unsort = np.empty_like(sort_order)
    unsort[sort_order] = np.arange(len(sort_order))
    return concat_scores[unsort]

def condition_query(q: str) -> str:
    words = q.strip().split()
    if len(words) <= 5 and not any(w.endswith(('.', '!', '?')) for w in words):
        return f"Experience with {q}"
    return q

@dataclass
class MatchResult:
    task_id: str
    task: str
    score: float
    query_text: Optional[str] = None
    verification_score: Optional[float] = None

class Match:
    def __init__(
        self,
        embedding_model: str = "ibm-granite/granite-embedding-english-r2",
        cross_encoder_model: str = "mixedbread-ai/mxbai-rerank-base-v2",
        classification_model: str = "sjmeis/skill-classifier-small-v3.5",
        nli_model: Optional[str] = "cross-encoder/nli-deberta-v3-xsmall", #"cross-encoder/nli-deberta-v3-base"
        hf_token: Optional[str] = None,
        default_taxonomies: Optional[List[str]] = ("tasks", "skills", "ai", "title"),
        query_prefix: str = "",
        document_prefix: str = "",
        initial_top_n: int = 25,
        threshold: float = 0.70,
        cross_temp: float = 1.0,
    ) -> None:
        logger.info("Initializing Match pipeline...")
        self.device, self.batch_size = get_device_settings()
        self.num_workers = 1 if GLOBAL_SETTINGS.get("single_threaded", False) else int(mp.cpu_count() / 2)
        self.attn_backend = get_optimal_attn_implementation()

        self.threshold = threshold
        self.initial_top_n = initial_top_n
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.cross_temp = cross_temp
        self.hf_token = hf_token

        # Bi-encoder and task embeddings
        logger.info("Setting up bi-encoder and taxonomies...")
        self.embedding_model_name = embedding_model
        self.embedding_model = get_shared_model(embedding_model, self.device, token=self.hf_token)

        self._taxonomies: Dict[str, Taxonomy] = {}
        if default_taxonomies:
            for tax_name in default_taxonomies:
                tax_instance = load_default_taxonomy(tax_name)
                self.register_taxonomy(tax_instance)

        # Candidate extraction pipeline
        logger.info("Setting up candidate extraction classifier...")

        torch_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(
                classification_model,
                token=self.hf_token,
                torch_dtype=torch_dtype,
                attn_implementation=self.attn_backend,
            ).to(self.device)
        self.clf_model.eval()
        self.clf_tokenizer = AutoTokenizer.from_pretrained(classification_model, use_fast=True, max_length=128, truncation=True, token=self.hf_token)

        # Cross-Encoder Reranker
        self.cross_encoder = None
        if cross_encoder_model:
            logger.info(f"Loading Cross-Encoder reranker: {cross_encoder_model}...")
            self.cross_encoder = get_shared_cross_encoder(cross_encoder_model, self.device, token=self.hf_token, attn_backend=self.attn_backend)

        # NLI Model for verification
        self.nli_tokenizer = None
        self.nli_model = None
        self.entail_idx = None
        self.neutral_idx = None
        if nli_model:
            logger.info(f"Setting up NLI verification engine: {nli_model}...")
            self._init_nli(nli_model)

        logger.info("Match initialized successfully.")

    def _init_nli(self, nli_model_name: str) -> None:
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name, use_fast=True)

        is_deberta = "deberta" in nli_model_name.lower()
        attn_backend = "eager" if is_deberta else self.attn_backend

        torch_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                nli_model_name,
                token=self.hf_token,
                dtype=torch_dtype,
                attn_implementation=attn_backend
            ).to(self.device)
        self.nli_model.eval()

        id2label = {v.lower(): k for k, v in self.nli_model.config.id2label.items()}
        self.entail_idx = id2label.get("entailment", 0)
        self.neutral_idx = id2label.get("neutral", 1)
        self.contra_idx = id2label.get("contradiction", 2 if self.entail_idx != 2 and self.neutral_idx != 2 else 0)

    def register_taxonomy(self, taxonomy: Taxonomy) -> None:
        """Indexes and stores a taxonomy."""
        logger.info(f"Registering taxonomy '{taxonomy.name}' ({len(taxonomy.df)} items)...")
        taxonomy.build_embeddings(
            model=self.embedding_model,
            model_name=self.embedding_model_name,
            device=self.device,
            show_progress=GLOBAL_SETTINGS.get("show_progress", False)
        )
        self._taxonomies[taxonomy.name] = taxonomy

    def get_candidates_batch(self, texts: List[str]) -> List[Tuple[int, str]]:
        splitter = get_sentence_splitter(device=self.device)
        docs_segmented = splitter.split_two_pass(texts)

        all_data: List[Tuple[int, str]] = []
        for doc_idx, sentences in enumerate(docs_segmented):
            for s in sentences:
                all_data.append((doc_idx, s))

        if not all_data:
            return []

        raw_sentences = [p[1] for p in all_data]
        unique_sents, inverse_idx = np.unique(raw_sentences, return_inverse=True)
        num_unique = len(unique_sents)

        predictions = []
        use_cuda = "cuda" in str(self.device)
        autocast_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    
        total_batches = (num_unique + self.batch_size - 1) // self.batch_size
        pbar = progress_bar(
            range(0, num_unique, self.batch_size),
            total=total_batches,
            desc="Classifying sentences",
        )

        with torch.inference_mode():
            for b_start in pbar:
                batch_texts = unique_sents[b_start:b_start + self.batch_size].tolist()
                inputs = self.clf_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)

                with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=autocast_dtype):
                    logits = self.clf_model(**inputs).logits
                    probs = F.softmax(logits, dim=-1)

                pos_probs = probs[:, 1].cpu().numpy()
                for p in pos_probs:
                    predictions.append({
                        "label": "LABEL_1" if p >= 0.50 else "LABEL_0",
                        "score": float(p)
                    })

        filtered: List[Tuple[int, str]] = []
        for i, (doc_idx, sent) in enumerate(all_data):
            pred = predictions[inverse_idx[i]]
            if pred["label"] in ("LABEL_1", "POSITIVE", "1") and pred["score"] >= 0.80:
                filtered.append((doc_idx, sent))

        return filtered

    def _score_nli_entailment(
        self,
        pairs: List[Tuple[str, str]],
        tax: Taxonomy,
        batch_size: int = 1024,
        min_nli_margin: float = 0.05,
        min_entail_prob: float = 0.50,
        max_contradiction_prob: float = 0.15
    ) -> List[Dict[str, Any]]:
        """Runs NLI verification: premise (query/sentence), hypothesis (templated task)."""
        if not pairs or not self.nli_model:
            return []

        unique_pairs, inverse_indices = [], []
        pair_to_idx = {}
        for p in pairs:
            premise = p["query_sent"]
            hypothesis = tax.nli_hypotheses[p["tax_idx"]]
            pair = (premise, hypothesis)

            if pair not in pair_to_idx:
                pair_to_idx[pair] = len(unique_pairs)
                unique_pairs.append(pair)
            inverse_indices.append(pair_to_idx[pair])

        premises = [p for p, _ in unique_pairs]
        hypotheses = [h for _, h in unique_pairs]
        num_unique = len(unique_pairs)

        lengths = [len(p) + len(h) for p, h in zip(premises, hypotheses)]
        sort_order = np.argsort(lengths)
        sorted_premises = [premises[i] for i in sort_order]
        sorted_hypotheses = [hypotheses[i] for i in sort_order]

        unique_results = [None] * num_unique
        use_cuda = "cuda" in str(self.device)
        autocast_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

        pbar = progress_bar(
            range(0, num_unique, batch_size),
            desc="NLI Verification",
            total=(num_unique + batch_size - 1) // batch_size
        )

        for i in pbar:
            b_prem = sorted_premises[i:i + batch_size]
            b_hypo = sorted_hypotheses[i:i + batch_size]

            encoded = self.nli_tokenizer(
                b_prem,
                b_hypo,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.inference_mode():
                with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=autocast_dtype, enabled=use_cuda):
                    logits = self.nli_model(**encoded).logits
                    probs = F.softmax(logits, dim=-1)
                
                p_entail = probs[:, self.entail_idx].cpu().to(torch.float32).numpy()
                p_neutral = probs[:, self.neutral_idx].cpu().to(torch.float32).numpy()
                p_contra = probs[:, self.contra_idx].cpu().to(torch.float32).numpy()

            for idx_in_batch, (pe, pn, pc) in enumerate(zip(p_entail, p_neutral, p_contra)):
                orig_sorted_idx = i + idx_in_batch
                orig_unique_idx = sort_order[orig_sorted_idx]
                not_contradiction = (pc <= max_contradiction_prob)

                has_entailment_signal = (
                    (pe >= min_entail_prob and pe > pc * 3.0) or
                    (pe >= min_entail_prob - 0.10 and (pe - pn) >= -min_nli_margin)
                )

                is_verified = bool(not_contradiction and has_entailment_signal)

                unique_results[orig_unique_idx] = {
                    "entailment_prob": round(float(pe), 4),
                    "neutral_prob": round(float(pn), 4),
                    "contradiction_prob": round(float(pc), 4),
                    "margin": round(float(pe - pn), 4),
                    "verified": is_verified
                }

        return [unique_results[idx] for idx in inverse_indices]

    @staticmethod
    def _find_adaptive_elbow(
        scores: List[float],
        min_score_floor: float = 0.35,
        min_drop_ratio: float = 0.08,
        high_confidence_retention: float = 0.70
    ) -> int:
        """
        Detects the separation boundary between valid multi-matches and noise.
        - Retains all candidates scoring above `high_confidence_retention`.
        - Prunes candidates below `min_score_floor`.
        - Locates the largest drop weighted by absolute score level.
        """
        if not scores:
            return 0
        if scores[0] < min_score_floor:
            return 0
        if len(scores) == 1:
            return 1 if scores[0] >= min_score_floor else 0

        scores_arr = np.array(scores, dtype=np.float32)
        valid_indices = np.where(scores_arr >= min_score_floor)[0]
        if len(valid_indices) == 0:
            return 0

        num_valid = len(valid_indices)
        if num_valid == 1:
            return 1
        
        active_scores = scores_arr[valid_indices]

        drops = -np.diff(active_scores)
        weighted_drops = drops * (active_scores[:-1] ** 1.5)
        best_drop_idx = int(np.argmax(weighted_drops))

        if drops[best_drop_idx] >= min_drop_ratio:
            cutoff = best_drop_idx + 1
        else:
            cutoff = num_valid

        high_conf_count = int(np.sum(active_scores >= high_confidence_retention))
        final_cutoff = max(cutoff, high_conf_count)

        return min(final_cutoff, len(scores))

    def match_batch(
        self,
        texts: List[str],
        taxonomy_name: str,
        top_k: Optional[int] = None,
        verify_nli: bool = False,
        return_original: bool = False
    ) -> List[List[Dict[str, Any]]]:
        if taxonomy_name not in self._taxonomies:
            raise KeyError(f"Taxonomy '{taxonomy_name}' is not registered. Registered: {list(self._taxonomies.keys())}")

        tax = self._taxonomies[taxonomy_name]

        logger.info("Extracting candidates...")
        all_data = self.get_candidates_batch(texts)
        results: List[List[Dict[str, Any]]] = [[] for _ in range(len(texts))]
        if not all_data:
            return results

        raw_query_texts = [d[1] for d in all_data]
        unique_queries, inverse_map = np.unique(raw_query_texts, return_inverse=True)
        U = len(unique_queries)

        use_cuda = "cuda" in str(self.device)
        autocast_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

        # Embedding coarse search
        logger.info("Running coarse retrieval...")
        prefixed_unique = [f"{self.query_prefix}{q}" for q in unique_queries]

        with torch.inference_mode():
            q_embed = self.embedding_model.encode(
                prefixed_unique,
                convert_to_tensor=True,
                batch_size=512,
                show_progress_bar=GLOBAL_SETTINGS.get("show_progress", False)
            ).to(self.device, dtype=autocast_dtype)
            q_embed = F.normalize(q_embed, p=2, dim=1)

            k_pool = min(self.initial_top_n * 2, tax.embeddings.shape[0])
            with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=autocast_dtype, enabled=use_cuda):
                sim_matrix = torch.mm(q_embed, tax.embeddings_T)
                dense_top_scores, dense_top_indices_gpu = torch.topk(sim_matrix, k=k_pool, dim=1)
            del q_embed, sim_matrix

            confident_mask = (dense_top_scores[:, 0] >= 0.85)
            uncertain_indices = torch.where(~confident_mask)[0].cpu().tolist()
            fused_cand_gpu = dense_top_indices_gpu[:, :self.initial_top_n].clone()

            if uncertain_indices:
                uncertain_texts = [unique_queries[i] for i in uncertain_indices]
                bm25_top_indices_gpu = tax.bm25.query_topk(uncertain_texts, k=k_pool)

                sub_dense = dense_top_indices_gpu[uncertain_indices]
                sub_M = len(uncertain_indices)

                combined_cands = torch.cat([sub_dense, bm25_top_indices_gpu], dim=1)
                
                ranks = torch.arange(1, k_pool + 1, device=self.device, dtype=torch.float32)
                rrf_weights = 1.0 / (60.0 + ranks)
                combined_weights = torch.cat([rrf_weights, rrf_weights]).unsqueeze(0).expand(sub_M, -1)

                fused_scores = torch.zeros((sub_M, len(tax.codes_arr)), device=self.device, dtype=torch.float32)
                fused_scores.scatter_add_(1, combined_cands, combined_weights)
                _, sub_fused = torch.topk(fused_scores, k=self.initial_top_n, dim=1)

                fused_cand_gpu[uncertain_indices] = sub_fused
                del fused_scores, combined_cands, combined_weights
                fused_cand_gpu[uncertain_indices] = sub_fused
                del fused_scores, bm25_top_indices_gpu

            dense_top_scores_cpu = dense_top_scores[:, :self.initial_top_n].cpu().numpy()
            del dense_top_scores, dense_top_indices_gpu

        fused_cand_unique = fused_cand_gpu.cpu().numpy()
        del fused_cand_gpu

        # prune low-confidence dense matches to save CE runtime, keep MIN (10)
        K = self.initial_top_n
        MIN_KEEP_TOP_N = min(10, K)
        DENSE_SCORE_FLOOR = 0.40

        ce_pairs: List[Tuple[str, str]] = []
        surviving_coords: List[Tuple[int, int]] = []

        for u_i in range(U):
            q_text = unique_queries[u_i]
            conditioned_q = condition_query(q_text)
            
            for rank_pos in range(K):
                c_idx = fused_cand_unique[u_i, rank_pos]
                bi_score = float(dense_top_scores_cpu[u_i, rank_pos])

                if rank_pos < MIN_KEEP_TOP_N or bi_score >= DENSE_SCORE_FLOOR:
                    ce_pairs.append((conditioned_q, tax.ce_targets[c_idx]))
                    surviving_coords.append((u_i, rank_pos))

        # Cross-Encoder Rerank
        logger.info("Re-ranking...")
        raw_ce_preds = cross_encoder_predict(
                self.cross_encoder,
                ce_pairs,
                batch_size=512,
                show_progress_bar=GLOBAL_SETTINGS.get("show_progress", True)
            )
        raw_arr = np.asarray(raw_ce_preds, dtype=np.float32)
        if np.all(raw_arr >= 0.0) and np.all(raw_arr <= 1.0):
            if abs(self.cross_temp - 1.0) > 1e-3:
                eps = 1e-7
                clipped = np.clip(raw_arr, eps, 1.0 - eps)
                logits = np.log(clipped / (1.0 - clipped))
                scaled_scores = 1.0 / (1.0 + np.exp(-logits / self.cross_temp))
            else:
                scaled_scores = raw_arr
        else:
            scaled_scores = 1.0 / (1.0 + np.exp(-raw_arr / max(self.cross_temp, 0.1)))

        unique_query_cand_scores = np.zeros((U, K), dtype=np.float32)
        for (u_i, rank_pos), score in zip(surviving_coords, scaled_scores):
            unique_query_cand_scores[u_i, rank_pos] = score

        # Apply Cutoff
        M = len(all_data)
        doc_indices = [d[0] for d in all_data]

        selected_records = []
        for i in range(M):
            u_idx = inverse_map[i]
            scores_row = unique_query_cand_scores[u_idx]
            cands_row = fused_cand_unique[u_idx]

            sort_order = np.argsort(-scores_row)
            sorted_scores = scores_row[sort_order]
            sorted_cands = cands_row[sort_order]

            cutoff = min(top_k, K) if top_k is not None else self._find_adaptive_elbow(sorted_scores.tolist())

            for rank_idx in range(cutoff):
                sc = sorted_scores[rank_idx]
                if sc < self.threshold:
                    break
                
                c_idx = sorted_cands[rank_idx]
                selected_records.append({
                    "orig_doc_idx": doc_indices[i],
                    "query_sent": raw_query_texts[i],
                    "tax_idx": c_idx,
                    "code": tax.codes_arr[c_idx],
                    "text": tax.texts_arr[c_idx],
                    "score": float(sc)
                })

        # Optional NLI Verification
        if verify_nli and self.nli_model and selected_records:
            nli_pairs = [(r["text"], r["query_sent"]) for r in selected_records]
            nli_scores = self._score_nli_entailment(
                nli_pairs,
                tax=tax,
                batch_size=1024,
                min_entail_prob=0.40,
                max_contradiction_prob=0.20
            )
            for r, n_data in zip(selected_records, nli_scores):
                r["nli_data"] = n_data
        else:
            for r in selected_records:
                r["nli_data"] = None

        # Filter by minimum confidence and structure output
        logger.info("Aggregating results...")
        aggregated_by_doc: List[Dict[str, Dict[str, Any]]] = [{} for _ in range(len(texts))]
        for r in selected_records:
            if r["score"] < self.threshold:
                continue

            if verify_nli and r["nli_data"] is not None:
                n_data = r["nli_data"]               
                if not n_data["verified"]:
                    continue

            d_idx = r["orig_doc_idx"]
            c_id = r["code"]
            curr_score = round(r["score"], 4)
            v_score = (
                round(r["nli_data"]["entailment_prob"], 4)
                if (verify_nli and r["nli_data"] is not None)
                else None
            )

            evidence_item = {
                "sentence": r["query_sent"],
                "score": curr_score,
            }
            if return_original:
                evidence_item["sentence"] = r["query_sent"]
            if verify_nli:
                evidence_item["verification_score"] = v_score

            doc_store = aggregated_by_doc[d_idx]

            if c_id not in doc_store:
                doc_store[c_id] = {
                    "code": c_id,
                    "matched_text": r["text"],
                    "taxonomy": tax.name,
                    "score": curr_score,
                    "frequency": 1,
                    "evidence": [evidence_item],
                }
                if return_original:
                    doc_store[c_id]["query_sentence"] = r["query_sent"]
                if verify_nli:
                    doc_store[c_id]["verification_score"] = v_score
            else:
                doc_store[c_id]["frequency"] += 1
                doc_store[c_id]["evidence"].append(evidence_item)

                if curr_score > doc_store[c_id]["score"]:
                    doc_store[c_id]["score"] = curr_score
                    if return_original:
                        doc_store[c_id]["query_sentence"] = r["query_sent"]
                    if verify_nli:
                        doc_store[c_id]["verification_score"] = v_score

        # Sort descending within each match
        final_results: List[List[Dict[str, Any]]] = []
        for doc_store in aggregated_by_doc:
            doc_items = list(doc_store.values())

            for item in doc_items:
                item["evidence"].sort(key=lambda x: x["score"], reverse=True)

            doc_items.sort(key=lambda x: x["score"], reverse=True)
            final_results.append(doc_items)

        return final_results

    def match_dataset(
        self,
        texts: List[str],
        taxonomy_name: str,
        doc_chunk_size: int = 1000,
        top_k: Optional[int] = None,
        verify_nli: bool = False,
        return_original: bool = False
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Streaming runner for many documents.
        Processes documents in chunks, yielding results doc by doc.
        """
        total_docs = len(texts)
        logger.info(f"Processing {total_docs} documents in batches of {doc_chunk_size}...")

        for start_idx in range(0, total_docs, doc_chunk_size):
            end_idx = min(start_idx + doc_chunk_size, total_docs)
            chunk_texts = texts[start_idx:end_idx]

            logger.info(f"Processing chunk {start_idx} to {end_idx}...")
            chunk_results = self.match_batch(
                texts=chunk_texts,
                taxonomy_name=taxonomy_name,
                top_k=top_k,
                verify_nli=verify_nli,
                return_original=return_original
            )

            for doc_result in chunk_results:
                yield doc_result

            del chunk_results

    def match(
        self,
        text: str,
        taxonomy_name: str,
        top_k: Optional[int] = None,
        verify_nli: bool = False,
        return_original: bool = False
    ) -> List[Dict[str, Any]]:
        return self.match_batch(
            [text],
            taxonomy_name=taxonomy_name,
            top_k=top_k,
            verify_nli=verify_nli,
            return_original=return_original
        )[0]