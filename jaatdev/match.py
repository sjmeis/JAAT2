from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import multiprocessing as mp
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, CrossEncoder

from .base import ListDataset, get_device_settings, get_sentence_splitter
from .config import MODEL_CACHE, logger, GLOBAL_SETTINGS
from .utils import progress_bar
from .taxonomy import Taxonomy, load_default_taxonomy, bm25_tokenize

def get_shared_model(model_name: str, device: str, token: Optional[str] = None) -> SentenceTransformer:
    if model_name not in MODEL_CACHE:
        model = SentenceTransformer(model_name, device=device, token=token)
        model.max_seq_length = 128
        model.tokenizer.model_max_length = 128
        MODEL_CACHE[model_name] = model
    return MODEL_CACHE[model_name]

def get_shared_cross_encoder(model_name: str, device: str, token: Optional[str] = None) -> CrossEncoder:
    key = f"ce_{model_name}"
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = CrossEncoder(model_name, device=device, token=token)
    return MODEL_CACHE[key]

def reciprocal_rank_fusion(
    dense_ranked_indices: List[int],
    bm25_ranked_indices: List[int],
    top_n: int,
    rrf_k: int = 60
) -> List[int]:
    """Combines dense and lexical rankings using standard RRF."""
    rrf_scores: Dict[int, float] = {}

    for rank, idx in enumerate(dense_ranked_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

    for rank, idx in enumerate(bm25_ranked_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

    sorted_candidates = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return sorted_candidates[:top_n]

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
        embedding_model: str = "sjmeis/skills-embedding-gemma",
        cross_encoder_model: str = "sjmeis/task-cross-encoder-v1",
        classification_model: str = "sjmeis/skill-classifier-base-v3.5",
        nli_model: Optional[str] = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", #"cross-encoder/nli-deberta-v3-base"
        hf_token: Optional[str] = None,
        default_taxonomies: Optional[List[str]] = ("tasks", "skills", "ai"),
        query_prefix: str = "",
        document_prefix: str = "",
        initial_top_n: int = 25,
        threshold: float = 0.60,
        cross_temp: float = 1.0,
    ) -> None:
        logger.info("Initializing Match pipeline...")
        self.device, self.batch_size = get_device_settings()
        self.num_workers = 1 if GLOBAL_SETTINGS.get("single_threaded", False) else int(mp.cpu_count() / 2)

        self.threshold = threshold
        self.initial_top_n = initial_top_n
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.cross_temp = cross_temp
        self.hf_token = hf_token

        # Bi-encoder and task embeddings
        logger.info("Setting up bi-encoder and task corpus...")
        self.embedding_model_name = embedding_model
        self.embedding_model = get_shared_model(embedding_model, self.device, token=self.hf_token)

        self._taxonomies: Dict[str, Taxonomy] = {}
        if default_taxonomies:
            for tax_name in default_taxonomies:
                tax_instance = load_default_taxonomy(tax_name)
                self.register_taxonomy(tax_instance)

        # Candidate extraction pipeline
        logger.info("Setting up candidate extraction classifier...")
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(classification_model, token=self.hf_token)
        self.clf_tokenizer = AutoTokenizer.from_pretrained(classification_model, use_fast=True, max_length=64, truncation=True, token=self.hf_token)
        self.pipe = pipeline(
            "text-classification",
            model=self.clf_model,
            tokenizer=self.clf_tokenizer,
            max_length=64,
            device=self.device,
            truncation=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        # Cross-Encoder Reranker
        self.cross_encoder = None
        if cross_encoder_model:
            logger.info(f"Loading Cross-Encoder reranker: {cross_encoder_model}...")
            self.cross_encoder = get_shared_cross_encoder(cross_encoder_model, self.device, token=self.hf_token)

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
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name, token=self.hf_token).to(self.device)
        self.nli_model.eval()

        id2label = {v.lower(): k for k, v in self.nli_model.config.id2label.items()}
        self.entail_idx = id2label.get("entailment", 0)
        self.neutral_idx = id2label.get("neutral", 1)

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

        dataset = ListDataset([x[1] for x in all_data])
        predictions = [r for r in progress_bar(self.pipe(dataset))]

        return [
            pair for pair, pred in zip(all_data, predictions) 
            if pred["label"] in ("LABEL_1", "POSITIVE", "1")
        ]

    def _score_cross_encoder(self, pairs: List[Tuple[str, str]], template: str) -> List[float]:
        """Expects pairs of (task_desc, context_sent)."""
        if not pairs or not self.cross_encoder:
            return []
        
        conditioned_pairs = [
            (ctx, template.format(text=target.lower()))
            for target, ctx in pairs
        ]
        preds = self.cross_encoder.predict(conditioned_pairs)
        # Scaled sigmoid calibration
        return [float(1.0 / (1.0 + np.exp(-(float(x) + 1.5) / self.cross_temp))) for x in preds]

    def _score_nli_entailment(
        self,
        pairs: List[Tuple[str, str]],
        template: str,
        batch_size: int = 128,
        min_nli_margin: float = 0.10
    ) -> List[float]:
        """Runs NLI verification: premise (query/sentence), hypothesis (templated task)."""
        if not pairs or not self.nli_model:
            return []

        premises = [ctx for _, ctx in pairs]
        hypotheses = [template.format(text=target.lower().strip()) for target, _ in pairs]

        all_scores: List[Dict[str, Any]] = []
        use_cuda = "cuda" in str(self.device)
        autocast_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

        for i in range(0, len(pairs), batch_size):
            b_prem = premises[i:i + batch_size]
            b_hypo = hypotheses[i:i + batch_size]

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
                
                p_entail = probs[:, self.entail_idx].cpu().numpy()
                p_neutral = probs[:, self.neutral_idx].cpu().numpy()

            for pe, pn in zip(p_entail, p_neutral):
                margin = float(pe - pn)
                is_verified = (pe > pn) and (margin >= min_nli_margin)
                all_scores.append({
                    "entailment_prob": round(float(pe), 4),
                    "neutral_prob": round(float(pn), 4),
                    "margin": round(margin, 4),
                    "verified": bool(is_verified)
                })

        return all_scores

    @staticmethod
    def _find_adaptive_elbow(
        scores: List[float],
        min_score_floor: float = 0.35,
        min_drop_ratio: float = 0.08,
        high_confidence_retention: float = 0.90
    ) -> int:
        """
        Detects the separation boundary between valid multi-matches and noise.
        - Retains all candidates scoring above `high_confidence_retention`.
        - Prunes candidates below `min_score_floor`.
        - Locates the largest drop weighted by absolute score level.
        """
        if not scores:
            return 0
        if len(scores) == 1:
            return 1 if scores[0] >= min_score_floor else 0

        scores_arr = np.array(scores, dtype=np.float32)
        valid_indices = np.where(scores_arr >= min_score_floor)[0]
        if len(valid_indices) == 0:
            return 0
        
        active_scores = scores_arr[valid_indices]
        if len(active_scores) == 1:
            return 1

        drops = -np.diff(active_scores)
        weighted_drops = drops * (active_scores[:-1] ** 1.5)
        best_drop_idx = int(np.argmax(weighted_drops))

        if drops[best_drop_idx] >= min_drop_ratio:
            cutoff = best_drop_idx + 1
        else:
            cutoff = len(active_scores)

        high_conf_count = int(np.sum(scores_arr >= high_confidence_retention))
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
        
        all_data = self.get_candidates_batch(texts)
        results: List[List[Dict[str, Any]]] = [[] for _ in range(len(texts))]
        if not all_data:
            return results

        # Embedding coarse search
        queries = [f"{self.query_prefix}{d[1]}" for d in all_data]
        use_cuda = "cuda" in str(self.device)
        autocast_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

        with torch.inference_mode():
            q_embed = self.embedding_model.encode(
                queries,
                convert_to_tensor=True,
                batch_size=128,
                show_progress_bar=GLOBAL_SETTINGS.get("show_progress", False)
            ).to(self.device)
            q_embed = F.normalize(q_embed, p=2, dim=1)

            with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=autocast_dtype, enabled=use_cuda):
                sim_matrix = torch.mm(q_embed, tax.embeddings.T)

            k_retrieval = min(self.initial_top_n, tax.embeddings.shape[0])
            top_scores, top_indices = torch.topk(sim_matrix, k=k_retrieval, dim=1)

        top_indices_cpu = top_indices.cpu().numpy()
        top_scores_cpu = top_scores.cpu().to(torch.float32).numpy()

        # Per-candidate reranking, flattening for global calculation
        flat_candidate_records = []
        slice_offsets = [0]
        for i, (orig_doc_idx, query_sent) in enumerate(all_data):
            dense_cand_ids = top_indices_cpu[i].tolist()
            dense_score_lookup = dict(zip(dense_cand_ids, top_indices_cpu[i].tolist()))

            q_tokens = bm25_tokenize(query_sent)
            bm25_scores = tax.bm25.get_scores(q_tokens)
            bm25_cand_ids = np.argsort(bm25_scores)[::-1][:k_retrieval].tolist()

            fused_cand_ids = reciprocal_rank_fusion(
                dense_ranked_indices=dense_cand_ids,
                bm25_ranked_indices=bm25_cand_ids,
                top_n=self.initial_top_n
            )

            rows = tax.df.iloc[fused_cand_ids]
            for c_idx, (_, row) in zip(fused_cand_ids, rows.iterrows()):
                flat_candidate_records.append({
                    "orig_doc_idx": orig_doc_idx,
                    "query_sent": query_sent,
                    "code": row["id"],
                    "text": row["text"],
                    "bi_score": float(dense_score_lookup.get(c_idx, 0.0))
                })
            slice_offsets.append(len(flat_candidate_records))

        # Cross-Encoder Rerank
        if self.cross_encoder and flat_candidate_records:
            ce_pairs = [
                (r["query_sent"], tax.cross_encoder_template.format(text=r["text"].lower()))
                for r in flat_candidate_records
            ]
            raw_ce_preds = self.cross_encoder.predict(
                ce_pairs,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            scaled_scores = 1.0 / (1.0 + np.exp(-(raw_ce_preds + 1.5) / self.cross_temp))
            for r, score in zip(flat_candidate_records, scaled_scores):
                r["final_score"] = float(score)
        else:
            for r in flat_candidate_records:
                r["final_score"] = r["bi_score"]


        # Apply Cutoff
        selected_records = []
        for i in range(len(all_data)):
            start, end = slice_offsets[i], slice_offsets[i + 1]
            q_records = flat_candidate_records[start:end]
            q_records.sort(key=lambda x: x["final_score"], reverse=True)

            scores = [r["final_score"] for r in q_records]
            cutoff = min(top_k, len(q_records)) if top_k is not None else self._find_adaptive_elbow(scores)
            selected_records.extend(q_records[:cutoff])


        # Optional NLI Verification
        if verify_nli and self.nli_model and selected_records:
            nli_pairs = [(r["text"], r["query_sent"]) for r in selected_records]
            nli_scores = self._score_nli_entailment(
                nli_pairs,
                template=tax.nli_template,
                batch_size=128,
                min_nli_margin=0.10
            )
            for r, n_score in zip(selected_records, nli_scores):
                r["nli_score"] = n_score
        else:
            for r in selected_records:
                r["nli_score"] = None

        # Filter by minimum confidence and structure output
        aggregated_by_doc: List[Dict[str, Dict[str, Any]]] = [{} for _ in range(len(texts))]
        for r in selected_records:
            if r["final_score"] < self.threshold:
                continue

            if verify_nli and r["nli_data"] is not None:
                if not r["nli_data"]["verified"]:
                    continue

            d_idx = r["orig_doc_idx"]
            c_id = r["code"]
            curr_score = round(r["final_score"], 4)
            v_score = (
                round(r["nli_data"]["entailment_prob"], 4)
                if (verify_nli and r["nli_data"] is not None)
                else None
            )

            evidence_item = {
                "sentence": r["query_sent"],
                "score": curr_score,
            }
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
                if verify_nli:
                    doc_store[c_id]["verification_score"] = v_score
            else:
                doc_store[c_id]["frequency"] += 1
                doc_store[c_id]["evidence"].append(evidence_item)

                if curr_score > doc_store[c_id]["score"]:
                    doc_store[c_id]["score"] = curr_score
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