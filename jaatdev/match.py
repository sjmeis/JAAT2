from typing import List, Tuple, Union, Generator, Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataclasses import dataclass
import multiprocessing as mp
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, CrossEncoder
import importlib_resources as impresources
import pickle
import json

from .base import ListDataset, get_device_settings, sent_tokenize
from .config import MODEL_CACHE, logger, GLOBAL_SETTINGS
from .utils import progress_bar

def get_shared_model(model_name: str, device: str) -> SentenceTransformer:
    if model_name not in MODEL_CACHE:
        model = SentenceTransformer(model_name, device=device)
        MODEL_CACHE[model_name] = model
    return MODEL_CACHE[model_name]

def get_shared_cross_encoder(model_name: str, device: str) -> CrossEncoder:
    key = f"ce_{model_name}"
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = CrossEncoder(model_name, device=device)
    return MODEL_CACHE[key]

@dataclass
class MatchResult:
    task_id: str
    task: str
    score: float
    query_text: Optional[str] = None
    verification_score: Optional[float] = None

class TaskMatch:
    def __init__(
        self,
        embedding_model: str = "sjmeis/skills-embedding-large-v1",
        cross_encoder_model: str = "sjmeis/task-cross-encoder-v1",
        classification_model: str = "loyoladatamining/task-classifier-mini-v3",
        nli_model: Optional[str] = "cross-encoder/nli-deberta-v3-base",
        query_prefix: str = "",
        document_prefix: str = "",
        initial_top_n: int = 25,
        threshold: float = 0.60,
        cross_temp: float = 1.0,
    ) -> None:
        logger.info("Initializing TaskMatch pipeline...")
        self.device, self.batch_size = get_device_settings()
        self.num_workers = 1 if GLOBAL_SETTINGS.get("single_threaded", False) else int(mp.cpu_count() / 2)

        self.threshold = threshold
        self.initial_top_n = initial_top_n
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.cross_temp = cross_temp

        # Bi-encoder and task embeddings
        logger.info("Setting up bi-encoder and task corpus...")
        self.embedding_model = get_shared_model(embedding_model, self.device)
        
        tasks_df = pd.read_csv(impresources.files("JAAT.data") / "Task_DWA.csv")[["Task ID", "Task"]].drop_duplicates()
        self.tasks = tasks_df.reset_index(drop=True)
        
        prefixed_tasks = [f"{self.document_prefix}{t}" for t in self.tasks["Task"].tolist()]
        self.task_embed = self.embedding_model.encode(
            prefixed_tasks,
            convert_to_tensor=True,
            batch_size=64,
            show_progress_bar=GLOBAL_SETTINGS.get("show_progress", False)
        ).to(self.device)
        self.task_embed = F.normalize(self.task_embed, p=2, dim=1)

        # Candidate extraction pipeline
        logger.info("Setting up candidate extraction classifier...")
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(classification_model)
        self.clf_tokenizer = AutoTokenizer.from_pretrained(classification_model, use_fast=True, max_length=64, truncation=True)
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
            self.cross_encoder = get_shared_cross_encoder(cross_encoder_model, self.device)

        # NLI Model for verification
        self.nli_tokenizer = None
        self.nli_model = None
        self.entail_idx = None
        self.neutral_idx = None
        if nli_model:
            logger.info(f"Setting up NLI verification engine: {nli_model}...")
            self._init_nli(nli_model)

        logger.info("TaskMatch initialized successfully.")

    def _init_nli(self, nli_model_name: str) -> None:
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name, use_fast=True)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
        self.nli_model.eval()

        id2label = {v.lower(): k for k, v in self.nli_model.config.id2label.items()}
        self.entail_idx = id2label.get("entailment", 0)
        self.neutral_idx = id2label.get("neutral", 1)

    def get_candidates_batch(self, texts: List[str]) -> List[Tuple[int, str]]:
        all_data: List[Tuple[int, str]] = []
        for i, t in enumerate(texts):
            s = sent_tokenize(t.strip())
            all_data.extend([(i, ss) for ss in s if 4 < len(ss.split()) <= 48])

        if not all_data:
            return []

        dataset = ListDataset([x[1] for x in all_data])
        predictions = [r for r in progress_bar(self.pipe(dataset))]

        return [
            pair for pair, pred in zip(all_data, predictions) 
            if pred["label"] in ("LABEL_1", "POSITIVE", "1")
        ]

    def _score_cross_encoder(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Expects pairs of (task_desc, context_sent)."""
        if not pairs or not self.cross_encoder:
            return []
        
        conditioned_pairs = [
            (ctx, f"Demonstrates competence and activity in {task.lower()}.")
            for task, ctx in pairs
        ]
        preds = self.cross_encoder.predict(conditioned_pairs)
        # Scaled sigmoid calibration
        return [float(1.0 / (1.0 + np.exp(-(float(x) + 1.5) / self.cross_temp))) for x in preds]

    def _score_nli_entailment(
        self,
        pairs: List[Tuple[str, str]],
        neutral_weight: float = 0.50,
        soften_exp: float = 0.25,
        template: str = "This statement involves or demonstrates the task of {}."
    ) -> List[float]:
        """Runs NLI verification: premise (query/sentence), hypothesis (templated task)."""
        if not pairs or not self.nli_model:
            return []

        premises = [ctx for _, ctx in pairs]
        hypotheses = [template.format(task.lower().strip()) for task, _ in pairs]

        encoded = self.nli_tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.nli_model(**encoded).logits
            probs = F.softmax(logits, dim=-1)
            p_entail = probs[:, self.entail_idx].cpu().numpy()
            p_neutral = probs[:, self.neutral_idx].cpu().numpy()

        combined = p_entail + (neutral_weight * p_neutral)
        return [float(np.clip(s, 0.0, 1.0) ** soften_exp) for s in combined]

    @staticmethod
    def _find_adaptive_elbow(scores: List[float], min_drop: float = 0.05) -> int:
        """Finds the index after the steepest dropoff. Expects descending scores."""
        if len(scores) <= 1:
            return len(scores)

        diffs = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
        max_idx = int(np.argmax(diffs))

        if diffs[max_idx] >= min_drop:
            return max_idx + 1
        return len(scores)

    def get_tasks_batch(
        self,
        texts: List[str],
        top_k: Optional[int] = None,
        verify_nli: bool = False,
        return_original: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        all_data = self.get_candidates_batch(texts)
        results: List[List[Dict[str, Any]]] = [[] for _ in range(len(texts))]
        if not all_data:
            return results

        # Embedding coarse search
        queries = [f"{self.query_prefix}{d[1]}" for d in all_data]
        q_embed = self.embedding_model.encode(
            queries,
            convert_to_tensor=True,
            batch_size=64,
            show_progress_bar=GLOBAL_SETTINGS.get("show_progress", False)
        ).to(self.device)
        q_embed = F.normalize(q_embed, p=2, dim=1)

        with torch.no_grad():
            sim_matrix = torch.mm(q_embed, self.task_embed.T)

        k_retrieval = min(self.initial_top_n, self.task_embed.shape[0])
        top_scores, top_indices = torch.topk(sim_matrix, k=k_retrieval, dim=1)

        # Per-candidate Reranking
        for i, (orig_doc_idx, query_sentence) in enumerate(all_data):
            candidate_indices = top_indices[i].cpu().tolist()
            bi_scores = top_scores[i].cpu().tolist()
            candidate_tasks = self.tasks.iloc[candidate_indices]

            # Cross-Encoder Rerank
            if self.cross_encoder:
                pairs = [(row["Task"], query_sentence) for _, row in candidate_tasks.iterrows()]
                fine_scores = self._score_cross_encoder(pairs)
            else:
                fine_scores = bi_scores

            ranked_items = sorted(
                zip(fine_scores, candidate_tasks.iterrows()),
                key=lambda x: x[0],
                reverse=True
            )

            # Apply Cutoff
            scores_only = [item[0] for item in ranked_items]
            if top_k is not None:
                cutoff = min(top_k, len(ranked_items))
            else:
                cutoff = self._find_adaptive_elbow(scores_only)

            selected = ranked_items[:cutoff]

            # Optional NLI Verification
            nli_scores = []
            if verify_nli and self.nli_model and selected:
                v_pairs = [(row["Task"], query_sentence) for _, (_, row) in selected]
                nli_scores = self._score_nli_entailment(v_pairs)
            else:
                nli_scores = [None] * len(selected)

            # Filter by minimum confidence and structure output
            for (score, (_, row)), nli_s in zip(selected, nli_scores):
                if score < self.threshold:
                    continue

                item_dict = {
                    "task_id": str(row["Task ID"]),
                    "task": row["Task"],
                    "score": round(score, 4),
                }
                if verify_nli:
                    item_dict["verification_score"] = round(nli_s, 4) if nli_s is not None else None
                if return_original:
                    item_dict["query_sentence"] = query_sentence

                results[orig_doc_idx].append(item_dict)

        return results

    def get_tasks(
        self,
        text: str,
        top_k: Optional[int] = None,
        verify_nli: bool = False,
        return_original: bool = False,
    ) -> List[Dict[str, Any]]:
        return self.get_tasks_batch(
            [text],
            top_k=top_k,
            verify_nli=verify_nli,
            return_original=return_original
        )[0]