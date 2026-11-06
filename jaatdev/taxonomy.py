import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from collections import Counter
import hashlib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import importlib_resources as impresources

from .config import logger, GLOBAL_SETTINGS

_TOKENIZE_RE = re.compile(r"\w+")

def fast_tokenize(text: str) -> List[str]:
    return _TOKENIZE_RE.findall(text.lower())

class GPUBM25:
    """
    GPU-accelerated BM25 index implemented via sparse-dense matrix operations.
    """
    def __init__(self, corpus: List[str], device: str = "cuda", k1: float = 1.5, b: float = 0.75):
        self.device = device
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)

        self.vocab: Dict[str, int] = {}
        tokenized_corpus = [fast_tokenize(doc) for doc in corpus]
        for tokens in tokenized_corpus:
            for t in tokens:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)

        self.vocab_size = len(self.vocab)
        doc_lens = np.array([len(t) for t in tokenized_corpus], dtype=np.float32)
        avgdl = float(np.mean(doc_lens)) if self.corpus_size > 0 else 1.0

        df = Counter()
        for tokens in tokenized_corpus:
            df.update(set(tokens))

        idf_arr = np.zeros(self.vocab_size, dtype=np.float32)
        for term, term_id in self.vocab.items():
            freq = df[term]
            # Standard Lucene/BM25 IDF
            idf_arr[term_id] = np.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

        rows, cols, data = [], [], []
        for doc_id, tokens in enumerate(tokenized_corpus):
            counts = Counter(tokens)
            doc_len = doc_lens[doc_id]
            len_norm = 1.0 - b + b * (doc_len / avgdl)

            for t, cnt in counts.items():
                t_id = self.vocab[t]
                tf_weight = (cnt * (k1 + 1.0)) / (cnt + k1 * len_norm)
                val = idf_arr[t_id] * tf_weight
                rows.append(t_id)
                cols.append(doc_id)
                data.append(val)

        if not rows:
            indices = torch.zeros((2, 0), dtype=torch.long)
            values = torch.zeros(0, dtype=torch.float32)
        else:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(data, dtype=torch.float32)
        
        self.doc_matrix_t = torch.sparse_coo_tensor(
            indices, values, size=(self.vocab_size, self.corpus_size), device=self.device
        ).coalesce()

    def query_topk(self, queries: List[str], k: int) -> torch.Tensor:
        """
        Runs BM25 for all queries in a single matrix multiply on GPU.
        Returns tensor of indices (M, k).
        """
        M = len(queries)
        k_clamped = min(k, self.corpus_size)
        if M == 0 or self.corpus_size == 0:
            return torch.zeros((M, k_clamped), dtype=torch.long, device=self.device)
        
        q_rows, q_cols = [], []
        for q_id, q_text in enumerate(queries):
            tokens = fast_tokenize(q_text)
            for t in set(tokens):
                t_id = self.vocab.get(t)
                if t_id is not None:
                    q_rows.append(q_id)
                    q_cols.append(t_id)

        if not q_rows:
            return torch.zeros((M, k_clamped), dtype=torch.long, device=self.device)

        q_indices = torch.tensor([q_rows, q_cols], dtype=torch.long)
        q_values = torch.ones(len(q_rows), dtype=torch.float32)

        query_sparse = torch.sparse_coo_tensor(
            q_indices, q_values, size=(M, self.vocab_size), device=self.device
        ).coalesce()

        scores = torch.sparse.mm(query_sparse, self.doc_matrix_t.to_dense())

        _, top_indices = torch.topk(scores, k=k_clamped, dim=1)
        return top_indices

def _get_cache_path(
    data_source: Union[str, Path],
    model_name: str, 
    prefix: str, 
    df: pd.DataFrame
) -> Path:
    hash_payload = f"{model_name}::{prefix}::{len(df)}::{df['text'].iloc[0]}::{df['text'].iloc[-1]}"
    cache_hash = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()[:12]
    clean_model_str = model_name.replace("/", "_").replace("\\", "_")
    try:
        base_path = Path(str(data_source)).resolve()
        cache_dir = base_path.parent
        test_file = cache_dir / ".test_write"
        test_file.touch()
        test_file.unlink()
        stem = base_path.stem
    except Exception:
        cache_dir = Path.home() / ".cache" / "jaat"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = "taxonomy"

    return cache_dir / f"{stem}_{clean_model_str}_{cache_hash}.pt"

@dataclass
class Taxonomy:
    name: str
    data_source: Union[str, Path, pd.DataFrame]
    id_column: str
    text_column: str
    document_prefix: str = ""
    cross_encoder_template: str = "Demonstrates competence and activity in {text}."
    nli_template: str = "This statement directly involves or demonstrates {text}."

    df: pd.DataFrame = field(init=False, repr=False)
    embeddings: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    codes_arr: np.ndarray = field(init=False, repr=False)
    texts_arr: np.ndarray = field(init=False, repr=False)
    ce_targets: List[str] = field(init=False, repr=False)
    bm25: Optional[GPUBM25] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.data_source, (str, Path)):
            if str(self.data_source).endswith("parquet"):
                raw_df = pd.read_parquet(self.data_source)
            else:
                raw_df = pd.read_csv(self.data_source)
        elif isinstance(self.data_source, pd.DataFrame):
            raw_df = self.data_source.copy()
        else:
            raise ValueError("data_source must be a file path or pandas DataFrame")

        self.df = (
            raw_df[[self.id_column, self.text_column]]
            .drop_duplicates()
            .rename(columns={self.id_column: "id", self.text_column: "text"})
            .reset_index(drop=True)
        )
        self.df["id"] = self.df["id"].astype(str)
        self.df["text"] = self.df["text"].astype(str)

        self.codes_arr = self.df["id"].to_numpy()
        self.texts_arr = self.df["text"].to_numpy()
        
        self.ce_targets = [
            t.strip()
            for t in self.texts_arr
        ]

        self.nli_hypotheses = {
            code: self.nli_template.format(text=text.lower().strip())
            for code, text in zip(self.codes_arr, self.texts_arr)
        }

        self.bm25 = GPUBM25(self.texts_arr.tolist(), device="cuda" if torch.cuda.is_available() else "cpu")

    def build_embeddings(
        self,
        model: SentenceTransformer,
        model_name: str,
        device: str,
        batch_size: int = 128,
        show_progress: bool = False,
        use_cache: bool = True
    ) -> None:
        cache_file = None
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        if use_cache and isinstance(self.data_source, (str, Path)):
            cache_file = _get_cache_path(
                Path(self.data_source), 
                model_name, 
                self.document_prefix, 
                self.df
            )
            if cache_file.exists():
                logger.info(f"Loading precomputed embeddings from {cache_file.name}")
                loaded = torch.load(cache_file, map_location=device, weights_only=True)
                self.embeddings = loaded.to(device=device, dtype=dtype)
                self.embeddings_T = self.embeddings.t().contiguous()
                return

        logger.info(f"Computing embeddings for taxonomy '{self.name}' using {model_name}...")
        texts = [f"{self.document_prefix}{t}" for t in self.df["text"].tolist()]
        embeds = model.encode(
            texts,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=show_progress
        ).to(device, dtype=dtype)
        self.embeddings = F.normalize(embeds, p=2, dim=1)
        self.embeddings_T = self.embeddings.t().contiguous()

        if cache_file is not None:
            try:
                torch.save(self.embeddings, cache_file)
                logger.info(f"Saved precomputed embeddings to {cache_file.name}")
            except Exception as e:
                logger.warning(f"Failed to cache embeddings to disk: {e}")

DEFAULT_TAXONOMIES: Dict[str, Dict[str, Any]] = {
    "tasks": {
        "filename": "Task_DWA.csv",
        "id_column": "Task ID",
        "text_column": "Task",
        "cross_encoder_template": "Job task: {text}.",
        "nli_template": "This statement indicates experience with or responsibility for {text}."
    },
    "skills": {
        "filename": "skills.csv",
        "id_column": "EuropaCode",
        "text_column": "label",
        "cross_encoder_template": "Job skill: {text}.",
        "nli_template": "This statement demonstrates knowledge of or skill in {text}."
    },
    "ai": {
        "filename": "ai.csv",
        "id_column": "Code",
        "text_column": "Statement",
        "cross_encoder_template": "Uses or builds AI systems involving {text}.",
        "nli_template": "This statement indicates the utilization of {text}."
    },
    "title": {
        "filename": "titles.parquet",
        "id_column": "code",
        "text_column": "title",
        "cross_encoder_template": "Represents the role of {text}.",
        "nli_template": "This text indicates the role of {text}."
    }
}

def load_default_taxonomy(name: str) -> Taxonomy:
    """Factory helper to build a default taxonomy instance from JAAT.data."""
    if name not in DEFAULT_TAXONOMIES:
        raise ValueError(f"Unknown default taxonomy '{name}'. Available: {list(DEFAULT_TAXONOMIES.keys())}")
    
    cfg = DEFAULT_TAXONOMIES[name]
    data_path = impresources.files("jaatdev.data") / cfg["filename"]
    
    return Taxonomy(
        name=name,
        data_source=data_path,
        id_column=cfg["id_column"],
        text_column=cfg["text_column"],
        cross_encoder_template=cfg["cross_encoder_template"],
        nli_template=cfg["nli_template"]
    )