import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import hashlib
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import importlib_resources as impresources
from rank_bm25 import BM25Okapi

from .config import logger, GLOBAL_SETTINGS

_TOKENIZE_RE = re.compile(r"\w+")

def bm25_tokenize(text: str) -> List[str]:
    return _TOKENIZE_RE.findall(text.lower())

def _get_cache_path(
    base_path: Path, 
    model_name: str, 
    prefix: str, 
    df: pd.DataFrame
) -> Path:
    hash_payload = f"{model_name}::{prefix}::{len(df)}::{df['text'].iloc[0]}::{df['text'].iloc[-1]}"
    cache_hash = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()[:12]
    clean_model_str = model_name.replace("/", "_").replace("\\", "_")
    return base_path.parent / f"{base_path.stem}_{clean_model_str}_{cache_hash}.pt"

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
    bm25: Optional[BM25Okapi] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.data_source, (str, Path)):
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
            self.cross_encoder_template.format(text=t.lower()) 
            for t in self.texts_arr
        ]

        corpus_tokenized = [bm25_tokenize(t) for t in self.df["text"].tolist()]
        self.bm25 = BM25Okapi(corpus_tokenized)

    def build_embeddings(
        self,
        model: SentenceTransformer,
        model_name: str,
        device: str,
        batch_size: int = 64,
        show_progress: bool = False,
        use_cache: bool = True
    ) -> None:
        cache_file = None
        if use_cache and isinstance(self.data_source, (str, Path)):
            cache_file = _get_cache_path(
                Path(self.data_source), 
                model_name, 
                self.document_prefix, 
                self.df
            )
            if cache_file.exists():
                logger.info(f"Loading precomputed embeddings from {cache_file.name}")
                self.embeddings = torch.load(cache_file, map_location=device, weights_only=True)
                return

        logger.info(f"Computing embeddings for taxonomy '{self.name}' using {model_name}...")
        texts = [f"{self.document_prefix}{t}" for t in self.df["text"].tolist()]
        embeds = model.encode(
            texts,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=show_progress
        ).to(device)
        self.embeddings = F.normalize(embeds, p=2, dim=1)

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
        "cross_encoder_template": "Demonstrates competence and activity in {text}.",
        "nli_template": "This statement directly describes performing the work of {text}."
    },
    "skills": {
        "filename": "skills.csv",
        "id_column": "EuropaCode",
        "text_column": "label",
        "cross_encoder_template": "Demonstrates proficiency and application of {text}.",
        "nli_template": "This statement demonstrates personal proficiency in {text}."
    },
    "ai": {
        "filename": "ai_a6_5_redacted_final2.csv",
        "id_column": "Code",
        "text_column": "Statement",
        "cross_encoder_template": "Uses or builds AI systems involving {text}.",
        "nli_template": "This statement indicates the active utilization of {text}."
    }
}

def load_default_taxonomy(name: str) -> Taxonomy:
    """Factory helper to build a default taxonomy instance from JAAT.data."""
    if name not in DEFAULT_TAXONOMIES:
        raise ValueError(f"Unknown default taxonomy '{name}'. Available: {list(DEFAULT_TAXONOMIES.keys())}")
    
    cfg = DEFAULT_TAXONOMIES[name]
    data_path = impresources.files("JAAT.data") / cfg["filename"]
    
    return Taxonomy(
        name=name,
        data_source=data_path,
        id_column=cfg["id_column"],
        text_column=cfg["text_column"],
        cross_encoder_template=cfg["cross_encoder_template"],
        nli_template=cfg["nli_template"]
    )