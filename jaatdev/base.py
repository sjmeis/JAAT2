import os
import psutil
from typing import List, Tuple, Any, Optional
import torch
import re
from torch.utils.data import Dataset
from transformers.utils import is_flash_attn_2_available
from wtpsplit import SaT

from .config import logger, MODEL_CACHE

BULLET_PATTERN = re.compile(r"(^|\n)[\s•·\-\*]+\s*")
DELIM_PATTERN = re.compile(r"[;]\s*")
CLEAN_PATTERN = re.compile(r"(;|\s\+\s|\s\*\s|\s\-\s|\s•\s|\s·\s|--|\*\*)")
MULTI_PUNCT = re.compile(r"\.{2,}")
MULTI_SPACE = re.compile(r"[ \t]+")
TRAILING_PUNCT = re.compile(r"^[ \t\n\r.\-•*]+|[ \t\n\r.\-•*]+$")
CLAUSE_DELIMS = re.compile(r"(\s*;\s*|\s*—\s*|\s*--\s*)")

def get_device_settings() -> Tuple[str, int]:
    """
    Detects hardware capabilities and derives a sensible batch size.
    """

    # CUDA
    if torch.cuda.is_available():
        device = "cuda"
        
        try:
            device_props = torch.cuda.get_device_properties(0)
            total_vram_gb = device_props.total_memory / (1024 ** 3)
        except Exception:
            total_vram_gb = 8.0 

        if total_vram_gb >= 40.0:
            batch_size = 2048
        elif total_vram_gb >= 20.0:
            batch_size = 1024
        elif total_vram_gb >= 14.0:
            batch_size = 512
        elif total_vram_gb >= 8.0:
            batch_size = 256
        else:
            batch_size = 64

        return device, batch_size

    # Apple Silicon (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        device = "mps"
        
        if total_ram_gb >= 32.0:
            batch_size = 256
        elif total_ram_gb >= 16.0:
            batch_size = 128
        else:
            batch_size = 64
            
        return device, batch_size

    # CPU Fallback
    device = "cpu"
    cpu_count = os.cpu_count() or 4
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    if cpu_count >= 16 and total_ram_gb >= 32.0:
        batch_size = 64
    else:
        batch_size = 32

    return device, batch_size

def get_optimal_attn_implementation() -> str:
    """
    Selects the fastest attention backend available for the current hardware.
    """
    if not torch.cuda.is_available():
        return "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"

    major, _ = torch.cuda.get_device_capability(0)
    has_compatible_arch = major >= 8

    if has_compatible_arch and is_flash_attn_2_available():
        return "flash_attention_2"

    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"

    return "eager"

class ListDataset(Dataset):
    def __init__(self, original_list: List[Any]) -> None:
        self.original_list = original_list

    def __len__(self) -> int:
        return len(self.original_list)

    def __getitem__(self, i: int) -> Any:
        return self.original_list[i]

def clean_text(text: str) -> str:
    """Pre-normalizes delimiters, bullets, and line breaks into standard clause boundaries."""
    if not text:
        return ""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    normalized_lines = [
        line if line.endswith((".", "!", "?")) else f"{line}."
        for line in (l.strip() for l in text.splitlines())
        if line
    ]
    text = " ".join(normalized_lines)
    text = BULLET_PATTERN.sub(" ", text)

    text = CLAUSE_DELIMS.sub(". ", text)
    text = MULTI_PUNCT.sub(".", text)
    text = MULTI_SPACE.sub(" ", text).strip()
    return text

def filter_valid_sentence(sent: str, min_words: int = 2, max_words: int = 64) -> Optional[str]:
    s = TRAILING_PUNCT.sub("", sent)
    words = s.split()
    word_count = len(words)
    if min_words <= word_count <= max_words:
        return s
    return None

class BatchedSaTSplitter:
    def __init__(
        self,
        model_name: str = "sat-3l-sm",
        device: Optional[str] = None
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.sat = None

        cache_key = f"sat_{model_name}"
        if cache_key not in MODEL_CACHE:
            logger.info(f"Loading SaT segmenter '{model_name}' on {self.device}...")
            sat_instance = SaT(model_name)
            if self.device == "cuda":
                sat_instance.half().to("cuda")
            MODEL_CACHE[cache_key] = sat_instance
        self.sat = MODEL_CACHE[cache_key]

    def split_two_pass(
        self,
        docs: List[str],
        batch_size: int = 1024,
        pass1_threshold: float = 0.50,
        pass2_threshold: float = 0.20,
        min_words: int = 2,
        max_words: int = 48,
        pass2_char_length: int = 80
    ) -> List[List[str]]:
        if not docs:
            return []

        cleaned_docs = [clean_text(d) for d in docs]

        doc_sentences_raw = self.sat.split(
            cleaned_docs,
            batch_size=batch_size,
            threshold=pass1_threshold,
            outer_batch_size=batch_size * 2,
            verbose=False
        )

        doc_sentence_groups: List[List[str]] = []
        pass2_candidates: List[str] = []
        pass2_mapping: List[Tuple[int, int]] = []

        for d_idx, raw_sents in enumerate(doc_sentences_raw):
            filtered_doc_sents = []
            for s in raw_sents:
                clean_s = s.strip()
                if clean_s:
                    filtered_doc_sents.append(clean_s)

            doc_sentence_groups.append(filtered_doc_sents)

            for s_idx, sent in enumerate(filtered_doc_sents):
                if len(sent) >= pass2_char_length or len(sent.split()) > 15: # > (max_words // 2):
                    pass2_candidates.append(sent)
                    pass2_mapping.append((d_idx, s_idx))

        replacement_map: dict[Tuple[int, int], List[str]] = {}
        if pass2_candidates:
            pass2_results = self.sat.split(
                pass2_candidates,
                batch_size=batch_size,
                threshold=pass2_threshold,
                outer_batch_size=batch_size * 2,
                verbose=False
            )

            for (d_idx, s_idx), refined_sents in zip(pass2_mapping, pass2_results):
                valid_splits = [s.strip(" ,;.-•*") for s in refined_sents if s.strip()]
                cleaned_splits = []
                for split in valid_splits:
                    split_cleaned = re.sub(r"^(and|or|whether that's|whether|such as|including)\s+", "", split, flags=re.IGNORECASE)
                    if len(split_cleaned.split()) >= min_words:
                        cleaned_splits.append(split_cleaned)

                if len(cleaned_splits) > 1:
                    replacement_map[(d_idx, s_idx)] = cleaned_splits

        final_docs: List[List[str]] = []
        for d_idx, sents in enumerate(doc_sentence_groups):
            doc_refined = []
            for s_idx, sent in enumerate(sents):
                key = (d_idx, s_idx)
                candidates = replacement_map.get(key, [sent])

                for candidate in candidates:
                    valid_item = filter_valid_sentence(candidate, min_words, max_words)
                    if valid_item is not None:
                        doc_refined.append(valid_item)

            final_docs.append(doc_refined)

        return final_docs

_GLOBAL_SPLITTER: Optional[BatchedSaTSplitter] = None

def get_sentence_splitter(device: Optional[str] = None) -> BatchedSaTSplitter:
    global _GLOBAL_SPLITTER
    if _GLOBAL_SPLITTER is None:
        _GLOBAL_SPLITTER = BatchedSaTSplitter(device=device)
    return _GLOBAL_SPLITTER

def sent_tokenize(text: str) -> List[str]:
    splitter = get_sentence_splitter()
    results = splitter.split_two_pass([text])
    return results[0] if results else []