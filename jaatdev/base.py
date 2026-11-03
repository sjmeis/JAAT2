from typing import List, Tuple, Any, Optional
import torch
import nltk
import re
from torch.utils.data import Dataset
from wtpsplit import SaT

from .config import logger, MODEL_CACHE

BULLET_PATTERN = re.compile(r"(^|\n)[\s•·\-\*]+\s*")
DELIM_PATTERN = re.compile(r"[;]\s*")
CLEAN_PATTERN = re.compile(r"(;|\s\+\s|\s\*\s|\s\-\s|\s•\s|\s·\s|--|\*\*)")
MULTI_PUNCT = re.compile(r"\.{2,}")
MULTI_SPACE = re.compile(r"[ \t]+")
TRAILING_PUNCT = re.compile(r"^[ \t\n\r.\-•*]+|[ \t\n\r.\-•*]+$")

def get_device_settings() -> Tuple[str, int]:
    if torch.cuda.is_available():
        return "cuda", 1024
    return "cpu", 64

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
    normalized_lines = []
    for line in lines:
        if line[-1] not in ".!?":
            normalized_lines.append(line + ".")
        else:
            normalized_lines.append(line)
    text = ". ".join(normalized_lines)
    text = BULLET_PATTERN.sub(" ", text)
    text = DELIM_PATTERN.sub(". ", text)
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
        pass2_threshold: float = 0.30,
        min_words: int = 2,
        max_words: int = 64,
        pass2_char_length: int = 100
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
                if len(sent) >= pass2_char_length or len(sent.split()) > (max_words // 2):
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
                valid_splits = [s.strip() for s in refined_sents if s.strip()]
                if len(valid_splits) > 1:
                    replacement_map[(d_idx, s_idx)] = valid_splits

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