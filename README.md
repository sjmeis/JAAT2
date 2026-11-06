<div align="center">
  <img src="https://github.com/sjmeis/JAAT2/blob/main/public/logo.png?raw=true" alt="JAAT" width="200"/>

  # Job Ad Analysis Toolkit (JAATv2)

  JAAT is an NLP-powered pipeline for aligning job posting texts (and similar) to standardized taxonomies such as O*NET and ESCO, as well as custom taxonomies describing compentencies. This newest v2 version improves our original (JAAT)[https://github.com/Job-Ad-Research-at-QSB-LUC/JAAT] pipeline with some more advanced matching techniques!

</div>

## Installation

```bash
pip install jaat2
```

Note: for quicker processing (on GPU), we recommend `flash-attn` to be installed, even though it is not a strict requirement.

## Usage
### Single Document Matching
Use `JAAT` to match a single document to any of our default taxonomies:

```python
from jaat2 import Match

matcher = Match(
    threshold=0.70,
    verify_nli=True
)

results = matcher.match(
    text=job_description,
    taxonomy_name="tasks",
    top_k=None, # or set a strict limit
    verify_nli=True
)
```

### Batch Processing
For processing larger corpora, use our optimized batch matcher:

```python
for doc_result in matcher.match_dataset(
    texts=documents,
    taxonomy_name="skills",
    doc_chunk_size=1000,
    verify_nli=True
):
  # Process document-level matches
  pass
```

## Pipeline Overview
This newest version of JAAT expands upon the older one by adding steps both to ensure match accuracy, but also to recognize where multiple standard matches might be applicable.

```
Input Text
   │
   ▼
[1] Two-Pass Segmentation (SaT / wtpsplit)
   │  └── Deconstructs text into clauses
   ▼
[2] Clause Classification
   │  └── Discards non-target candidates
   ▼
[3] Hybrid Retrieval (BM25 + Embeddings)
   │  └── Reciprocal Rank Fusion (RRF) over Top-K candidate pool
   ▼
[4] Pre-Pruning & Cross-Encoder Reranker
   │  └── Evaluates high-probability pairs using cross-attention reranking
   ▼
[5] NLI Verification
   │  └── Eliminates remaining weak matches
   ▼
Aggregated Taxonomy Matches and Corresponding Evidence
```

## Built-in Taxonomies

By default, the package includes four taxonomies for matching:

|Taxonomy|Source             |Description                                                  |
|--------|-------------------|-------------------------------------------------------------|
|tasks   |O*NET Task / DWA   |Task statements from O*NET                                   |
|skills  |ESCO Skills        |European skill framework                                     |
|ai      |AI Task Taxonomy   |Specialized classification of AI/ML task statements          |
|title   |O*NET              |Standardized occupational titles                             |

You can also register a custom taxonomy via:

```python
import pandas as pd
from jaat2.taxonomy import Taxonomy

df = pd.DataFrame({
    "code": ["ENG-01", "ENG-02"],
    "description": [
        "Provision cloud infrastructure using Terraform and AWS.",
        "Configure automated CI/CD pipelines in GitHub Actions."
    ]
})

custom_tax = Taxonomy(
    name="devops_skills",
    data_source=df,
    id_column="code",
    text_column="description",
    cross_encoder_template="DevOps task: {text}.",
    nli_template="The candidate directly implements {text}."
)

matcher.register_taxonomy(custom_tax)
```