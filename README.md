# Decoding Path Recovery

This repository contains the replication code and evaluation pipeline for the paper *"Decoding Path Recovery in Number Generation: Shifting to Intermediate Layers and Top-k Ranks"*.

## Setup

Use Python 3.10 or newer, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Inference uses Hugging Face models and may require a CUDA-enabled PyTorch installation. Follow each model's access and authentication requirements before running it.

## Data and Usage

The inference entry point expects a JSONL dataset. Each non-empty line must contain `equation`, `answer`, and the prompt field selected by `--type` (normally `question`). The default dataset name is `FERMAT_all.json`; place it in the repository root or pass its path explicitly.

Run inference with:

```bash
python inference.py \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--dataset /path/to/FERMAT_all.json \
	--type question \
	--output_file inference_eos_uni/inference_data_llama_fermat_question.json
```

Analysis scripts read JSON results from `inference_eos_uni` and can be run individually, for example:

```bash
python analysis/analysis_recovery.py
```

The FERMAT dataset is not redistributed in this repository. Obtain it from the [FERMAT project repository](https://github.com/jasivan/FERMAT/tree/main) and pass the local JSONL path with `--dataset`.

## Publication

The paper was published in the *Proceedings of the 19th International Natural Language Generation Conference (INLG 2026)*, pages 799-814.

- [Read the paper](https://2026.inlgmeeting.org/static/proceedings/long/watermarked_pdfs/45.pdf)

## Licence

This repository is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

For questions about the paper or this repository, contact:

- Jasivan Alex Sivakumar: jasivakumar1@sheffield.ac.uk
- Nafise Sadat Moosavi: n.s.moosavi@sheffield.ac.uk

## Citation

```bibtex
@inproceedings{sivakumar-moosavi-2026-decoding,
	title = "Decoding Path Recovery in Number Generation: Shifting to Intermediate Layers and Top-k Ranks",
	author = "Sivakumar, Jasivan Alex and Moosavi, Nafise Sadat",
	booktitle = "Proceedings of the 19th International Natural Language Generation Conference",
	month = oct,
	year = "2026",
	address = "Utrecht, the Netherlands",
	pages = "799--814",
	publisher = "Association for Computational Linguistics",
	url = "https://2026.inlgmeeting.org/static/proceedings/long/watermarked_pdfs/45.pdf"
}
```

---

## 📁 Repository Structure

```text
├── analysis/               # Evaluation scripts used to generate graphs and tables
├── inference.py            # Main entry point to execute the evaluation on LLMs
├── .gitignore              # Excludes generated results and local artifacts
├── CITATION.cff             # Machine-readable citation metadata
├── requirements.txt         # Python dependencies
└── README.md               # Repository documentation