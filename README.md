# Decoding Path Recovery

This repository contains the replication code and evaluation pipeline for the paper *"Decoding Path Recovery in Number Generation: Shifting to Intermediate Layers and Top-k Ranks"*.

## Publication

The paper was published in the *Proceedings of the 19th International Natural Language Generation Conference (INLG 2026)*, pages 799-814.

- [Read the paper](https://2026.inlgmeeting.org/static/proceedings/long/watermarked_pdfs/45.pdf)

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
	address = "Sheffield, UK",
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
├── .gitignore              # Configured to exclude heavy raw evaluation JSONs
└── README.md               # Repository documentation