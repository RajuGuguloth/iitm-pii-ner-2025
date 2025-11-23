
# PII Entity Recognition from Noisy Speech-to-Text (STT) Transcripts

**IIT Madras Assignment 2025**

## Overview

This project implements a high-precision Named Entity Recognition (NER) system for identifying Personally Identifiable Information (PII) entities in noisy Speech-to-Text (STT) transcripts. The model is designed for challenging, real-world data simulating errors and variability from ASR (Automatic Speech Recognition) systems.

- **Goal:** Detect token-level PII entities in noisy STT, output entity spans as character offsets.
- **Focus:** High precision for PII entities, efficient CPU inference.

## Entity Types

- **PII Entities** (`PII = true`):
  - `CREDITCARD`, `PHONE`, `EMAIL`, `PERSONNAME`, `DATE`
- **Non-PII Entities** (`PII = false`):
  - `CITY`, `LOCATION`

## Data

- Data is stored in the `data/` folder.
- Format: JSONL, one example per line, with fields:
  - `id`: unique identifier
  - `text`: noisy transcript
  - `entities`: list of `{start, end, label}` for each entity (character offsets)
- You must generate your own train set (500–1000 examples) and dev set (100–200 examples).

## Project Structure

```
pii-ner-noisy-stt/
├── data/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── out/
│   └── (saved models, logs, predictions)
├── src/
│   ├── dataset.py
│   ├── labels.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── eval_span_f1.py
│   └── measure_latency.py
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Generate Data

- Use the provided script(s) or your own logic to generate noisy `train.jsonl` and `dev.jsonl` as described above.

### 3. Train the Model

```bash
python src/train.py --model_name distilbert-base-uncased --output_dir out/
```

### 4. Run Predictions

```bash
python src/predict.py --model_dir out/ --input data/test.jsonl --output out/predictions.jsonl
```

### 5. Evaluate Results

```bash
python src/eval_span_f1.py --pred out/predictions.jsonl --label data/dev.jsonl
```

### 6. Measure Latency

```bash
python src/measure_latency.py --model_dir out/ --input data/dev.jsonl
```

## Model & Approach

- The default model is a BERT-style token classifier (e.g., DistilBERT) trained via BIO tagging.
- Data is pre-processed for noisy transcript patterns typical of real STT output.
- Loss and evaluation are tuned for high **PII precision** (≥0.80 is target).
- Span decoding outputs character offset matches for robust entity localization.
- All outputs, models, and logs are saved in the `out/` directory.

## Evaluation Metrics

- Character-level, per-entity F1 score (with primary focus on PII entities).
- p50 and p95 latency measurements (in ms) on CPU, batch size 1.

## Notes

- Regex and dictionary heuristics are allowed only as helpers, NOT the primary detector.
- Codebase is modular; you may swap architectures in `src/model.py` for improved results under constraints.




***

This README outlines your project’s objective, structure, data, quick start, methodology, and evaluation for clear communication with graders or recruiters. Adjust as needed for personal details or extra technical sections.
