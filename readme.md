# Paediatric Appendicitis AI System

An artificial intelligence system for automatically grading paediatric appendicitis severity from free-text operative and ultrasound reports.

## Project Overview

This research develops and evaluates an AI system to automate the labelling and grading of paediatric appendicitis from clinical narratives, comparing performance against manual data extraction and ChatGPT-4.

## Features

- **Data Preprocessing**: Automated cleaning and standardization of clinical text
- **Feature Engineering**: TF-IDF, clinical NER, and BERT embeddings
- **Multiple Models**: Traditional ML (Logistic Regression, Random Forest, SVM) and Deep Learning (LSTM, BERT)
- **Comprehensive Evaluation**: Comparison against manual extraction and ChatGPT-4
- **Reproducible Pipeline**: Modular, configurable, and well-documented codebase

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/paediatric_appendicitis_ai.git
cd paediatric_appendicitis_ai
```

2. Create and activate a conda environment:
```
conda create -n appendicitis-ai python=3.8
conda activate appendicitis-ai
```

3. Install dependencies:
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4.Install scispaCy model:
```
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
```
