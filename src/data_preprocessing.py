import pandas as pd
import numpy as np
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Load only CPU-efficient NLP models
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        except OSError:
            logger.warning("SpaCy model not found. Using simple tokenization.")
            self.nlp = None

        self.label_encoder = LabelEncoder()

    def load_data(self, data_type="merged"):
        """Load raw data from CSV files."""
        data_config = self.config['data']
        file_path = f"{data_config['raw_path']}{data_config['file_names'][data_type]}"

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data from {file_path}. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None

    def clean_clinical_text(self, text):
        """Clean and preprocess clinical text (CPU efficient)."""
        if pd.isna(text):
            return ""

        # Basic cleaning operations
        text = str(text).lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\[\*.*?\*\]', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def simple_tokenize(self, text):
        """Simple tokenization without heavy NLP."""
        if not text:
            return []
        return text.split()

    def advanced_text_processing(self, text):
        """Advanced processing with light CPU usage."""
        if self.nlp is None or not text:
            # Fallback to simple processing
            tokens = self.simple_tokenize(text)
            return ' '.join(tokens)

        doc = self.nlp(text)
        processed_tokens = []

        for token in doc:
            if (not token.is_stop and
                    not token.is_punct and
                    token.is_alpha and
                    len(token.lemma_) > 2):
                processed_tokens.append(token.lemma_)

        return ' '.join(processed_tokens)

    def extract_medical_keywords(self, text):
        """Extract medical keywords using simple pattern matching."""
        medical_terms = [
            'appendicitis', 'perforated', 'gangrenous', 'inflamed', 'abscess',
            'appendicolith', 'fluid', 'thickening', 'diameter', 'normal',
            'simple', 'complicated', 'phlegmon', 'contamination'
        ]

        found_terms = []
        text_lower = text.lower()
        for term in medical_terms:
            if term in text_lower:
                found_terms.append(term)

        return found_terms

    def prepare_dataset(self, df, text_column, target_column):
        """Prepare the dataset for training."""
        logger.info("Starting data preprocessing...")

        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_clinical_text)

        # Advanced processing
        df['processed_text'] = df['cleaned_text'].apply(self.advanced_text_processing)

        # Extract medical keywords
        df['medical_keywords'] = df['cleaned_text'].apply(self.extract_medical_keywords)

        # Encode target labels
        df['label_encoded'] = self.label_encoder.fit_transform(df[target_column])

        logger.info(f"Label classes: {list(self.label_encoder.classes_)}")
        logger.info(f"Label distribution:\n{df[target_column].value_counts()}")

        return df

    def split_data(self, df, text_column='processed_text', target_column='label_encoded'):
        """Split data into train, validation, and test sets."""
        prep_config = self.config['preprocessing']

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=prep_config['test_size'],
            stratify=df[target_column],
            random_state=prep_config['random_state']
        )

        # Second split: separate validation set from train+val
        val_ratio = prep_config['val_size'] / (1 - prep_config['test_size'])
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            stratify=train_val_df[target_column],
            random_state=prep_config['random_state']
        )

        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Validation set: {len(val_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")

        return train_df, val_df, test_df

    def save_processed_data(self, train_df, val_df, test_df, suffix=""):
        """Save processed datasets."""
        import os
        processed_path = self.config['data']['processed_path']
        os.makedirs(processed_path, exist_ok=True)

        train_df.to_csv(f"{processed_path}train_data{suffix}.csv", index=False)
        val_df.to_csv(f"{processed_path}val_data{suffix}.csv", index=False)
        test_df.to_csv(f"{processed_path}test_data{suffix}.csv", index=False)

        # Save label encoder
        import joblib
        joblib.dump(self.label_encoder, f"{processed_path}label_encoder{suffix}.pkl")

        logger.info(f"Processed data saved to {processed_path}")