import pandas as pd
import numpy as np
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Load NLP models
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.nlp_medical = spacy.load('en_core_sci_sm')
        except OSError:
            logger.warning("SpaCy models not found. Please install them.")
            self.nlp = None
            self.nlp_medical = None

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
        """Clean and preprocess clinical text."""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove numbers and specific patterns
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\[\*.*?\*\]', '', text)  # Remove de-identification markers

        # Remove punctuation and special characters, but keep clinical terms
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def advanced_text_processing(self, text):
        """Advanced NLP processing with spaCy."""
        if self.nlp is None or not text:
            return text

        doc = self.nlp(text)

        # Lemmatization and filtering
        processed_tokens = []
        for token in doc:
            if (not token.is_stop and
                    not token.is_punct and
                    token.is_alpha and
                    len(token.lemma_) > 2):
                processed_tokens.append(token.lemma_)

        return ' '.join(processed_tokens)

    def extract_medical_entities(self, text):
        """Extract medical entities using scispaCy."""
        if self.nlp_medical is None or not text:
            return []

        doc = self.nlp_medical(text)
        entities = [ent.text for ent in doc.ents]
        return entities

    def prepare_dataset(self, df, text_column, target_column):
        """Prepare the dataset for training."""
        logger.info("Starting data preprocessing...")

        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_clinical_text)

        # Advanced processing
        df['processed_text'] = df['cleaned_text'].apply(self.advanced_text_processing)

        # Extract medical entities
        df['medical_entities'] = df['cleaned_text'].apply(self.extract_medical_entities)

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

        logger.info(f"Processed data saved to {processed_path}")


# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()

    # Load and preprocess data
    df = preprocessor.load_data("merged")
    if df is not None:
        df = preprocessor.prepare_dataset(
            df,
            text_column='report_text',
            target_column='severity_grade'
        )

        # Split data
        train_df, val_df, test_df = preprocessor.split_data(df)

        # Save processed data
        preprocessor.save_processed_data(train_df, val_df, test_df)