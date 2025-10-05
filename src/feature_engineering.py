import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from transformers import AutoTokenizer, AutoModel
import torch
import yaml
import logging
import joblib
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.tfidf_vectorizer = None
        self.bert_tokenizer = None
        self.bert_model = None

    def initialize_bert_model(self):
        """Initialize BERT model for feature extraction."""
        model_name = self.config['features']['embeddings']['model_name']
        logger.info(f"Loading BERT model: {model_name}")

        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model.to(self.device)
        self.bert_model.eval()

    def extract_tfidf_features(self, train_texts, test_texts, val_texts=None):
        """Extract TF-IDF features from text."""
        tfidf_config = self.config['features']['tfidf']

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config['max_features'],
            ngram_range=tuple(tfidf_config['ngram_range']),
            stop_words='english'
        )

        # Fit on training data
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(train_texts)
        X_test_tfidf = self.tfidf_vectorizer.transform(test_texts)

        logger.info(f"TF-IDF features shape - Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}")

        if val_texts is not None:
            X_val_tfidf = self.tfidf_vectorizer.transform(val_texts)
            return X_train_tfidf, X_val_tfidf, X_test_tfidf

        return X_train_tfidf, X_test_tfidf

    def get_bert_embeddings(self, texts, batch_size=16):
        """Generate BERT embeddings for a list of texts."""
        if self.bert_model is None:
            self.initialize_bert_model()

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT embeddings"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config['preprocessing']['max_sequence_length'],
                return_tensors="pt"
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding as sentence representation
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def create_ner_features(self, entity_lists, entity_vocab):
        """Create binary features for medical entities."""
        features = np.zeros((len(entity_lists), len(entity_vocab)))

        for i, entities in enumerate(entity_lists):
            for entity in entities:
                if entity in entity_vocab:
                    features[i, entity_vocab[entity]] = 1

        return features

    def build_entity_vocabulary(self, entity_lists, min_freq=5):
        """Build vocabulary of medical entities."""
        from collections import Counter

        # Count entity frequencies
        entity_counter = Counter()
        for entities in entity_lists:
            entity_counter.update(entities)

        # Filter by minimum frequency
        entity_vocab = {entity: idx for idx, (entity, count) in
                        enumerate(entity_counter.items()) if count >= min_freq}

        logger.info(f"Built entity vocabulary with {len(entity_vocab)} entities")
        return entity_vocab

    def reduce_dimensionality(self, features, n_components=100):
        """Reduce feature dimensionality using TruncatedSVD."""
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_features = svd.fit_transform(features)

        logger.info(f"Reduced features from {features.shape[1]} to {n_components} dimensions")
        return reduced_features, svd

    def save_features(self, features, feature_names, suffix=""):
        """Save extracted features."""
        import os
        features_path = self.config['data']['features_path']
        os.makedirs(features_path, exist_ok=True)

        for name, feature in zip(feature_names, features):
            if hasattr(feature, 'toarray'):
                # Sparse matrix
                feature = feature.toarray()

            np.save(f"{features_path}{name}{suffix}.npy", feature)

        # Save vectorizers
        if self.tfidf_vectorizer:
            joblib.dump(self.tfidf_vectorizer, f"{features_path}tfidf_vectorizer{suffix}.pkl")

        logger.info(f"Features saved to {features_path}")


# Example usage
if __name__ == "__main__":
    feature_engineer = FeatureEngineer()

    # Load processed data
    train_df = pd.read_csv("data/processed/train_data.csv")
    test_df = pd.read_csv("data/processed/test_data.csv")

    # Extract TF-IDF features
    X_train_tfidf, X_test_tfidf = feature_engineer.extract_tfidf_features(
        train_df['processed_text'],
        test_df['processed_text']
    )

    # Extract BERT embeddings (for smaller subset if needed)
    sample_texts = train_df['processed_text'].tolist()[:100]  # Sample for demo
    bert_embeddings = feature_engineer.get_bert_embeddings(sample_texts)

    # Save features
    feature_engineer.save_features(
        [X_train_tfidf, X_test_tfidf, bert_embeddings],
        ['X_train_tfidf', 'X_test_tfidf', 'bert_embeddings_sample']
    )