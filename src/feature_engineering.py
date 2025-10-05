import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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

    def extract_tfidf_features(self, train_texts, test_texts, val_texts=None):
        """Extract TF-IDF features from text."""
        tfidf_config = self.config['features']['tfidf']

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config['max_features'],
            ngram_range=tuple(tfidf_config['ngram_range']),
            stop_words='english',
            min_df=2,  # Ignore terms that appear in only 1 document
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            n_jobs=1  # Single-threaded
        )

        # Fit on training data
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(train_texts)
        X_test_tfidf = self.tfidf_vectorizer.transform(test_texts)

        logger.info(f"TF-IDF features shape - Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}")

        if val_texts is not None:
            X_val_tfidf = self.tfidf_vectorizer.transform(val_texts)
            return X_train_tfidf, X_val_tfidf, X_test_tfidf

        return X_train_tfidf, X_test_tfidf

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

    # Save features
    feature_engineer.save_features(
        [X_train_tfidf, X_test_tfidf],
        ['X_train_tfidf', 'X_test_tfidf']
    )