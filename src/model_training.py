import pandas as pd
import numpy as np
import joblib
import yaml
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from scipy.sparse import issparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.models = {}
        self.best_model = None

    def load_features(self, feature_names, suffix=""):
        """Load precomputed features."""
        features_path = self.config['data']['features_path']
        features = []

        for name in feature_names:
            try:
                # Try loading as sparse matrix first
                file_path = f"{features_path}{name}{suffix}.npz"
                sparse_data = np.load(file_path, allow_pickle=True)
                from scipy.sparse import csr_matrix
                feature = csr_matrix((sparse_data['data'], sparse_data['indices'],
                                      sparse_data['indptr']), shape=sparse_data['shape'])
            except FileNotFoundError:
                # Try loading as dense matrix
                try:
                    file_path = f"{features_path}{name}{suffix}.npy"
                    feature = np.load(file_path, allow_pickle=True)
                except FileNotFoundError:
                    logger.error(f"Feature file not found: {file_path}")
                    feature = None

            features.append(feature)
            if feature is not None:
                logger.info(f"Loaded {name}: {feature.shape}")

        return features

    def train_baseline_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train traditional machine learning models (CPU optimized, single-threaded)."""
        baseline_config = self.config['models']['baseline']

        # Convert to dense if sparse and small enough
        if issparse(X_train) and X_train.shape[1] < 10000:
            X_train = X_train.toarray()
            if X_val is not None and issparse(X_val):
                X_val = X_val.toarray()

        models = {
            'logistic_regression': LogisticRegression(
                **baseline_config['logistic_regression'],
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                **baseline_config['random_forest'],
                random_state=42
            ),
            'svm': SVC(
                **baseline_config['svm'],
                random_state=42,
                probability=True
            ),
            'xgboost': xgb.XGBClassifier(
                **baseline_config['xgboost'],
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                **baseline_config['lightgbm'],
                random_state=42,
                verbose=-1  # Suppress output
            )
        }

        trained_models = {}
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model

            # Optional validation
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                logger.info(f"{name} validation accuracy: {accuracy:.4f}")

        self.models.update(trained_models)
        return trained_models

    def hyperparameter_tuning(self, model, param_grid, X_train, y_train, cv=3):
        """Perform hyperparameter tuning using RandomizedSearchCV (CPU efficient)."""
        search = RandomizedSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            n_iter=10,  # Limit iterations for CPU efficiency
            random_state=42
        )

        search.fit(X_train, y_train)

        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best cross-validation score: {search.best_score_:.4f}")

        return search.best_estimator_

    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """Train an ensemble of models."""
        from sklearn.ensemble import VotingClassifier

        # First train individual models
        individual_models = self.train_baseline_models(X_train, y_train, X_val, y_val)

        # Create voting classifier
        estimators = [(name, model) for name, model in individual_models.items()]
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')

        logger.info("Training ensemble model...")
        voting_clf.fit(X_train, y_train)

        self.models['ensemble'] = voting_clf
        return voting_clf

    def select_best_model(self, X_val, y_val):
        """Select the best performing model on validation set."""
        best_score = 0
        best_model_name = None

        for name, model in self.models.items():
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name

        self.best_model = self.models[best_model_name]
        logger.info(f"Best model: {best_model_name} with accuracy: {best_score:.4f}")

        return best_model_name, self.best_model

    def save_models(self, suffix=""):
        """Save trained models."""
        import os
        models_path = "models/"
        os.makedirs(models_path, exist_ok=True)

        for name, model in self.models.items():
            joblib.dump(model, f"{models_path}{name}_model{suffix}.pkl")

        if self.best_model:
            joblib.dump(self.best_model, f"{models_path}best_model{suffix}.pkl")

        logger.info(f"Models saved to {models_path}")


def train_cpu_pipeline():
    """Convenience function to train all models quickly."""
    trainer = ModelTrainer()

    # Load features
    X_train = trainer.load_features(['X_train_combined'])[0]

    # Load labels
    train_df = pd.read_csv("data/processed/train_data.csv")
    y_train = train_df['label_encoded']

    # Load validation data if available
    try:
        val_df = pd.read_csv("data/processed/val_data.csv")
        X_val = trainer.load_features(['X_val_combined'])[0]
        y_val = val_df['label_encoded']
    except:
        X_val, y_val = None, None

    # Train models
    if X_val is not None:
        models = trainer.train_baseline_models(X_train, y_train, X_val, y_val)
        best_name, best_model = trainer.select_best_model(X_val, y_val)
    else:
        models = trainer.train_baseline_models(X_train, y_train)

    # Train ensemble
    ensemble = trainer.train_ensemble(X_train, y_train, X_val, y_val)

    trainer.save_models()

    return trainer.models


if __name__ == "__main__":
    models = train_cpu_pipeline()
    print("✅ All models trained and saved successfully!")