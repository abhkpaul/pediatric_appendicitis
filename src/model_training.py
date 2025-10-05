import pandas as pd
import numpy as np
import joblib
import yaml
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

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
            feature = np.load(f"{features_path}{name}{suffix}.npy")
            features.append(feature)
            logger.info(f"Loaded {name}: {feature.shape}")

        return features

    def train_baseline_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train traditional machine learning models."""
        baseline_config = self.config['models']['baseline']

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
                random_state=42,
                eval_metric='logloss'
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

    def train_lstm_model(self, X_train, y_train, X_val=None, y_val=None, vocab_size=10000, embedding_dim=100):
        """Train LSTM model for text classification."""
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

        lstm_config = self.config['models']['neural_networks']['lstm']

        # Tokenize text
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(X_train)

        # Convert texts to sequences
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.config['preprocessing']['max_sequence_length'])

        if X_val is not None:
            X_val_seq = tokenizer.texts_to_sequences(X_val)
            X_val_pad = pad_sequences(X_val_seq, maxlen=self.config['preprocessing']['max_sequence_length'])

        # Build LSTM model
        num_classes = len(np.unique(y_train))
        model = Sequential([
            Embedding(vocab_size, lstm_config['embedding_dim'], input_length=X_train_pad.shape[1]),
            LSTM(lstm_config['hidden_dim'], return_sequences=True, dropout=lstm_config['dropout_rate']),
            LSTM(lstm_config['hidden_dim'] // 2, dropout=lstm_config['dropout_rate']),
            Dense(64, activation='relu'),
            Dropout(lstm_config['dropout_rate']),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lstm_config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("LSTM model architecture:")
        model.summary()

        # Train model
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2)
        ]

        if X_val is not None:
            history = model.fit(
                X_train_pad, y_train,
                validation_data=(X_val_pad, y_val),
                epochs=20,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = model.fit(
                X_train_pad, y_train,
                epochs=20,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )

        self.models['lstm'] = model
        self.models['lstm_tokenizer'] = tokenizer

        return model, history

    def train_bert_model(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """Fine-tune BERT model for classification."""
        bert_config = self.config['models']['neural_networks']['bert']
        model_name = self.config['features']['embeddings']['model_name']

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        num_classes = len(np.unique(train_labels))
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )

        # Tokenize datasets
        def tokenize_function(batch):
            return tokenizer(
                batch['text'],
                padding=True,
                truncation=True,
                max_length=self.config['preprocessing']['max_sequence_length']
            )

        # Prepare datasets
        train_data = {'text': train_texts, 'labels': train_labels}
        train_dataset = Dataset.from_dict(train_data)
        train_dataset = train_dataset.map(tokenize_function, batched=True)

        if val_texts is not None and val_labels is not None:
            val_data = {'text': val_texts, 'labels': val_labels}
            val_dataset = Dataset.from_dict(val_data)
            val_dataset = val_dataset.map(tokenize_function, batched=True)
        else:
            val_dataset = None

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./models/bert/',
            num_train_epochs=bert_config['num_epochs'],
            per_device_train_batch_size=bert_config['batch_size'],
            per_device_eval_batch_size=bert_config['batch_size'],
            learning_rate=bert_config['learning_rate'],
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            logging_dir='./logs/bert/',
            logging_steps=10,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss",
        )

        # Define compute_metrics function
        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=1)
            return {"accuracy": (predictions == labels).mean()}

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # Train model
        logger.info("Fine-tuning BERT model...")
        trainer.train()

        self.models['bert'] = model
        self.models['bert_tokenizer'] = tokenizer

        return model, trainer

    def hyperparameter_tuning(self, model, param_grid, X_train, y_train):
        """Perform hyperparameter tuning using GridSearchCV."""
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def save_models(self, suffix=""):
        """Save trained models."""
        import os
        models_path = self.config['data']['processed_path'].replace('processed', 'models')

        for name, model in self.models.items():
            if 'tokenizer' in name:
                continue

            if hasattr(model, 'save'):
                # TensorFlow/Keras model
                os.makedirs(f"{models_path}{name}/", exist_ok=True)
                model.save(f"{models_path}{name}/model{suffix}.h5")
            elif hasattr(model, 'save_pretrained'):
                # Transformers model
                model.save_pretrained(f"{models_path}{name}{suffix}/")
            else:
                # Scikit-learn model
                joblib.dump(model, f"{models_path}{name}_model{suffix}.pkl")

        # Save tokenizers
        if 'lstm_tokenizer' in self.models:
            joblib.dump(self.models['lstm_tokenizer'], f"{models_path}lstm_tokenizer{suffix}.pkl")

        logger.info(f"Models saved to {models_path}")


# Example usage
if __name__ == "__main__":
    trainer = ModelTrainer()

    # Load features and labels
    X_train_tfidf, X_test_tfidf = trainer.load_features(['X_train_tfidf', 'X_test_tfidf'])
    train_df = pd.read_csv("data/processed/train_data.csv")
    y_train = train_df['label_encoded']

    # Train baseline models
    baseline_models = trainer.train_baseline_models(X_train_tfidf, y_train)

    # Save models
    trainer.save_models()