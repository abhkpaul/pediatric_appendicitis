#!/usr/bin/env python3
"""
Main pipeline script for Paediatric Appendicitis AI Project.
Run the entire pipeline or specific stages.
"""

import argparse
import yaml
import logging
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """Run the complete ML pipeline."""
    logger.info("Starting full pipeline execution...")

    # 1. Data Preprocessing
    logger.info("=== STAGE 1: Data Preprocessing ===")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data("merged")

    if df is not None:
        df = preprocessor.prepare_dataset(df, 'report_text', 'severity_grade')
        train_df, val_df, test_df = preprocessor.split_data(df)
        preprocessor.save_processed_data(train_df, val_df, test_df)

    # 2. Feature Engineering
    logger.info("=== STAGE 2: Feature Engineering ===")
    feature_engineer = FeatureEngineer()

    # Load processed data
    train_df = preprocessor.load_data("train")
    test_df = preprocessor.load_data("test")

    if train_df is not None and test_df is not None:
        # TF-IDF features
        X_train_tfidf, X_test_tfidf = feature_engineer.extract_tfidf_features(
            train_df['processed_text'],
            test_df['processed_text']
        )

        # BERT embeddings (sample for demonstration)
        sample_texts = train_df['processed_text'].tolist()[:100]
        bert_embeddings = feature_engineer.get_bert_embeddings(sample_texts)

        feature_engineer.save_features(
            [X_train_tfidf, X_test_tfidf, bert_embeddings],
            ['X_train_tfidf', 'X_test_tfidf', 'bert_embeddings_sample']
        )

    # 3. Model Training
    logger.info("=== STAGE 3: Model Training ===")
    trainer = ModelTrainer()

    # Load features
    X_train_tfidf, X_test_tfidf = trainer.load_features(['X_train_tfidf', 'X_test_tfidf'])

    if X_train_tfidf is not None:
        # Train baseline models
        baseline_models = trainer.train_baseline_models(X_train_tfidf, train_df['label_encoded'])
        trainer.save_models()

    # 4. Evaluation
    logger.info("=== STAGE 4: Model Evaluation ===")
    evaluator = ModelEvaluator()

    # Load test data
    test_df, manual_results = evaluator.load_test_data()

    if test_df is not None and 'random_forest_model.pkl' in trainer.models:
        # Evaluate Random Forest
        rf_results = evaluator.evaluate_model(
            trainer.models['random_forest_model.pkl'],
            X_test_tfidf,
            test_df['label_encoded'],
            "Random Forest"
        )

        # Evaluate manual extraction
        manual_results = evaluator.evaluate_manual_extraction(manual_results)

        # Create comparison
        all_results = [rf_results, manual_results]
        comparison_df = evaluator.create_comparison_table(all_results)

        print("\n=== PIPELINE COMPLETED ===")
        print(comparison_df.to_string(index=False))
