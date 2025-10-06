#!/usr/bin/env python3
"""
Main pipeline script for Paediatric Appendicitis AI Project (CPU-only, single-threaded version).
"""

import os

# Set CPU-only and single-threaded environment variables
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import yaml
import logging
import sys
import pandas as pd
import numpy as np

# Add the src directory to Python path
sys.path.append('src')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """Run the complete CPU-only, single-threaded ML pipeline."""
    logger.info("Starting CPU-only, single-threaded pipeline execution...")

    try:
        from src.data_preprocessing import DataPreprocessor
        from src.feature_engineering import FeatureEngineer
        from src.model_training import ModelTrainer
        from src.evaluation import ModelEvaluator

        # 1. Data Preprocessing
        logger.info("=== STAGE 1: Data Preprocessing ===")
        preprocessor = DataPreprocessor()

        # Create sample data if it doesn't exist
        import os
        sample_data_path = "data/raw/merged_data.csv"
        if not os.path.exists(sample_data_path):
            logger.info("Creating sample data structure...")
            create_sample_data()

        df = preprocessor.load_data("merged")

        if df is not None:
            df = preprocessor.prepare_dataset(df, 'report_text', 'severity_grade')
            train_df, val_df, test_df = preprocessor.split_data(df)
            preprocessor.save_processed_data(train_df, val_df, test_df)
            logger.info("✅ Preprocessing stage completed successfully!")
        else:
            logger.warning("No data found. Please add your dataset to data/raw/")
            return

        # 2. Feature Engineering
        logger.info("=== STAGE 2: Feature Engineering ===")
        feature_engineer = FeatureEngineer()

        # Load processed data
        train_df = pd.read_csv("data/processed/train_data.csv")
        test_df = pd.read_csv("data/processed/test_data.csv")

        if train_df is not None and test_df is not None:
            # Extract multiple feature types
            X_train_tfidf, X_test_tfidf = feature_engineer.extract_tfidf_features(
                train_df['processed_text'],
                test_df['processed_text']
            )

            # Extract keyword features
            keyword_vocab = feature_engineer.build_keyword_vocabulary(train_df['medical_keywords'])
            X_train_keywords = feature_engineer.create_keyword_features(train_df['medical_keywords'], keyword_vocab)
            X_test_keywords = feature_engineer.create_keyword_features(test_df['medical_keywords'], keyword_vocab)

            # Extract text statistics
            X_train_stats = feature_engineer.create_text_statistics(train_df['processed_text'])
            X_test_stats = feature_engineer.create_text_statistics(test_df['processed_text'])

            # Combine all features
            X_train_combined = feature_engineer.combine_features([X_train_tfidf, X_train_keywords, X_train_stats])
            X_test_combined = feature_engineer.combine_features([X_test_tfidf, X_test_keywords, X_test_stats])

            feature_engineer.save_features(
                [X_train_combined, X_test_combined],
                ['X_train_combined', 'X_test_combined']
            )
            logger.info("✅ Feature engineering stage completed successfully!")

        # 3. Model Training
        logger.info("=== STAGE 3: Model Training ===")
        trainer = ModelTrainer()

        # Load features
        X_train = feature_engineer.load_features(['X_train_combined'])[0]

        if X_train is not None:
            # Train all models
            models = trainer.train_baseline_models(X_train, train_df['label_encoded'])

            # Train ensemble
            ensemble = trainer.train_ensemble(X_train, train_df['label_encoded'])

            trainer.save_models()
            logger.info("✅ Training stage completed successfully!")

        # 4. Evaluation
        logger.info("=== STAGE 4: Model Evaluation ===")
        evaluator = ModelEvaluator()

        # Load test data
        test_df = pd.read_csv("data/processed/test_data.csv")
        X_test = feature_engineer.load_features(['X_test_combined'])[0]

        if test_df is not None and X_test is not None:
            # Load the best model
            try:
                best_model = joblib.load("models/best_model.pkl")
            except:
                # Use random forest as default
                best_model = joblib.load("models/random_forest_model.pkl")

            # Evaluate model
            results = evaluator.evaluate_model(
                best_model,
                X_test,
                test_df['label_encoded'],
                "Best Model"
            )

            # Create results summary
            comparison_data = [{
                'Method': 'Best Model',
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
            }]

            comparison_df = pd.DataFrame(comparison_data)

            print("\n=== FINAL RESULTS ===")
            print(comparison_df.to_string(index=False))

            # Save results
            import os
            os.makedirs('results', exist_ok=True)
            comparison_df.to_csv('results/final_results.csv', index=False)

            logger.info("✅ Evaluation stage completed successfully!")

        logger.info("🎉 CPU-only, single-threaded pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()


def create_sample_data():
    """Create sample data for testing the pipeline."""
    import pandas as pd
    import os

    os.makedirs("data/raw", exist_ok=True)

    # Create more comprehensive sample dataset
    sample_data = {
        'report_text': [
            "Ultrasound shows inflamed appendix measuring 8mm with surrounding fluid. Findings consistent with acute appendicitis.",
            "Operative findings: Gangrenous appendix with localized perforation. Purulent fluid in pelvis.",
            "US: Normal appendix visualized measuring 4mm. No evidence of appendicitis.",
            "OR: Simple acute appendicitis. Appendix erythematous and swollen but no gangrene or perforation.",
            "Ultrasound: Appendix not clearly visualized. Non-specific inflammatory changes in RLQ.",
            "Operative report: Perforated appendix with abscess formation. Extensive contamination.",
            "US: Appendix diameter 7mm with wall thickening and hyperemia. Suggestive of early appendicitis.",
            "OR: Gangrenous changes of appendix without perforation. Minimal free fluid.",
            "Ultrasound: Unremarkable appendix. No signs of inflammation.",
            "Operative findings: Simple appendicitis with fibrinous exudate.",
        ],
        'severity_grade': [
            'Simple/Uncomplicated',
            'Perforated',
            'Normal',
            'Simple/Uncomplicated',
            'Simple/Uncomplicated',
            'Perforated',
            'Simple/Uncomplicated',
            'Gangrenous',
            'Normal',
            'Simple/Uncomplicated'
        ],
        'manually_extracted_grade': [
            'Simple/Uncomplicated',
            'Perforated',
            'Normal',
            'Simple/Uncomplicated',
            'Simple/Uncomplicated',
            'Perforated',
            'Simple/Uncomplicated',
            'Gangrenous',
            'Normal',
            'Simple/Uncomplicated'
        ],
        'surgeon_adjudicated_grade': [
            'Simple/Uncomplicated',
            'Perforated',
            'Normal',
            'Simple/Uncomplicated',
            'Simple/Uncomplicated',
            'Perforated',
            'Simple/Uncomplicated',
            'Gangrenous',
            'Normal',
            'Simple/Uncomplicated'
        ]
    }

    df = pd.DataFrame(sample_data)
    df.to_csv("data/raw/merged_data.csv", index=False)
    logger.info("Sample data created at data/raw/merged_data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paediatric Appendicitis AI Pipeline (CPU-only, single-threaded)')
    parser.add_argument('--stage', type=str,
                        choices=['all', 'preprocessing', 'features', 'training', 'evaluation', 'create_sample'],
                        default='all', help='Pipeline stage to run')

    args = parser.parse_args()

    if args.stage == 'create_sample':
        create_sample_data()
    elif args.stage == 'all':
        run_full_pipeline()
    else:
        logger.info(f"Individual stage execution not implemented in CPU-only version. Use 'all' to run full pipeline.")