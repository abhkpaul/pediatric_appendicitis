# Only change the imports at the top to remove deep learning dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json
import yaml
import logging
import joblib

# Rest of the code remains the same...
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize OpenAI client for ChatGPT-4 comparison
        self.openai_client = None
        if self.config['api']['openai'].get('api_key'):
            self.openai_client = OpenAI(api_key=self.config['api']['openai']['api_key'])

    def load_test_data(self):
        """Load test data and gold standard labels."""
        test_df = pd.read_csv("data/processed/test_data.csv")

        # Load manual extraction results if available
        try:
            manual_results = pd.read_csv("data/raw/adjudicated_gold_standard_labels.csv")
        except FileNotFoundError:
            manual_results = None
            logger.warning("Manual extraction results not found")

        return test_df, manual_results

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Comprehensive model evaluation."""
        # Predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
        elif hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
            y_pred_proba = None
        else:
            # For neural networks
            y_pred = np.argmax(model.predict(X_test), axis=1)
            y_pred_proba = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Multi-class AUC
        if y_pred_proba is not None and len(np.unique(y_test)) > 2:
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                auc = None
        else:
            auc = None

        # Detailed report
        report = classification_report(y_test, y_pred, output_dict=True)

        results = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'classification_report': report,
            'predictions': y_pred
        }

        logger.info(f"\n=== {model_name} Evaluation ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        if auc:
            logger.info(f"AUC-ROC: {auc:.4f}")

        return results

    def evaluate_manual_extraction(self, manual_df, gold_standard_col='surgeon_adjudicated_grade',
                                   manual_col='manually_extracted_grade'):
        """Evaluate manual data extraction performance."""
        if manual_df is None:
            return None

        y_true = manual_df[gold_standard_col]
        y_pred = manual_df[manual_col]

        # Encode if needed
        if y_true.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_true_encoded = le.fit_transform(y_true)
            y_pred_encoded = le.transform(y_pred)
        else:
            y_true_encoded = y_true
            y_pred_encoded = y_pred

        accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
        precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted')
        recall = recall_score(y_true_encoded, y_pred_encoded, average='weighted')
        f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted')

        results = {
            'method': 'Manual Extraction',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred_encoded
        }

        logger.info(f"\n=== Manual Extraction Evaluation ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")

        return results

    def query_chatgpt4(self, text, prompt_template=None):
        """Query ChatGPT-4 for appendicitis grading."""
        if self.openai_client is None:
            logger.warning("OpenAI client not initialized")
            return None

        if prompt_template is None:
            prompt_template = """
            Analyze the following clinical report and classify the appendicitis severity into one of these categories:
            - Normal
            - Simple/Uncomplicated
            - Gangrenous
            - Perforated

            Report: {text}

            Provide only the classification category as your response.
            """

        prompt = prompt_template.format(text=text)

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config['api']['openai']['model'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config['api']['openai']['max_tokens'],
                temperature=self.config['api']['openai']['temperature']
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error querying ChatGPT-4: {e}")
            return None

    def evaluate_chatgpt4(self, test_texts, test_labels, sample_size=50):
        """Evaluate ChatGPT-4 performance on a sample of test data."""
        if self.openai_client is None:
            logger.warning("Skipping ChatGPT-4 evaluation - API key not configured")
            return None

        # Sample for evaluation (to manage API costs)
        if sample_size < len(test_texts):
            indices = np.random.choice(len(test_texts), sample_size, replace=False)
            sample_texts = [test_texts[i] for i in indices]
            sample_labels = [test_labels[i] for i in indices]
        else:
            sample_texts = test_texts
            sample_labels = test_labels

        logger.info(f"Evaluating ChatGPT-4 on {len(sample_texts)} samples...")

        predictions = []
        for i, text in enumerate(sample_texts):
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(sample_texts)} samples")

            prediction = self.query_chatgpt4(text)
            predictions.append(prediction)

        # Map predictions to encoded labels
        label_encoder = joblib.load("data/processed/label_encoder.pkl")

        # Clean and map predictions
        cleaned_predictions = []
        for pred in predictions:
            pred_lower = pred.lower()
            if 'normal' in pred_lower:
                cleaned_predictions.append('Normal')
            elif 'simple' in pred_lower or 'uncomplicated' in pred_lower:
                cleaned_predictions.append('Simple/Uncomplicated')
            elif 'gangrenous' in pred_lower:
                cleaned_predictions.append('Gangrenous')
            elif 'perforated' in pred_lower:
                cleaned_predictions.append('Perforated')
            else:
                cleaned_predictions.append('Unknown')

        # Encode for comparison
        try:
            y_true_encoded = label_encoder.transform(sample_labels)
            y_pred_encoded = label_encoder.transform(cleaned_predictions)
        except:
            logger.warning("Could not encode all ChatGPT-4 predictions")
            return None

        accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
        precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
        recall = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
        f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)

        results = {
            'method': 'ChatGPT-4',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred_encoded,
            'sample_size': len(sample_texts)
        }

        logger.info(f"\n=== ChatGPT-4 Evaluation ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"Sample Size: {len(sample_texts)}")

        return results

    def create_comparison_table(self, results_list):
        """Create a comparison table of all evaluation results."""
        comparison_data = []

        for results in results_list:
            if results is None:
                continue

            row = {
                'Method': results.get('method', 'Unknown'),
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
            }

            if 'auc' in results and results['auc'] is not None:
                row['AUC-ROC'] = f"{results['auc']:.4f}"
            else:
                row['AUC-ROC'] = 'N/A'

            if 'sample_size' in results:
                row['Sample Size'] = results['sample_size']
            else:
                row['Sample Size'] = 'Full Test Set'

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df

    def plot_confusion_matrices(self, results_list, label_encoder):
        """Plot confusion matrices for all methods."""
        n_methods = len(results_list)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))

        if n_methods == 1:
            axes = [axes]

        class_names = label_encoder.classes_

        for i, results in enumerate(results_list):
            if results is None:
                continue

            cm = confusion_matrix(results.get('true_labels', []), results['predictions'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names, ax=axes[i])
            axes[i].set_title(f"{results['method']} Confusion Matrix")
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_evaluation_results(self, results_list, comparison_df):
        """Save all evaluation results to files."""
        import os
        os.makedirs('results', exist_ok=True)

        # Save individual results
        for i, results in enumerate(results_list):
            if results is None:
                continue

            method_name = results['method'].replace(' ', '_').lower()
            with open(f'results/{method_name}_results.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = results.copy()
                if 'predictions' in json_results:
                    json_results['predictions'] = json_results['predictions'].tolist()
                json.dump(json_results, f, indent=2)

        # Save comparison table
        comparison_df.to_csv('results/method_comparison.csv', index=False)
        comparison_df.to_latex('results/method_comparison.tex', index=False)

        logger.info("Evaluation results saved to 'results/' directory")


# Example usage
if __name__ == "__main__":
    evaluator = ModelEvaluator()

    # Load test data
    test_df, manual_results = evaluator.load_test_data()

    # Load trained models
    rf_model = joblib.load("models/random_forest_model.pkl")
    X_test_tfidf = np.load("data/features/X_test_tfidf.npy")

    # Evaluate models
    rf_results = evaluator.evaluate_model(rf_model, X_test_tfidf, test_df['label_encoded'], "Random Forest")

    # Evaluate manual extraction
    manual_results = evaluator.evaluate_manual_extraction(manual_results)

    # Evaluate ChatGPT-4
    chatgpt_results = evaluator.evaluate_chatgpt4(
        test_df['processed_text'].tolist(),
        test_df['severity_grade'].tolist(),
        sample_size=50
    )

    # Create comparison
    all_results = [rf_results, manual_results, chatgpt_results]
    comparison_df = evaluator.create_comparison_table(all_results)

    print("\n=== FINAL COMPARISON ===")
    print(comparison_df.to_string(index=False))

    # Save results
    evaluator.save_evaluation_results(all_results, comparison_df)