import pandas as pd
import re
import ollama
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LLaMA3Classifier:
    """
    Uses LLaMA 3.2 via Ollama to classify emails as 'phishing' or 'clean' using prompt-based inference.
    """
    def __init__(self, model_name='llama3.2:latest'):
        self.model = model_name

    def build_prompt(self, email_text):
        return f"""You are a cybersecurity assistant. Classify the following email as either "phishing" or "clean":

\"\"\"{email_text}\"\"\"

Only reply with one word: phishing or clean."""

    def classify(self, email_text):
        prompt = self.build_prompt(email_text)
        try:
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            reply = response['message']['content'].strip().lower()

            # Extract 'phishing' or 'clean' using regex
            match = re.search(r'\b(phishing|clean)\b', reply)
            if match:
                return match.group(1)
            else:
                return 'unknown'
        except Exception as e:
            return 'unknown'


class MetricsEvaluator:
    """
    Computes and prints evaluation metrics: accuracy, precision, recall, F1 (macro and weighted).
    """
    def compute_print_f1(self, predict, targ, average):
        f1 = f1_score(targ, predict, average=average, zero_division=0)
        recall = recall_score(targ, predict, average=average, zero_division=0)
        precision = precision_score(targ, predict, average=average, zero_division=0)
        print("[%s] test_f1: %.5f \t test_precision: %.5f \t test_recall: %.5f" % (average.upper(), f1, precision, recall))

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        print("\n=== Evaluation Results ===")
        print("Accuracy: %.5f" % acc)
        self.compute_print_f1(y_pred, y_true, average='macro')
        self.compute_print_f1(y_pred, y_true, average='weighted')


class LLamaModel:
    """
    Loads CSV, truncates body to 1500 characters, classifies all emails with LLaMA 3.2,
    evaluates metrics, and exports predictions to CSV.
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.classifier = LLaMA3Classifier()
        self.evaluator = MetricsEvaluator()

    def load_data(self):
        df = pd.read_csv(self.dataset_path)
        df.columns = df.columns.str.strip().str.lower()

        if 'body' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'body' and 'label' columns.")

        df = df[['body', 'label']].dropna()
        df['body'] = df['body'].astype(str).str.slice(0, 1500)
        df.rename(columns={'body': 'text'}, inplace=True)

        # Normalize labels
        df['label'] = df['label'].str.lower().map({
            'phishing': 'phishing',
            'clean': 'clean'
        })

        df = df.dropna(subset=['label'])  # Drop rows with invalid labels
        return df

    def run(self):
        print("Loading dataset...")
        df = self.load_data()

        print(f"Classifying {len(df)} emails with LLaMA 3.2...")
        tqdm.pandas(desc="Classifying")
        df['predicted'] = df['text'].progress_apply(self.classifier.classify)

        # Map to numeric: clean = 0, phishing = 1
        label_map = {'clean': 0, 'phishing': 1}
        df['label_num'] = df['label'].map(label_map)
        df['predicted_num'] = df['predicted'].map(label_map)

        unknowns = df['predicted'].value_counts().get('unknown', 0)
        if unknowns > 0:
            print(f"{unknowns} predictions were 'unknown' and will be excluded from evaluation.")

        # Evaluate
        eval_df = df.dropna(subset=['predicted_num'])
        self.evaluator.evaluate(eval_df['label_num'].tolist(), eval_df['predicted_num'].tolist())

        # Export CSV
        output_path = self.dataset_path.replace('.csv', '_with_predictions_numeric.csv')
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    dataset_path = "../../data/subset_1000.csv" 
    pipeline = LLamaModel(dataset_path)
    pipeline.run() 
