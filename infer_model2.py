import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BertModel, BertTokenizer
import torch
import joblib
import numpy as np
from collections import Counter

class AIPlagiarismDetector:
    def __init__(self, model_path):
        # Set device to CPU for Raspberry Pi
        self.device = torch.device("cpu")
        
        # set model names
        self.gpt_model_name = "EleutherAI/gpt-neo-125M"
        self.bert_model_name = "bert-base-uncased"
        
        # Load the tokenizer and model
        self.gpt_tokenizer = AutoTokenizer.from_pretrained(
            self.gpt_model_name, cache_dir=".transformers_cache"
        )
        self.gpt_model = AutoModelForCausalLM.from_pretrained(
            self.gpt_model_name, cache_dir=".transformers_cache"
        ).to(self.device)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = BertModel.from_pretrained(self.bert_model_name).to(self.device)
        self.bert_model.eval()
        
        # Load the trained classifier
        self.classifier = joblib.load(model_path)
        
    def calculate_perplexity(self, text):
        print("Calculating perplexity...")
        encodings = self.gpt_tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)
        with torch.no_grad():
            outputs = self.gpt_model(encodings, labels=encodings)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        print(f"Perplexity: {perplexity.item()}")
        return perplexity.item()

    def calculate_burstiness(self, text):
        print("Calculating burstiness...")
        words = text.split()
        word_counts = Counter(words)
        mean_frequency = np.mean(list(word_counts.values()))
        variance = np.var(list(word_counts.values()))
        burstiness = variance / mean_frequency if mean_frequency != 0 else 0
        print(f"Burstiness: {burstiness}")
        return burstiness
    
    def calculate_bert_embedding(self, text):
        print("Calculating BERT embedding...")
        inputs = self.bert_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        print("BERT embedding calculated.")
        return sentence_embedding
    
    def extract_features_for_prediction(self, text):
        perplexity = self.calculate_perplexity(text)
        burstiness = self.calculate_burstiness(text)
        bert_embedding = self.calculate_bert_embedding(text)

        # Placeholder feature
        placeholder_feature = 0  # Adjust if additional features are needed

        # Combine features
        combined_features = np.hstack(
            [[perplexity, burstiness, placeholder_feature], bert_embedding]
        )
        return perplexity, burstiness, combined_features

    def detect_ai_text(self, text):
        print("Starting detection process...")
        perplexity, burstiness, features = self.extract_features_for_prediction(text)

        # Check if features match classifier's expectation
        if features.shape[0] != 771:
            print(
                f"Error: Feature count mismatch. Expected 771, got {features.shape[0]}."
            )
            return ["Feature count mismatch", perplexity, burstiness]

        prediction_input = pd.DataFrame([features])
        prediction = self.classifier.predict(prediction_input)

        # Interpretation based on classifier result
        if prediction[0] == 1:
            result = "The text is likely AI-generated."
        else:
            result = "The text is likely human-written."

        interpretation = None
        if perplexity < 30 and burstiness > 1.5:
            interpretation = (
                "Low Perplexity and High Burstiness suggest the text is AI-generated."
            )
        elif perplexity > 30 and burstiness < 1.5:
            interpretation = (
                "High Perplexity and Low Burstiness suggest the text is human-written."
            )

        print(f"Result: {result}")
        print(f"Perplexity: {perplexity}")
        print(f"Burstiness: {burstiness}")
        if interpretation:
            print(f"Interpretation: {interpretation}")

        return [result, perplexity, burstiness, interpretation]
