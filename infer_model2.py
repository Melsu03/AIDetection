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
        self.model_name = "EleutherAI/gpt-neo-1.3B"
        self.bert_model_name = "bert-base-uncased"
        
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir="./transformers_cache")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir="./transformers_cache").to(self.device)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = BertModel.from_pretrained(self.bert_model_name).to(self.device)
        self.bert_model.eval()
        
        # Load the trained classifier
        self.best_classifier = joblib.load(model_path)
        
    def calculate_perplexity(self, text):
        encodings = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            outputs = self.model(encodings, labels=encodings)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return perplexity.item()

    def calculate_burstiness(self, text):
        words = text.split()
        word_counts = Counter(words)
        mean_frequency = np.mean(list(word_counts.values()))
        variance = np.var(list(word_counts.values()))
        burstiness = variance / mean_frequency if mean_frequency != 0 else 0
        return burstiness
    
    def calculate_bert_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        return sentence_embedding
    
    def extract_features_for_prediction(self, text):
        perplexity = self.calculate_perplexity(self.model, self.tokenizer, text)
        burstiness = self.calculate_burstiness(text)
        bert_embedding = self.calculate_bert_embedding(text)
        
        placeholder_feature = 0  # not sure dito nilagay lang ni chatgpt para ata d mag error???

        combined_features = np.hstack([[perplexity, burstiness, placeholder_feature], bert_embedding])
        return perplexity, burstiness, combined_features

    def detect_ai_text(self, text):
        return_arr = ["", "", 0, 0]
        perplexity, burstiness, features = self.extract_features_for_prediction(text)
        if features.shape[0] != 771:
            print(f"Error: Feature count mismatch. Expected 771, but got {features.shape[0]}.")
            return
        
        prediction_input = pd.DataFrame([features])
        prediction = self.best_classifier.predict(prediction_input)
        
        return_arr[2] = perplexity
        return_arr[3] = burstiness
        
        if prediction[0] == 1:
            return_arr[0] = "The text is likely AI-generated."
        else:
            return_arr[0] = "The text is likely human-written."
        
        if perplexity < 30 and burstiness > 1.5:
            return_arr[1] = "Low Perplexity and High Burstiness suggest the text is AI-generated."
        elif perplexity > 30 and burstiness < 1.5:
            return_arr[1] = "High Perplexity and Low Burstiness suggest the text is human-generated."
