import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BertModel, BertTokenizer
import torch
import joblib
import numpy as np
from collections import Counter
import re
import os

class AIPlagiarismDetector:
    def __init__(self, model_path1, model_path2=None):
        # Set device to CPU for Raspberry Pi
        self.device = torch.device("cpu")
        
        # Use smaller models suitable for Raspberry Pi
        self.model_name = "EleutherAI/gpt-neo-125M"  # Smaller model than the 1.3B version
        self.bert_model_name = "bert-base-uncased"
        
        print("Loading tokenizer and models...")
        # Set cache directory to a local path
        cache_dir = os.path.join(os.getcwd(), ".transformers_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, cache_dir=cache_dir
        ).to(self.device)
        
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            self.bert_model_name, cache_dir=cache_dir
        )
        self.bert_model = BertModel.from_pretrained(
            self.bert_model_name, cache_dir=cache_dir
        ).to(self.device)
        self.bert_model.eval()
        print("Models loaded successfully.")

        # Load the trained classifiers
        print("Loading the trained classifiers...")
        self.best_classifier1 = joblib.load(model_path1)
        self.best_classifier2 = joblib.load(model_path2) if model_path2 else None
        print("Trained classifiers loaded successfully.")

    def get_confidence_label(self, prob):
        if prob < 70:
            return "Moderately confident"
        elif prob < 90:
            return "Most likely confident"
        else:
            return "Highly confident"

    def calculate_perplexity(self, text):
        # For longer texts, process in chunks to avoid memory issues
        max_length = 512
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        perplexities = []
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors='pt', truncation=True, max_length=max_length)
            input_ids = inputs.input_ids.to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss)
                perplexities.append(perplexity.item())
        
        # Return average perplexity across chunks
        return sum(perplexities) / len(perplexities) if perplexities else 0

    def calculate_burstiness(self, text):
        words = text.split()
        word_counts = Counter(words)
        mean_frequency = np.mean(list(word_counts.values()))
        variance = np.var(list(word_counts.values()))
        burstiness = variance / mean_frequency if mean_frequency != 0 else 0
        return burstiness

    def calculate_bert_embedding(self, text):
        # For longer texts, process in chunks
        max_length = 512
        if len(text.split()) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            all_embeddings = []
            
            for chunk in chunks:
                inputs = self.bert_tokenizer(
                    chunk, return_tensors='pt', padding=True, truncation=True, max_length=512
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                chunk_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
                all_embeddings.append(chunk_embedding)
            
            # Average all chunk embeddings
            sentence_embedding = np.mean(all_embeddings, axis=0)
        else:
            # For shorter texts, process as before
            inputs = self.bert_tokenizer(
                text, return_tensors='pt', padding=True, truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        
        return sentence_embedding

    def extract_features_for_prediction(self, text):
        perplexity = self.calculate_perplexity(text)
        burstiness = self.calculate_burstiness(text)
        bert_embedding = self.calculate_bert_embedding(text)
        
        placeholder_feature = 0  # Placeholder for future features
        combined_features = np.hstack([[perplexity, burstiness, placeholder_feature], bert_embedding])
        return perplexity, burstiness, combined_features

    def split_into_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences

    def get_combined_probabilities(self, features):
        prob1 = self.best_classifier1.predict_proba([features])[0]
        
        if self.best_classifier2:
            prob2 = self.best_classifier2.predict_proba([features])[0]
            # Weighted average
            avg_prob_ai = prob1[1] * 0.75 + prob2[1] * 0.25
            avg_prob_human = prob1[0] * 0.75 + prob2[0] * 0.25
        else:
            # If only one classifier is available
            avg_prob_ai = prob1[1]
            avg_prob_human = prob1[0]
        
        return avg_prob_ai * 100, avg_prob_human * 100

    def detect_ai_text(self, text):
        print("\nStarting detection process...\n")
        
        # For multi-page documents, add some preprocessing
        if "--- Page" in text:
            print("Multi-page document detected")
            # Count pages
            page_count = len(re.findall(r'--- Page \d+ ---', text))
            print(f"Document contains approximately {page_count} pages")
        
        # Step 1: Analyze the WHOLE text
        overall_perplexity, overall_burstiness, overall_features = self.extract_features_for_prediction(text)

        if overall_features.shape[0] != 771:
            print(f"Error: Feature count mismatch. Expected 771, got {overall_features.shape[0]}.")
            return ["Feature count mismatch", overall_perplexity, overall_burstiness, "Error in analysis"]

        # Get combined probabilities
        ai_prob, human_prob = self.get_combined_probabilities(overall_features)

        # Apply the threshold of 60% for AI classification
        if ai_prob > 60:
            classification = "AI-Generated"
            confidence_label = self.get_confidence_label(ai_prob)
        else:
            classification = "Human-Written"
            confidence_label = self.get_confidence_label(human_prob)

        result_message = f"{classification} ({confidence_label})"
        interpretation = f"AI Probability: {ai_prob:.2f}% | Human Probability: {human_prob:.2f}%"
        
        print(f"Overall Classification: {result_message}")
        print(f"{interpretation}")
        print(f"Overall Perplexity: {overall_perplexity:.2f} | Overall Burstiness: {overall_burstiness:.2f}\n")

        # Return the results in a format compatible with the existing system
        return [result_message, overall_perplexity, overall_burstiness, interpretation]