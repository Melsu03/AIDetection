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
        # Handle empty or very short text
        if not text or len(text.strip()) < 5:
            print("Text too short for perplexity calculation")
            return 100.0  # Return a default high perplexity value
        
        try:
            # For very long texts (like concatenated pages), we'll process in chunks
            # to avoid memory issues and get a more representative perplexity
            if len(text) > 4000:
                print("Long text detected, processing in chunks...")
                chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
                perplexities = []
                
                for i, chunk in enumerate(chunks[:5]):  # Process up to 5 chunks (10K chars)
                    print(f"Processing chunk {i+1}/{min(len(chunks), 5)}...")
                    encodings = self.gpt_tokenizer.encode(
                        chunk, return_tensors="pt", truncation=True, max_length=1024
                    ).to(self.device)
                    
                    # Ensure encodings are of the correct type
                    if encodings.dtype != torch.long:
                        encodings = encodings.long()
                    
                    with torch.no_grad():
                        outputs = self.gpt_model(encodings, labels=encodings)
                        loss = outputs.loss
                        chunk_perplexity = torch.exp(loss).item()
                        perplexities.append(chunk_perplexity)
                
                # Use the average perplexity of all chunks
                perplexity = sum(perplexities) / len(perplexities)
                print(f"Average perplexity across chunks: {perplexity}")
                return perplexity
            else:
                # For shorter texts, process as before
                encodings = self.gpt_tokenizer.encode(
                    text, return_tensors="pt", truncation=True, max_length=2048
                ).to(self.device)
                
                # Ensure encodings are of the correct type
                if encodings.dtype != torch.long:
                    encodings = encodings.long()
                
                with torch.no_grad():
                    outputs = self.gpt_model(encodings, labels=encodings)
                    loss = outputs.loss
                    perplexity = torch.exp(loss)
                print(f"Perplexity: {perplexity.item()}")
                return perplexity.item()
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return 100.0  # Return a default high perplexity value

    def calculate_burstiness(self, text):
        print("Calculating burstiness...")
        # For multi-page documents, we want to analyze the text as a whole
        # but ignore page markers
        
        # Remove page markers if present (e.g., "--- Page 1 ---")
        cleaned_text = text
        if "--- Page" in text:
            import re
            cleaned_text = re.sub(r'--- Page \d+ ---', '', text)
        
        words = cleaned_text.split()
        
        # Skip calculation for very short texts
        if len(words) < 10:
            print("Text too short for meaningful burstiness calculation")
            return 0.0
        
        word_counts = Counter(words)
        mean_frequency = np.mean(list(word_counts.values()))
        variance = np.var(list(word_counts.values()))
        burstiness = variance / mean_frequency if mean_frequency != 0 else 0
        print(f"Burstiness: {burstiness}")
        return burstiness
    
    def calculate_bert_embedding(self, text):
        print("Calculating BERT embedding...")
        
        # For very long texts (like concatenated pages), we'll process in chunks
        # and average the embeddings
        if len(text) > 1000:
            print("Long text detected, processing in chunks for BERT embedding...")
            # Split into chunks of approximately 500 characters
            chunks = [text[i:i+500] for i in range(0, min(len(text), 5000), 500)]
            all_embeddings = []
            
            for i, chunk in enumerate(chunks):
                print(f"Processing BERT chunk {i+1}/{len(chunks)}...")
                inputs = self.bert_tokenizer(
                    chunk, return_tensors="pt", padding=True, truncation=True, max_length=512
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
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        
        print("BERT embedding calculated.")
        return sentence_embedding
    
    def extract_features_for_prediction(self, text):
        try:
            # Handle empty text
            if not text or len(text.strip()) < 5:
                print("Text too short for feature extraction")
                # Return default values
                return 100.0, 0.0, np.zeros(771)
            
            perplexity = self.calculate_perplexity(text)
            burstiness = self.calculate_burstiness(text)
            bert_embedding = self.calculate_bert_embedding(text)

            # Placeholder feature
            placeholder_feature = 0  # Adjust if additional features are needed

            # Combine features
            combined_features = np.hstack(
                [[perplexity, burstiness, placeholder_feature], bert_embedding]
            )
            
            # Ensure the feature vector has the expected length
            if combined_features.shape[0] != 771:
                print(f"Warning: Feature count mismatch. Expected 771, got {combined_features.shape[0]}.")
                # Pad or truncate to match expected size
                if combined_features.shape[0] < 771:
                    combined_features = np.pad(combined_features, (0, 771 - combined_features.shape[0]))
                else:
                    combined_features = combined_features[:771]
            
            return perplexity, burstiness, combined_features
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Return default values
            return 100.0, 0.0, np.zeros(771)

    def detect_ai_text(self, text):
        print("Starting detection process...")
        
        # For multi-page documents, add some preprocessing
        if "--- Page" in text:
            print("Multi-page document detected")
            # Count pages
            import re
            page_count = len(re.findall(r'--- Page \d+ ---', text))
            print(f"Document contains approximately {page_count} pages")
        
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

        # Provide more detailed interpretation for multi-page documents
        interpretation = None
        if "--- Page" in text:
            if perplexity < 30 and burstiness > 1.5:
                interpretation = (
                    "Low Perplexity and High Burstiness across multiple pages suggest the document is AI-generated."
                )
            elif perplexity > 30 and burstiness < 1.5:
                interpretation = (
                    "High Perplexity and Low Burstiness across multiple pages suggest the document is human-written."
                )
            else:
                interpretation = (
                    f"Document analysis: Perplexity={perplexity:.2f}, Burstiness={burstiness:.2f}. "
                    f"The document shows mixed characteristics."
                )
        else:
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
