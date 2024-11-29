import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import joblib

class AIPlagiarismDetector:
    def __init__(self, model_path):
        # Set device to CPU for Raspberry Pi
        self.device = torch.device("cpu")
        
        # Load the tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        
        # Load the trained classifier
        self.classifier = joblib.load(model_path)

    def calculate_perplexity(self, text):
        # Calculate perplexity for the given text
        encodings = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(encodings, labels=encodings)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return perplexity.item()

    def detect_ai_text(self, text):
        # Calculate perplexity for the given text
        perplexity = self.calculate_perplexity(text)
        #print(f"Calculated Perplexity: {perplexity}")
        
        # Create a DataFrame for prediction with the same column name used during training
        prediction_input = pd.DataFrame([[perplexity]], columns=['perplexity'])
        
        # Use the classifier to make a prediction
        prediction = self.classifier.predict(prediction_input)
        
        # Return the result
        if prediction[0] == 1:
            return_arr = ["The text is likely AI-generated.", perplexity]
            return return_arr
        else:
            return_arr = ["The text is likely human-written.", perplexity]
            return return_arr
