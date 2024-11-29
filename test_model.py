import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import joblib  # For loading the trained model

# Set device to CPU for Raspberry Pi
device = torch.device("cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer.encode(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(encodings, labels=encodings)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

# Load the trained classifier model
best_classifier = joblib.load('model/trained_model1.pkl')

# Function to detect if text is AI-generated
def detect_ai_text(model, tokenizer, classifier, text):
    # Calculate perplexity for the given text
    perplexity = calculate_perplexity(model, tokenizer, text)
    print(f"Calculated Perplexity: {perplexity}")
    
    # Create a DataFrame for prediction
    prediction_input = pd.DataFrame([[perplexity]], columns=['perplexity'])
    
    # Use classifier to predict
    prediction = classifier.predict(prediction_input)
    
    # Output the result based on prediction
    if prediction[0] == 1:
        print("The text is likely AI-generated.")
    else:
        print("The text is likely human-written.")

# Example usage
new_text = (
    "Mental health is crucial to an individual's well-being, affecting nearly every area of life relationships, work, and even physical health. Despite this, mental health has long been overlooked and stigmatized, leading to silence, misunderstandings, and insufficient support. The importance of mental health awareness can't be overstated: it fosters empathy, encourages people to seek help, and drives positive change in society.\nRaising awareness doesn't just support those with mental health issues; it promotes a culture where mental well-being is prioritized on par with physical health. By breaking down stigmas, we can create a society that values psychological health, ensuring people feel supported, understood, and empowered to seek the help they need."
)
detect_ai_text(model, tokenizer, best_classifier, new_text)
