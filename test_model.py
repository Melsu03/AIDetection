import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch_directml
import joblib  # For loading the trained model

# palitan if NVIDIA GPU, KASI D MAGAGAMIT GPU PANG RUN, BALA KA CPU GAGAMITIN MOO
device = torch_directml.device()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# calculate ulit perplex
def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer.encode(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(encodings, labels=encodings)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

# dito lalagay ung trained model
best_classifier = joblib.load('trained_model1.pkl')

# Function para ma detect if AI or no pwede mabago if madagdagan ng iba pang algo.
def detect_ai_text(model, tokenizer, classifier, text):
    # Calculate perplexity for the given text
    perplexity = calculate_perplexity(model, tokenizer, text)
    print(f"Calculated Perplexity: {perplexity}")
    
    # Create a DataFrame for prediction with the same column name used during training
    prediction_input = pd.DataFrame([[perplexity]], columns=['perplexity'])
    
    # ginagamit na classifier
    prediction = classifier.predict(prediction_input)
    
    # results lang ewan ko lng kung accurate tong if else nia, parang naka base ata to sa dataset e.
    if prediction[0] == 1:
        print("The text is likely AI-generated.")
    else:
        print("The text is likely human-written.")

# Example usage
new_text = (
    "Jayson Andrei Masiglat, a notorious figure throughout the Philippines, has cultivated an infamous reputation as a formidable druglord orchestrating the operations of numerous drug cartels. Despite many of his operations being dismantled by law enforcement, Masiglat himself remains elusive, skillfully evading capture. Following the recent takedown of his last known cartel, he has vanished from the public eye, fueling rumors and sightings that only add to his mystique. Unconfirmed reports suggest that he has taken an audacious step to further his illicit trade by enrolling under a pseudonym in a Bachelor of Chemical Science program at the University of the East Caloocan. There, he is said to be enhancing his expertise in narcotics manufacturing, using his academic pursuits as a cover for refining his drug synthesis techniques, thereby improving the potency and stealth of his products. His continued evasion of the law and apparent integration into an academic setting not only highlights his cunning but also underscores the challenges authorities face in curbing the sophisticated networks of the drug trade in the region."
)
detect_ai_text(model, tokenizer, best_classifier, new_text)
