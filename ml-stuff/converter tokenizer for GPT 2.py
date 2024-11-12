import pandas as pd
from transformers import GPT2Tokenizer

# Load the dataset
reduced_df = pd.read_csv('Reduced_1000_AI_and_1000_Human_Texts_Dataset.csv')
reduced_df = reduced_df[reduced_df['text'].notna()]

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to truncate text to ensure it has < 1,024 tokens
def truncate_text_to_limit(text, max_tokens=1024):
    tokens = tokenizer.encode(text, max_length=max_tokens, truncation=True)
    return tokenizer.decode(tokens)

# Apply truncation to the 'text' column
reduced_df['text'] = reduced_df['text'].apply(truncate_text_to_limit)

# Save the reduced dataset
reduced_df.to_csv('Reduced_Data_Dataset.csv', index=False)