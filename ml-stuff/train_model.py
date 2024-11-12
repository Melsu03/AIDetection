import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch_directml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib  

# Set device to use DirectML for AMD GPUs "pag NVIDIA GPU MO PATI IMPORT TORCH_DIRECTML IIBAHIN NG CUDA CHURURUT"
device = torch_directml.device()
# GPT 2 UNG PRE-TRAINED MODEL, NOT SURE IF KAYA NATIN GUMAWA NG MODEL NA GANUN KALAKI.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Function para ma calculate perplexity
def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer.encode(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(encodings, labels=encodings)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()
# loading ng data set
dataset = pd.read_csv('Reduced_Data_Dataset.csv')
dataset = dataset[dataset['text'].notna()]
dataset['perplexity'] = dataset['text'].apply(lambda x: calculate_perplexity(model, tokenizer, x))

print("Perplexity calculation complete.")

# Split data into training and testing sets
X = dataset[['perplexity']]  
y = dataset['generated']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# randomforestclassifier chuchu tas ung grid is hyperparameter tuning galing GPT ung mga grid nayan ok daw ung results pag ginamit ynag mga yan e, 
classifier = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# cross validation kineme tas results neto dun na sa line 56
grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1  #para mapabilis lang ung training 
)

grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Save the best model to a file
best_classifier = grid_search.best_estimator_
joblib.dump(best_classifier, 'trained_model1.pkl') #eto ung pinaka model na trained yang pkl na yan.

print("Model training and saving complete.")


