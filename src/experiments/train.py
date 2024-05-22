
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5"
print("Limpando cache")
torch.cuda.empty_cache()

import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset

import random
import inflect
import re

# Smaller than the original GPT-2, only 6 layers (instead of 12), 12 heads, 768 dimensions (like GPT-2)
model_name = "gpt2-medium" # "gpt2-medium" # distilgpt2 #gpt2-large it reachs the maximum gpu capacity
experiment_name = "olist"

path_to_train = f"/hadatasets/fillipe.silva/LLMSegm/data/{experiment_name}/train_rfm.csv"
print("reading the dataset", path_to_train)
df = pd.read_csv(path_to_train)
columns = df.columns.tolist()
ds = Dataset.from_pandas(df)

def combine_data_ordered(sample):
    concat = ""
    for col in columns:
        concat += "%s is %s, " % (col, str(sample[col]).strip())

    return {"concat": concat}

def combine_data_shuffled(sample):
    concat = ""
    for col in random.sample(columns, k=len(columns)):
        concat += "%s is %s, " % (col, str(sample[col]).strip())

    return {"concat": concat}

def convert_numbers_to_words(sentence):
    def convert(match):
        number = match.group(0)
        if '.' in number:
            # Handle floating-point numbers
            integer_part, fractional_part = number.split('.')
            integer_word = p.number_to_words(integer_part)
            fractional_word = p.number_to_words(fractional_part)
            return f"{integer_word} point {fractional_word}"
        else:
            # Handle integers
            return p.number_to_words(number)
    
    # Regular expression to find numbers
    pattern = re.compile(r'\d+(\.\d+)?')
    converted_sentence = pattern.sub(convert, sentence)
    return converted_sentence

# Shuffle the features or not
p = inflect.engine()
shuffle = True 
if shuffle:
    combined_ds = ds.map(combine_data_shuffled)
else:
    combined_ds = ds.map(combine_data_ordered)

combined_ds = combined_ds.remove_columns(ds.column_names)
print("Amostra do dataset:", combined_ds["concat"][0])

# Load tokenizer
print("Loading the tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

max_length = 256
def tokenizer_function(sample):
    result = tokenizer(sample["concat"], truncation=True, padding="max_length", max_length=max_length)
    result["labels"] = result["input_ids"].copy()
    return result

# Tokenize dataset and create pytorch tensors
tokenizer_ds = combined_ds.map(tokenizer_function, batched=True)
tokenizer_ds.set_format("torch")

print(f"Loading the model {model_name} from hugginface")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")
epochs = 10
batch_size = 16
print("epochs", epochs)
print("batch_size", batch_size)
training_args = TrainingArguments(f"/hadatasets/fillipe.silva/LLMSegm/models/{experiment_name}", 
                                  num_train_epochs=epochs, 
                                  per_device_train_batch_size=batch_size,
                                  save_steps=5000)
trainer = Trainer(model, training_args, train_dataset=tokenizer_ds, tokenizer=tokenizer)

print("Trainer.train()")
trainer.train() 

model_name = model_name.replace("/","_") + "_" + str(epochs) + ".pt"
model_path = f"/hadatasets/fillipe.silva/LLMSegm/models/{experiment_name}/{model_name}"
print(f"Saving the model in: {model_path}")
torch.save(model.state_dict(), model_path)

path_to_test = f"/hadatasets/fillipe.silva/LLMSegm/data/{experiment_name}/test_rfm.csv"
print(f"Reading dataset for embedding generation {path_to_test}")
val_df = pd.read_csv(path_to_test)
columns = val_df.columns.tolist()
ds = Dataset.from_pandas(val_df)

combined_ds = ds.map(combine_data_shuffled)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Computing embbedings")
embs = []
for text in combined_ds["concat"]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)

    # Extract logits
    logits = outputs.logits

    # Use logits as text embeddings
    text_embedding = logits[:, -1, :]  # Take the last token's logits as the embedding

    # Convert tensor to numpy array if needed
    text_embedding_np = text_embedding.detach().cpu().numpy()

    embs.append(text_embedding_np[0])

