import torch
import numpy as np
import openai
import os
from tqdm import tqdm
import time

from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2Model
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"

openai.api_key =  os.environ.get("OPENAI_API_KEY")

class EmbeddingModels:
    def __init__(self, name):
        self.name = name
        self.opai = openai.OpenAI()
        self.device = self.__set_device()

    def __set_device(self):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            #model = torch.nn.DataParallel(model)
        self.device = torch.device("cpu")
        print(f"device: {self.device}")
        return self.device    
        
    
    def calculate_embedding(self, texts):
        switcher = {
            'bert': self.__calculate_embedding_bert,
            'openai': self.__calculate_embedding_openai,
            'llama': self.__calculate_embedding_llama,
            'falcon': self.__calculate_embedding_falcon,
            'gpt2_medium': self.__calculate_embedding_gpt2_medium,
        }

        func = switcher.get(self.name, lambda x: f"No embedding model named '{self.name}'")
        return func(texts)

    def __calculate_embedding_gpt2_medium(self, texts):
        print("Computing embeddings with GPT2 Medium")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        model = GPT2Model.from_pretrained('gpt2-medium')
        model = model.to(self.device)

        # Verifique se há um token de padding e defina um se necessário
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batch_size = 16
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)

            # Move tensor inputs to the same device as the model
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.detach().cpu().numpy())
            
        # Concatenate all batch embeddings
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

    def __calculate_embedding_bert(self, texts):
        print("Computing BERT embeddings")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        model = model.to(self.device)
        return model.encode(texts)
    
    def __calculate_embedding_openai(self, texts):
        print("Computing ADA-002 (OpenAI) embeddings")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        truncated_texts = []
        for text in texts:
            tokenized_text = tokenizer.encode(text, truncation=True, max_length=7500, return_tensors='pt')
            truncated_text = tokenizer.decode(tokenized_text[0], skip_special_tokens=True)
            truncated_texts.append(truncated_text)

        batch_size = 500  
        all_embeddings = []
        for i in range(0, len(truncated_texts), batch_size):
            time.sleep(60)  # tokens per min (TPM): Limit 1000000,
            batch_texts = truncated_texts[i:i+batch_size]
            try:
                batch_embeddings = self.opai.embeddings.create(input=batch_texts, model="text-embedding-3-large").data
                all_embeddings.extend([embedding.embedding for embedding in batch_embeddings])
            except Exception as e:
                print(f"An error occurred: {e}")
        return all_embeddings

    def __calculate_embedding_falcon(self, texts):
        print("Computing Falcon embeddings")
        tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b')
        model = AutoModel.from_pretrained('tiiuae/falcon-7b', trust_remote_code=True)
        model = model.to(self.device)

        # Verifique se há um token de padding e defina um se necessário
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batch_size = 16
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=10)
            inputs = {key: value for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.detach().cpu().numpy())
            
        # Concatenate all batch embeddings
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings   

    def __calculate_embedding_llama(self, texts):
        print("Computing LLaMA embeddings")
        tokenizer = AutoTokenizer.from_pretrained("/hadatasets/fillipe.silva/cached_models_huggingface/Llama-2-7b-chat-hf")
        model = AutoModel.from_pretrained("/hadatasets/fillipe.silva/cached_models_huggingface/Llama-2-7b-chat-hf")
        model = model.to(self.device)

        # Verifique se há um token de padding e defina um se necessário
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batch_size = 16
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=10)
            inputs = {key: value for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.detach().cpu().numpy())
            
        # Concatenate all batch embeddings
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings   