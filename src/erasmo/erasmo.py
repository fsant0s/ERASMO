import os
import warnings
import json
import typing as tp
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          TrainingArguments)

from erasmo.erasmo_dataset import ErasmoDataset, ErasmoDataCollator
from erasmo.erasmo_trainer import ErasmoTrainer
from erasmo.erasmo_utils import _array_to_dataframe

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training , TaskType

class Erasmo:
    """ Erasmo Class

    The Erasmo class handles the whole generation flow. It is used to fine-tune a large language model for tabular data.
    """

    def __init__(self, model_name: str, experiment_dir: str = "trainer_Erasmo", text_to_num:bool = False, epochs: int = 100,
                 batch_size: int = 8, tokenizermax_length: int = 256, efficient_finetuning: str = "", **train_kwargs):
         # Load Model and Tokenizer from HuggingFace
        self.efficient_finetuning = efficient_finetuning
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, max_length=tokenizermax_length)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.text_to_num = text_to_num

        if self.efficient_finetuning == "lora":
            # Define LoRA Config
            lora_config = LoraConfig(
                r=16, # only training 0.16% of the parameters of the model
                lora_alpha=32,
                target_modules=["c_attn"], # this is specific for gpt2 model, to be adapted
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM # this is specific for gpt2 model, to be adapted
            )
            # prepare int-8 model for training
            self.model = prepare_model_for_kbit_training(self.model)
            # add LoRA adaptor
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Set the training hyperparameters
        self.experiment_dir = f"{experiment_dir}_{model_name}_{epochs}_{text_to_num}"
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_hyperparameters = train_kwargs

    def fit(self, data: tp.Union[pd.DataFrame, np.ndarray], column_names: tp.Optional[tp.List[str]] = None,
           resume_from_checkpoint: tp.Union[bool, str] = False) \
            -> ErasmoTrainer:
        
        df = _array_to_dataframe(data, columns=column_names)

        # Convert DataFrame into HuggingFace dataset object
        logging.info("Convert data into HuggingFace dataset object...")
        ds = ErasmoDataset.from_pandas(df)
        ds.set_tokenizer(self.tokenizer)
        ds.set_text_to_num(self.text_to_num)

        # Set training hyperparameters
        logging.info("Create Erasmo Trainer...")
        training_args = TrainingArguments(self.experiment_dir,
                                          num_train_epochs=self.epochs,
                                          per_device_train_batch_size=self.batch_size,
                                          **self.train_hyperparameters)
        trainer = ErasmoTrainer(self.model, training_args, train_dataset=ds, tokenizer=self.tokenizer,
                                     data_collator=ErasmoDataCollator(self.tokenizer))

        # Start training
        logging.info("Start training...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return trainer
    
    def __getstate__(self) -> object:
        pass

    def __get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def generate_embeddings_from_last_layer(self, data):
        device = self.__get_device()
        self.model.to(device)

        ds = ErasmoDataset.from_pandas(data)
        ds.set_tokenizer(self.tokenizer)
        ds.set_text_to_num(self.text_to_num)

        embeddings = []
        for token_ids in tqdm(ds):
            input_ids_tensor = torch.tensor(token_ids['input_ids'], dtype=torch.long, device=device)
            
            with torch.no_grad():
                outputs = self.model(input_ids_tensor)
                text_embedding = outputs.logits
                embeddings.append(text_embedding.detach().cpu().numpy()[0])
            
        logging.info("Saving embeddings from last layer...")
        embedding_df = pd.DataFrame(embeddings)
        embedding_df.to_csv(self.experiment_dir + "/embeddings_from_last_layer.csv", index=False)
        return embeddings

    def generate_embeddins_from_all_layers(self, data):
        device = self.__get_device()
        self.model.to(device)

        ds = ErasmoDataset.from_pandas(data)
        ds.set_tokenizer(self.tokenizer)
        ds.set_text_to_num(self.text_to_num)

        embeddings = []
        for token_ids in tqdm(ds):
            # Convert token IDs to tensor and move it to the model's device
            tokens_tensor = torch.tensor([token_ids['input_ids']], dtype=torch.long, device=self.model.device)
            with torch.no_grad():
                # Forward pass through the model
                outputs = self.model(tokens_tensor)
                # Retrieve the hidden states from the model output
                hidden_states = outputs[0]  # 'outputs' is a tuple, the first element is the hidden states
            # Averaging over the sequence length
            embeddings.append(hidden_states[0].mean(dim=0).detach().cpu().numpy())

        logging.info("Saving embeddings from all layer...")
        embedding_df = pd.DataFrame(embeddings)
        embedding_df.to_csv(self.experiment_dir + "/embeddins_from_all_layers.csv", index=False)
        return embeddings

    def save(self):
        """ Save Erasmo Model

        Saves the model weights and a configuration file in the given directory.

        Args:
            path: Path where to save the model
        """
        path = self.experiment_dir
        # Make directory
        if os.path.isdir(path):
            warnings.warn(
                f"Directory {path} already exists and is overwritten now.")
        else:
            os.mkdir(path)

        # Save attributes
        with open(path + "/config.json", "w") as f:
            attributes = self.__dict__.copy()
            attributes.pop("tokenizer")
            attributes.pop("model")

            json.dump(attributes, f)

        # Save model weights
        torch.save(self.model.state_dict(), path + "/model.pt")

    def load_finetuned_model(self, path: str):
        """ Load fine-tuned model

        Load the weights of a fine-tuned large language model into the Erasmo pipeline

        Args:
            path: Path to the fine-tuned model
        """
        self.model.load_state_dict(torch.load(path))

    @classmethod
    def load_from_dir(cls, path: str):
        """ Load Erasmo class

        Load trained Erasmo model from directory.

        Args:
            path: Directory where Erasmo model is saved

        Returns:
            New instance of Erasmo loaded from directory
        """
        assert os.path.isdir(path), f"Directory {path} does not exist."

        # Load attributes
        with open(path + "/config.json", "r") as f:
            attributes = json.load(f)

        # Create new erasmo model instance
        erasmo = cls(attributes["llm"])

        # Set all attributes
        for k, v in attributes.items():
            setattr(erasmo, k, v)

        # Load model weights
        erasmo.model.load_state_dict(torch.load(
            path + "/model.pt", map_location="cpu"))

        return erasmo
