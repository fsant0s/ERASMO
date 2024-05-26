from erasmo import Erasmo
import matplotlib.pyplot as plt

import pandas as pd
import sys
import os
import logging
import torch

print("Cleaning cache")
torch.cuda.empty_cache()
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5"
if torch.cuda.is_available():
    print("CUDA is available. Here are the CUDA devices:")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")

CURRENT_DIR = (os.path.dirname(os.path.abspath('__file__')))

stream_handler = logging.StreamHandler(sys.stderr)
file_handler = logging.FileHandler(
    filename=os.path.join("../logs/", "erasmo_banking.txt"),  encoding='utf-8', mode="w+"
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    level=logging.INFO,
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[stream_handler, file_handler],
)

#-----------------------------------------------
data = pd.read_csv("/hadatasets/fillipe.silva/LLMSegm/data/banking/train.csv")
data_to_embd = pd.read_csv("/hadatasets/fillipe.silva/LLMSegm/data/banking/test.csv")
#-----------------------------------------------

#-----------------------------------------------
#Hyperparameters
model_name = "gpt2-medium"
n_epochs = 20
save_steps = 2000
logging_steps = 300
experiment_dir = "banking"
batch_size = 16
tokenizermax_length = 256
#-----------------------------------------------

#-----------------------------------------------
logging.info("Training with 'Text to Num' disabled")
erasmo = Erasmo(model_name,                       # Name of the large language model used (see HuggingFace for more options)
              epochs=n_epochs,                             # Number of epochs to train (only one epoch for demonstration)
              save_steps=save_steps,                      # Save model weights every x steps
              logging_steps=logging_steps,                     # Log the loss and learning rate every x steps
              experiment_dir=experiment_dir,               # Name of the directory where all intermediate steps are saved
              text_to_num=False,                    # Convert text to numbers
              batch_size=batch_size,                      # Set the batch size
              tokenizermax_length=tokenizermax_length,            # Set the maximum length of the input text
              #lr_scheduler_type="constant",        # Specify the learning rate scheduler 
              #learning_rate=5e-5 ,                 # Set the inital learning rate
             )

trainer = erasmo.fit(data)
erasmo.generate_embeddins_from_all_layers(data_to_embd)
erasmo.generate_embeddings_from_last_layer(data_to_embd)
erasmo.save()

loss_hist = trainer.state.log_history.copy()
loss_hist.pop()
loss = [x["loss"] for x in loss_hist]
epochs = [x["epoch"] for x in loss_hist]
plt.figure()
plt.plot(epochs, loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.savefig(f'{CURRENT_DIR}/{experiment_dir}_{model_name}_{n_epochs}_False/training_loss_plot.png')
plt.close()
#-----------------------------------------------

#-----------------------------------------------
logging.info("Del erasmo object")
del erasmo
#-----------------------------------------------

#-----------------------------------------------
logging.info("Training with 'Text to Num' enabled")
erasmo = Erasmo(model_name,                       # Name of the large language model used (see HuggingFace for more options)
              epochs=n_epochs,                             # Number of epochs to train (only one epoch for demonstration)
              save_steps=save_steps,                      # Save model weights every x steps
              logging_steps=logging_steps,                     # Log the loss and learning rate every x steps
              experiment_dir=experiment_dir,               # Name of the directory where all intermediate steps are saved
              text_to_num=True,                    # Convert text to numbers
              batch_size=batch_size,                      # Set the batch size
              tokenizermax_length=tokenizermax_length,            # Set the maximum length of the input text
              #lr_scheduler_type="constant",        # Specify the learning rate scheduler 
              #learning_rate=5e-5 ,                 # Set the inital learning rate
             )

trainer = erasmo.fit(data)
erasmo.generate_embeddins_from_all_layers(data_to_embd)
erasmo.generate_embeddings_from_last_layer(data_to_embd)
erasmo.save()

loss_hist = trainer.state.log_history.copy()
loss_hist.pop()
loss = [x["loss"] for x in loss_hist]
epochs = [x["epoch"] for x in loss_hist]
plt.figure()
plt.plot(epochs, loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.savefig(f'{CURRENT_DIR}/{experiment_dir}_{model_name}_{n_epochs}_True/training_loss_plot.png')
plt.close()
#-----------------------------------------------

del erasmo

logging.info("Experiment done!")