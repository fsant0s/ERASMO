# embedding_utils/paraphrase_mpnet_embeddings.py
from sentence_transformers import SentenceTransformer
import torch

class ParaphraseMpnetEmbeddings:

    def __set_device(self):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            
        self.device = torch.device("cpu")
        print(f"device: {self.device}")
        return self.device    

    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name, device=self.__set_device())

    def calculate_embedding(self, text):
        """
        Returns the embedding of the given text using the specified model.
        """
        embedding = self.model.encode(text)
        return embedding
