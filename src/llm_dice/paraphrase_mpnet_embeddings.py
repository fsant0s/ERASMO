# embedding_utils/paraphrase_mpnet_embeddings.py
from sentence_transformers import SentenceTransformer

class ParaphraseMpnetEmbeddings:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        """
        Returns the embedding of the given text using the specified model.
        """
        embedding = self.model.encode(text)
        return embedding
