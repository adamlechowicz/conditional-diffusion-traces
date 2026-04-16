import faiss
import numpy as np
import torch
from sklearn.cluster import KMeans

class VectorDB:
    def __init__(self, config):
        self.config = config
        self.seq_length = config.seq_length
        self.num_clusters = config.num_clusters

        # We will use an L2 index for standard Euclidean distance
        # FAISS expects 2D float32 numpy arrays: (N, D)
        self.index = faiss.IndexFlatL2(self.seq_length)
        self.is_trained = False

    def build_index(self, data):
        """
        data: numpy array of shape (N, L) where N is number of daily traces,
              and L is seq_length (1440).
        """
        assert data.shape[1] == self.seq_length
        data = data.astype(np.float32)

        # If N > num_clusters, perform k-means to extract canonical shapes
        if data.shape[0] > self.num_clusters:
            print(f"Clustering {data.shape[0]} traces into {self.num_clusters} canonical shapes...")
            kmeans = faiss.Kmeans(d=self.seq_length, k=self.num_clusters, niter=20, verbose=True)
            kmeans.train(data)
            templates = kmeans.centroids
        else:
            print(f"Data size {data.shape[0]} <= {self.num_clusters}, using raw data as templates.")
            templates = data

        self.index.add(templates)
        self.is_trained = True
        print(f"Index built with {self.index.ntotal} templates.")

    def save(self, filepath):
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained index.")
        faiss.write_index(self.index, filepath)

    def load(self, filepath):
        self.index = faiss.read_index(filepath)
        self.is_trained = True

    def retrieve(self, query, k=1):
        """
        query: numpy array of shape (B, L) or (L,)
        Returns the nearest neighbor templates of shape (B, k, L)
        """
        if not self.is_trained:
            raise RuntimeError("Index is not trained/loaded.")

        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = query.astype(np.float32)

        # D: Distances, I: Indices
        D, I = self.index.search(query, k)

        # Fetch the actual vectors from the index
        # For IndexFlatL2, we can access the vectors directly
        retrieved_vectors = np.array([self.index.reconstruct(int(idx)) for idx in I.flatten()])
        retrieved_vectors = retrieved_vectors.reshape(query.shape[0], k, self.seq_length)

        return retrieved_vectors
