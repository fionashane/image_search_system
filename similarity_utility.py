from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from object_detector import encode_labels

class SimilarityMetric(ABC):
    """Abstract base class for a similarity metric."""

    @abstractmethod
    def calculate_similarity(self, input_labels, other_labels):
        """
        Calculate the similarity between two sets of labels.

        :param input_labels: The labels of the input data.
        :param other_labels: The labels of the other data for comparison.
        """

class CosineSimilarityMetric(SimilarityMetric):
    """Implementation of the Cosine Similarity metric."""

    def calculate_similarity(self, input_labels, other_labels):
        """
        Calculate the cosine similarity between two sets of labels.

        :param input_labels: The labels of the input data.
        :param other_labels: The labels of the other data for comparison.
        """
        try:
            input_labels_vector = np.array(encode_labels(input_labels)).reshape(1, -1)
            other_labels_vector = np.array(encode_labels(other_labels)).reshape(1, -1)
            return cosine_similarity(input_labels_vector, other_labels_vector)[0, 0]
        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            return 0.0

class ISimilarityUtility(ABC):
    """Abstract base class for a similarity utility."""

    @abstractmethod
    def process_similarity_scores(self, reader, input_labels):
        """
        Process and calculate similarity scores.

        :param reader: The data reader for reading image data.
        :param input_labels: The labels of the input data for comparison.
        """

class SimilarityUtility():
    """Implementation of a similarity utility."""

    def __init__(self, similarity_metric):
        self.similarity_metric = similarity_metric

    def process_similarity_scores(self, reader, input_labels):
        """
        Process and calculate similarity scores for input labels.

        :param reader: The data reader for reading image data.
        :param input_labels: The labels of the input data for comparison.
        """
        similarity_scores = []
        for row in reader:
            other_image_path = row["Image_Path"]
            other_labels = row["Detected_Objects"].split(",")
            similarity = self.similarity_metric.calculate_similarity(input_labels, other_labels)
            similarity_scores.append((other_image_path, similarity))
        return similarity_scores
