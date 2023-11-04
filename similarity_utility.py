from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from object_detector import encode_labels, ALL_LABELS

class SimilarityMetric(ABC):

    @abstractmethod
    def calculate_similarity(self, input_labels, other_labels):
        pass

class CosineSimilarityMetric(SimilarityMetric):

    def calculate_similarity(self, input_labels, other_labels):
        try:
            input_labels_vector = np.array(encode_labels(input_labels)).reshape(1, -1)
            other_labels_vector = np.array(encode_labels(other_labels)).reshape(1, -1)
            return cosine_similarity(input_labels_vector, other_labels_vector)[0, 0]
        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            return 0.0

class ISimilarityUtility(ABC):

    @abstractmethod
    def process_similarity_scores(self, reader, input_labels, similarity_scores):
        pass

class SimilarityUtility():

    def __init__(self, similarity_metric):
        self.similarity_metric = similarity_metric

    def process_similarity_scores(self, reader, input_labels, similarity_scores):
        for row in reader:
            other_image_path = row["Image_Path"]
            other_labels = row["Detected_Objects"].split(",")
            similarity = self.similarity_metric.calculate_similarity(input_labels, other_labels)
            similarity_scores.append((other_image_path, similarity))
        return similarity_scores
