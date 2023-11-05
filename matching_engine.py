from abc import ABC, abstractmethod
from index_access import *

class IMatchingEngine(ABC):

    @abstractmethod
    def find_matching_images(self, all, term_set):
        pass
    
    @abstractmethod
    def get_similar_images(self, similarity_scores, input_labels):
        pass

class MatchingEngine(IMatchingEngine):

    def __init__(self):
        self.index_access = IndexAccess()

    def find_matching_images(self, all, term_set):
        matching_images = []
        matching_images = self.index_access.access_matching_images(all, term_set, matching_images)
        return matching_images
    
    def get_similar_images(self, similarity_scores, input_labels):
        similarity_scores = self.index_access.calculate_similarity_scores(input_labels, similarity_scores)
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        return similarity_scores
