from abc import ABC, abstractmethod
from index_access import IndexAccess

class IMatchingEngine(ABC):
    """Abstract base class for a matching engine."""

    @abstractmethod
    def find_matching_images(self, all, term_set):
        """
        Find and return matching images based on query terms.

        :param all: If True, return images matching all query terms. If False, return images matching some query terms.
        :param term_set: A set of query terms.
        :return: A list of matching images.
        """

    @abstractmethod
    def get_similar_images(self, input_labels):
        """
        Get and return similar images based on input labels.

        :param input_labels: A list of input labels for similarity calculation.
        :return: A list of similar images and their similarity scores.
        """

class MatchingEngine(IMatchingEngine):
    """Implementation of a matching engine."""

    def __init__(self):
        """Initialize the MatchingEngine object."""
        self.index_access = IndexAccess()

    def find_matching_images(self, all, term_set):
        """
        Find and return matching images based on query terms.

        :param all: If True, return images matching all query terms. If False, return images matching some query terms.
        :param term_set: A set of query terms.
        :return: A list of matching images.
        """
        matching_images = self.index_access.access_matching_images(all, term_set)
        return matching_images

    def get_similar_images(self, input_labels):
        """
        Get and return similar images based on input labels.

        :param input_labels: A list of input labels for similarity calculation.
        :return: A list of similar images and their similarity scores.
        """
        similarity_scores = self.index_access.calculate_similarity_scores(input_labels)
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        return similarity_scores
