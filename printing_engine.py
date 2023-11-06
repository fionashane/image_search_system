from abc import ABC, abstractmethod
from index_access import *

class IPrintingEngine(ABC):
    """Abstract base class for a printing engine."""

    @abstractmethod
    def print_detection_results(self, detected_objects):
        """
        Print the detected objects.

        :param detected_objects: A list of detected objects.
        """

    @abstractmethod
    def print_image_data(self, image_data):
        """
        Print image data, including detected objects.

        :param image_data: A list of image data, each containing image path and detected objects.
        """

    @abstractmethod
    def print_matching_images(self, matching_images):
        """
        Print matching images and their detected objects.

        :param matching_images: A list of matching images with detected objects.
        """

    @abstractmethod
    def print_similar_images(self, image_path, similarity_scores, k):
        """
        Print similar images and their similarity scores.

        :param image_path: The path of the reference image.
        :param similarity_scores: A list of (image path, similarity score) pairs.
        :param k: The number of similar images to print.
        """

    @abstractmethod
    def print_total_num_images(self):
        """
        Print the total number of images in the dataset.
        """

class PrintingEngine(IPrintingEngine):
    """Implementation of a printing engine."""

    def __init__(self):
        self.index_access = IndexAccess()

    def print_detection_results(self, detected_objects):
        """
        Print the detected objects.

        :param detected_objects: A list of detected objects.
        """
        detected_objects_str = ",".join(detected_objects)
        print(f"Detected objects: {detected_objects_str}")

    def print_image_data(self, image_data):
        """
        Print image data, including detected objects.

        :param image_data: A list of image data, each containing image path and detected objects.
        """
        for image_path, detected_objects in image_data:
            detected_objects_str = ",".join(detected_objects)
            print(f"{image_path}: {detected_objects_str}")

    def print_matching_images(self, matching_images):
        """
        Print matching images and their detected objects.

        :param matching_images: A list of matching images with detected objects.
        """
        self.print_image_data(matching_images)
        print(f"{len(matching_images)} matches found.")

    def print_similar_images(self, image_path, similarity_scores, k):
        """
        Print similar images and their similarity scores.

        :param image_path: The path of the reference image.
        :param similarity_scores: A list of (image path, similarity score) pairs.
        :param k: The number of similar images to print.
        """
        for image_path, similarity in similarity_scores[:k]:
            print(f"{similarity:.4f} {image_path}")

    def print_total_num_images(self):
        """Print the total number of images in the dataset."""
        total_images = self.index_access.get_total_num_images()
        print(f"{total_images} images found.")
