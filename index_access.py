import os
from abc import ABC, abstractmethod
import csv
from similarity_utility import CosineSimilarityMetric, SimilarityUtility

CSV_FILE = 'image_data.csv'

class IIndexAccess(ABC):
    """Abstract base class for index access."""

    @abstractmethod
    def setup_csv_file(self):
        """Set up the CSV file for image data storage."""

    @abstractmethod
    def save_image_data(self, image_path, detected_objects):
        """Save image data, including detected objects, to the CSV file."""

    @abstractmethod
    def access_matching_images(self, all, term_set):
        """
        Access and return matching images based on query terms.

        :param all: If True, return images matching all query terms. If False, return images matching some query terms.
        :param term_set: A set of query terms.
        :return: A list of matching images and their detected objects.
        """

    @abstractmethod
    def read_image_data(self):
        """Read image data from the CSV file and return it."""

    @abstractmethod
    def calculate_similarity_scores(self, input_labels):
        """Calculate and return similarity scores based on input labels."""

    @abstractmethod
    def get_total_num_images(self):
        """Get the total number of images in the CSV file."""

class IndexAccess(IIndexAccess):
    """
    Implementation of IndexAccess for accessing and managing image data stored in a CSV file.
    """

    def __init__(self):
        """
        Initialize the IndexAccess object and set up necessary components.

        Initializes the CSV file, similarity metric, and similarity utility.
        """
        self.csv_file = CSV_FILE
        self.similarity_metric = CosineSimilarityMetric()
        self.similarity_utility = SimilarityUtility(self.similarity_metric)

    def setup_csv_file(self):
        """Set up the CSV file for image data storage if it doesn't exist."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                try:
                    writer = csv.writer(file)
                    writer.writerow(["Image_Path", "Detected_Objects"])
                except (csv.Error, IOError) as e:
                    print(f"Error: {e}")

    def save_image_data(self, image_path, detected_objects):
        """
        Save image data, including detected objects, to the CSV file.

        :param image_path: The path of the image to be saved.
        :param detected_objects: A list of detected objects in the image.
        """
        with open(self.csv_file, mode='a', newline='') as file:
            try:
                writer = csv.writer(file)
                writer.writerow([image_path, ",".join(detected_objects)])
            except (csv.Error, IOError) as e:
                print(f"Error: {e}")

    def access_matching_images(self, all, term_set):
        """
        Access and return matching images based on query terms.

        :param all: If True, return images matching all query terms. If False, return images matching some query terms.
        :param term_set: A set of query terms.
        :return: A list of matching images and their detected objects.
        """
        matching_images = []
        with open(self.csv_file, mode='r') as file:
            try:
                reader = csv.DictReader(file)
                for row in reader:
                    image_path = row["Image_Path"]
                    detected_objects = row["Detected_Objects"].split(",")
                    if all:
                        if term_set.issubset(detected_objects):
                            matching_images.append((image_path, detected_objects))
                    else:
                        if term_set.intersection(detected_objects):
                            matching_images.append((image_path, detected_objects))
            except (csv.Error, IOError) as e:
                print(f"Error: {e}")
        return matching_images

    def read_image_data(self):
        """Read image data from the CSV file and return it."""
        with open(self.csv_file, mode='r') as file:
            try:
                reader = csv.DictReader(file)
                return [(row["Image_Path"], row["Detected_Objects"].split(",")) for row in reader]
            except (csv.Error, IOError) as e:
                print(f"Error: {e}")

    def calculate_similarity_scores(self, input_labels):
        """
        Calculate and return similarity scores based on input labels.

        :param input_labels: A list of input labels for similarity calculation.
        :return: A list of similarity scores.
        """
        with open(self.csv_file, mode='r') as file:
            try:
                reader = csv.DictReader(file)
                similarity_scores = self.similarity_utility.process_similarity_scores(reader, input_labels)
            except (csv.Error, IOError) as e:
                print(f"Error: {e}")
        return similarity_scores

    def get_total_num_images(self):
        """Get the total number of images in the CSV file."""
        with open(CSV_FILE, mode='r') as file:
            try:
                return sum(1 for _ in file) - 1
            except (csv.Error, IOError) as e:
                print(f"Error: {e}")
