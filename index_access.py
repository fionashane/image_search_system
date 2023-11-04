import os
from abc import ABC, abstractmethod
import csv
from similarity_utility import *

CSV_FILE = 'image_data.csv'

class IIndexAccess(ABC):

    @abstractmethod
    def setup_csv_file(self):
        pass

    @abstractmethod
    def save_image_data(self, image_path, detected_objects):
        pass

    @abstractmethod
    def access_matching_images(self, all, term_set, matching_images):
        pass

    @abstractmethod
    def read_image_data(self):
        pass

    @abstractmethod
    def calculate_similarity_scores(self, input_labels, similarity_scores):
        pass

    @abstractmethod
    def get_total_num_images(self):
        pass

class IndexAccess(IIndexAccess):
    
    def __init__(self):
        self.csv_file = CSV_FILE
        self.similarity_metric = CosineSimilarityMetric()
        self.similarity_utility = SimilarityUtility(self.similarity_metric)

    def setup_csv_file(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                try:
                    writer = csv.writer(file)
                    writer.writerow(["Image_Path", "Detected_Objects"])
                except (csv.Error, IOError) as e:
                    print(f"Error: {e}")

    def save_image_data(self, image_path, detected_objects):
        with open(self.csv_file, mode='a', newline='') as file:
            try:
                writer = csv.writer(file)
                writer.writerow([image_path, ",".join(detected_objects)])
            except (csv.Error, IOError) as e:
                print(f"Error: {e}")

    def access_matching_images(self, all, term_set, matching_images):
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
        with open(self.csv_file, mode='r') as file:
            try:
                reader = csv.DictReader(file)
                return [(row["Image_Path"], row["Detected_Objects"].split(",")) for row in reader]
            except (csv.Error, IOError) as e:
                print(f"Error: {e}")

    def calculate_similarity_scores(self, input_labels, similarity_scores):
        with open(self.csv_file, mode='r') as file:
            try:
                reader = csv.DictReader(file)
                similarity_scores = self.similarity_utility.process_similarity_scores(reader, input_labels, similarity_scores)
            except (csv.Error, IOError) as e:
                print(f"Error: {e}")
        return similarity_scores

    def get_total_num_images(self):
        with open(CSV_FILE, mode='r') as file:
            try:
                return sum(1 for _ in file) - 1
            except (csv.Error, IOError) as e:
                print(f"Error: {e}")
