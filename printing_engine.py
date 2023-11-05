from abc import ABC, abstractmethod
from index_access import *

class IPrintingEngine(ABC):

    @abstractmethod
    def print_detection_results(self, detected_objects):
        pass

    @abstractmethod
    def print_image_data(self, image_data):
        pass

    @abstractmethod
    def print_matching_images(self, matching_images):
        pass

    @abstractmethod
    def print_similar_images(self, image_path, similarity_scores, k):
        pass

    @abstractmethod
    def print_total_num_images(self):
        pass

class PrintingEngine(IPrintingEngine):

    def __init__(self):
        self.index_access = IndexAccess()

    def print_detection_results(self, detected_objects):
        detected_objects_str = ",".join(detected_objects)
        print(f"Detected objects: {detected_objects_str}")

    def print_image_data(self, image_data):
        for image_path, detected_objects in image_data:
            detected_objects_str = ",".join(detected_objects)
            print(f"{image_path}: {detected_objects_str}")

    def print_matching_images(self, matching_images):
        self.print_image_data(matching_images)
        print(f"{len(matching_images)} matches found.")

    def print_similar_images(self, image_path, similarity_scores, k):
        for image_path, similarity in similarity_scores[:k]:
            print(f"{similarity:.4f} {image_path}")

    def print_total_num_images(self):
        total_images = self.index_access.get_total_num_images()
        print(f"{total_images} images found.")