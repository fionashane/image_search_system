from index_access import IndexAccess
from image_access import ImageAccess, ImreadImageLoader
from printing_engine import PrintingEngine
from matching_engine import MatchingEngine
from object_detection_engine import ObjectDetectionEngineFactory

class ImageSearchManager:
    """Manages image data, object detection, and searching for images based on object types."""

    def __init__(self, object_detection_engine_type):
        """
        Initialise the ImageSearchManager with the specified object detection engine type.

        :param object_detection_engine_type: The type of object detection engine to use.
        """
        self.index_access = IndexAccess()
        self.image_loader = ImreadImageLoader()
        self.image_access = ImageAccess(self.image_loader)
        self.printing_engine = PrintingEngine()
        self.matching_engine = MatchingEngine()
        self.object_detection_engine = ObjectDetectionEngineFactory.create_object_detection_engine(object_detection_engine_type)
        self.index_access.setup_csv_file()

    def ingest_image(self, image_path):
        """
        Ingest an image, detect objects, and store the image and detected objects in a CSV file.

        :param image_path: The path to the image file to ingest.
        """
        image = self.image_access.read_image_path(image_path)
        detected_objects = self.object_detection_engine.use_object_detector(image)
        self.index_access.save_image_data(image_path, detected_objects)
        self.printing_engine.print_detection_results(detected_objects)

    def retrieve_images_matching_terms(self, all, terms):
        """
        Retrieve images based on object types from the CSV file.

        :param all: If True, retrieve images that match all query terms. If False, retrieve images that match some of the terms.
        :param terms: A list of query terms to search for in the images.
        """
        term_set = set(terms)
        matching_images = self.matching_engine.find_matching_images(all, term_set)
        self.printing_engine.print_matching_images(matching_images)

    def retrieve_similar_images(self, k, image_path):
        """
        Retrieve similar images based on cosine similarity of object types using data from the CSV file.

        :param k: The number of similar images to return.
        :param image_path: The path to the image for which to find similar images.
        """
        image = self.image_access.read_image_path(image_path)
        input_labels = self.object_detection_engine.use_object_detector(image)
        similarity_scores = self.matching_engine.get_similar_images(input_labels)
        self.printing_engine.print_similar_images(image_path, similarity_scores, k)

    def list_images(self):
        """List all images and their associated object types from the CSV file."""
        image_data = self.index_access.read_image_data()
        self.printing_engine.print_image_data(image_data)
        self.printing_engine.print_total_num_images()
