from index_access import *
from image_access import *
from similarity_utility import *
from printing_engine import *
from matching_engine import *
from object_detection_engine import *

class ImageSearchManager():

    def __init__(self, object_detection_engine_type):
        self.index_access = IndexAccess()
        self.image_loader = ImreadImageLoader()
        self.image_access = ImageAccess(self.image_loader)
        self.printing_engine = PrintingEngine()
        self.matching_engine = MatchingEngine()
        self.object_detection_engine = ObjectDetectionEngineFactory.create_object_detection_engine(object_detection_engine_type)
        self.index_access.setup_csv_file()

    def ingest_image(self, image_path):
        image = self.image_access.read_image_path(image_path)
        detected_objects = self.object_detection_engine.use_object_detector(image)
        self.index_access.save_image_data(image_path, detected_objects)
        self.printing_engine.print_detection_results(detected_objects)

    def retrieve_images_matching_terms(self, all, terms):
        term_set = set(terms)
        matching_images = self.matching_engine.find_matching_images(all, term_set)
        self.printing_engine.print_matching_images(matching_images)

    def retrieve_similar_images(self, k, image_path):
        image = self.image_access.read_image_path(image_path)
        input_labels = self.object_detection_engine.use_object_detector(image)
        similarity_scores = []
        similarity_scores = self.matching_engine.get_similar_images(similarity_scores, input_labels)
        self.printing_engine.print_similar_images(image_path, similarity_scores, k)

    def list_images(self):
        image_data = self.index_access.read_image_data()
        self.printing_engine.print_image_data(image_data)
        self.printing_engine.print_total_num_images()