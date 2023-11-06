import pytest
import numpy as np
from image_search import *
from image_access import *
from index_access import *
from similarity_utility import *
from printing_engine import *
from matching_engine import *
from object_detection_engine import *
from image_search_manager import *
from object_detector import *

@pytest.fixture
def image_search_manager():
    return ImageSearchManager('default')

def test_ingest_image(image_search_manager):
    image_search_manager.ingest_image('example_images/image1.jpg')
    image_search_manager.ingest_image('example_images/image4.jpg')
    image_data = image_search_manager.index_access.read_image_data()
    assert len(image_data) == 2
    assert image_data[0][0] == 'example_images/image1.jpg'
    assert image_data[1][0] == 'example_images/image4.jpg'

def test_retrieve_images_matching_terms(image_search_manager):
    all = True
    terms = ['car', 'person']
    image_search_manager.ingest_image('example_images/image2.jpg')
    image_search_manager.ingest_image('example_images/image3.jpg')
    image_search_manager.ingest_image('example_images/image5.jpg')
    matching_images = image_search_manager.matching_engine.find_matching_images(all, set(terms))
    assert len(matching_images) == 2

def test_retrieve_similar_images(image_search_manager):
    input_labels = ['car', 'person']
    similarity_scores = image_search_manager.matching_engine.get_similar_images(input_labels)
    assert len(similarity_scores) >= 2

def test_list_images(image_search_manager, capsys):
    image_path = 'example_images/image6.jpg'
    image_search_manager.ingest_image(image_path)
    image_search_manager.list_images()
    captured = capsys.readouterr()
    assert image_path in captured.out

def test_default_object_detection_engine():
    image_path = ('example_images/image2.jpg')
    image_loader = ImreadImageLoader()
    image_access = ImageAccess(image_loader)
    image = image_access.read_image_path(image_path)
    object_detection = ObjectDetectionEngineFactory.create_object_detection_engine('default')
    detected_objects = object_detection.use_object_detector(image)
    expected_objects = {'person', 'car', 'truck'}    
    detected_objects_set = set(detected_objects)
    assert detected_objects_set == expected_objects

def test_matching_engine():
    matching_engine = MatchingEngine()
    all = True
    term_set = {'car', 'person'}
    matching_images = matching_engine.find_matching_images(all, term_set)
    similar_images = matching_engine.get_similar_images(term_set)
    assert isinstance(matching_images, type([]))
    assert isinstance(similar_images, type([]))

def test_printing_engine(capsys):
    printing_engine = PrintingEngine()
    detected_objects = ['person', 'car', 'truck']
    printing_engine.print_detection_results(detected_objects)
    captured = capsys.readouterr()
    assert captured.out.strip() == "Detected objects: " + ",".join(detected_objects)

def test_index_access():
    index_access = IndexAccess()
    index_access.setup_csv_file()
    image_data = index_access.read_image_data()
    assert isinstance(image_data, type([]))
    assert index_access.get_total_num_images() == 6

def test_image_access():
    image_loader = ImreadImageLoader()
    image_access = ImageAccess(image_loader)
    image_from_image_access = image_access.read_image_path('example_images/image1.jpg')
    image_from_imread = imread('example_images/image1.jpg')    
    assert np.array_equal(image_from_image_access, image_from_imread)

def test_cosine_similarity_metric():
    metric = CosineSimilarityMetric()
    input_labels = ['label1', 'label2']
    other_labels = ['label2', 'label3']
    similarity_result = metric.calculate_similarity(input_labels, other_labels)
    input_labels_vector = np.array(encode_labels(input_labels)).reshape(1, -1)
    other_labels_vector = np.array(encode_labels(other_labels)).reshape(1, -1)
    cosine_similarity_result = cosine_similarity(input_labels_vector, other_labels_vector)[0, 0]
    assert similarity_result == cosine_similarity_result
    assert isinstance(similarity_result, float)

def test_similarity_utility():
    metric = CosineSimilarityMetric()
    utility = SimilarityUtility(metric)
    input_labels = ['label1', 'label2']
    similarity_scores = utility.process_similarity_scores([], input_labels)
    assert len(similarity_scores) == 0
    assert isinstance(similarity_scores, type([]))

if __name__ == '__main__':
    pytest.main()

# image_data.csv file should only contain Image_Path,Detected_Objects on the first line
# warnings that may say 'Deprecated NumPy' is related to the deprecation of certain behavior in NumPy,
# and it's not something that can be directly controlled in the code.