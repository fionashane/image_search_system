from abc import ABC, abstractmethod
from object_detector import detect_objects

class IObjectDetectionEngine(ABC):
    """Abstract base class for an object detection engine."""

    @abstractmethod
    def use_object_detector(self, image):
        """
        Use an object detector to detect objects in the given image.

        :param image: The image for object detection.
        :return: A list of detected objects.
        """

class DefaultObjectDetectionEngine(IObjectDetectionEngine):
    """Default implementation of an object detection engine."""

    def __init__(self, detect_objects):
        """
        Initialise the DefaultObjectDetectionEngine with a detect_objects function.

        :param detect_objects: A function for object detection.
        """
        self.detect_objects = detect_objects

    def use_object_detector(self, image):
        """
        Use the object detector to detect objects in the given image.

        :param image: The image for object detection.
        :return: A list of detected objects.
        """
        try:
            return self.detect_objects(image)
        except Exception as e:
            print(f"Error in object detection: {e}")
            return []

class CustomObjectDetectionEngine(IObjectDetectionEngine):
    """Custom implementation of an object detection engine."""

    def use_object_detector(self, image):
        """
        Use a custom object detector to detect objects in the given image.

        :param image: The image for object detection.
        :return: A list of detected objects (not implemented in this class).
        """

class ObjectDetectionEngineFactory:
    """Factory for creating object detection engines."""

    def create_object_detection_engine(engine_type):
        """
        Create and return an object detection engine based on the specified type.

        :param engine_type: The type of object detection engine to create ('default' or 'custom').
        :return: An instance of the object detection engine.
        :raises ValueError: If an invalid object detection engine type is provided.
        """
        if engine_type == 'default':
            return DefaultObjectDetectionEngine(detect_objects)
        elif engine_type == 'custom':
            return CustomObjectDetectionEngine()
        else:
            raise ValueError("Invalid object detection engine type")
