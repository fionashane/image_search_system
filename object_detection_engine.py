from abc import ABC, abstractmethod
from object_detector import detect_objects

class IObjectDetectionEngine(ABC):

    @abstractmethod
    def use_object_detector(self, image):
        pass

class DefaultObjectDetectionEngine(IObjectDetectionEngine):
            
    def __init__(self, detect_objects):
        self.detect_objects = detect_objects

    def use_object_detector(self, image):
        try:
            return self.detect_objects(image)
        except Exception as e:
            print(f"Error in object detection: {e}")
            return []

class CustomObjectDetectionEngine(IObjectDetectionEngine):

    def use_object_detector(self, image):
        pass

class ObjectDetectionEngineFactory:

    def create_object_detection_engine(engine_type):
        if engine_type == 'default':
            return DefaultObjectDetectionEngine(detect_objects)
        elif engine_type == 'custom':
            return CustomObjectDetectionEngine()
        else:
            raise ValueError("Invalid object detection engine type")