from abc import ABC, abstractmethod
from matplotlib.image import imread

class ImageLoader(ABC):

    @abstractmethod
    def read_image(self, image_path):
        pass

class ImreadImageLoader(ImageLoader):

    def read_image(self, image_path):
        try:
            return imread(image_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None

class IImageAccess(ABC):

    @abstractmethod
    def read_image_path(self, image_path):
        return self.image_loader.read_image(image_path)

class ImageAccess():

    def __init__(self, image_loader):
        self.image_loader = image_loader

    def read_image_path(self, image_path):
        return self.image_loader.read_image(image_path)