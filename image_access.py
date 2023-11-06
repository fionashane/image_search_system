from abc import ABC, abstractmethod
from matplotlib.image import imread

class ImageLoader(ABC):
    """Abstract base class for image loading."""

    @abstractmethod
    def read_image(self, image_path):
        """
        Read an image from the specified path.

        :param image_path: The path to the image file.
        :return: The loaded image or None if an error occurs.
        """

class ImreadImageLoader(ImageLoader):
    """Image loader using the imread function from a library like OpenCV."""

    def read_image(self, image_path):
        """
        Read an image using the imread function.

        :param image_path: The path to the image file.
        :return: The loaded image or None if the image is not found.
        """
        try:
            return imread(image_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None

class IImageAccess(ABC):
    """Abstract base class for accessing images."""

    @abstractmethod
    def read_image_path(self, image_path):
        """
        Read an image from the specified path.

        :param image_path: The path to the image file.
        :return: The loaded image or None if an error occurs.
        """
        return self.image_loader.read_image(image_path)

class ImageAccess:
    """Image access manager that uses an image loader."""

    def __init__(self, image_loader):
        """
        Initialise the ImageAccess manager with the provided image loader.

        :param image_loader: An instance of an image loader.
        """
        self.image_loader = image_loader

    def read_image_path(self, image_path):
        """
        Read an image from the specified path using the assigned image loader.

        :param image_path: The path to the image file.
        :return: The loaded image or None if an error occurs.
        """
        return self.image_loader.read_image(image_path)
