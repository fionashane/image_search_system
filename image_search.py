import click
from image_search_manager import ImageSearchManager

@click.group()
def main():
    pass

#
# Each of the functions below is the entry point of a use case for the command line application.
# Call your code from each of these functions, but do not include all of your code in the functions
# in this file.
#

@main.command()
@click.argument('image_path', type=click.Path(exists=True, dir_okay=False))
def add(image_path):
    """Ingest an image, detect objects, and store the image and detected objects in a CSV file."""
    image_search_manager = ImageSearchManager(object_detection_engine_type='default')
    image_search_manager.ingest_image(image_path)

@main.command()
@click.option('--all/--some', default=True, show_default=True, help='List images that match all/some query terms')
@click.argument('terms', nargs=-1, required=True)
def search(all, terms):
    """Retrieve images based on object types from the CSV file."""
    image_search_manager = ImageSearchManager(object_detection_engine_type='default')
    image_search_manager.retrieve_images_matching_terms(all, terms)

@main.command()
@click.option('--k', default=1, type=click.IntRange(1), show_default=True, help='Number of matches to return')
@click.argument('image_path', type=click.Path(exists=True, dir_okay=False))
def similar(k, image_path):
    """Retrieve similar images based on cosine similarity of object types using data from the CSV file."""
    image_search_manager = ImageSearchManager(object_detection_engine_type='default')
    image_search_manager.retrieve_similar_images(k, image_path)

@main.command()
def list():
    """List all images and their associated object types from the CSV file."""
    image_search_manager = ImageSearchManager(object_detection_engine_type='default')
    image_search_manager.list_images()

if __name__ == '__main__':
    main()
