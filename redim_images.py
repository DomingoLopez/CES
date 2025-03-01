"""
SCRIPT PARA REDIMENSIONAR LAS IMÁGENES A LA HORA DE PROCESAR Y PASARLAS A PDF PARA REALIZAR LA TAREA MANUAL DE ANOTACIÓN.
AL HACER ESTO AHORRAMOS MUCHO TIEMPO DE PROCESADO AL EMBEBERLAS EN EL PDF
"""

from src.utils.image_loader import ImageLoader
from PIL import Image as PILImage

def load_images(path) -> list:
    # Finding images
    # image_loader = ImageLoader(folder="./data/Small_Data")
    image_loader = ImageLoader(folder=path)
    images = image_loader.find_images()
    return images


def redimensionar_imagen(input_path, output_path, max_width=80, max_heigth=80):
    with PILImage.open(input_path) as img:
        img.thumbnail((max_width, max_heigth))
        img.save(output_path)



if __name__ == "__main__":
    
    input_path ="./data/flickr/flickr_validated_imgs_7000/"
    images = load_images(input_path)    
    output_path = "./data/flickr/flickr_redimensioned_imgs_7000/"
    for i in images:
        redimensionar_imagen(str(i),output_path+str(i).split("/")[-1])

