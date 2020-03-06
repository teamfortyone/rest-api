import re
import base64
from io import BytesIO
from pickle import load


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = BytesIO(base64.b64decode(image_data))
    return pil_image


def load_pickle(path):
    with open(path, "rb") as f:
        return load(f)
