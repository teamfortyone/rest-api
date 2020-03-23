import re
import base64
from io import BytesIO
from pickle import load
from PIL import Image, ExifTags


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = BytesIO(base64.b64decode(image_data))
    return pil_image
    

def fix_image_orientation(img):
    try:
        image = Image.open(img)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        return buffered

    except Exception as e:
        # In case the image doesn't have any EXIF data, return the image as is
        print("Exception: ", e)
        return img
     

def load_pickle(path):
    with open(path, "rb") as f:
        return load(f)
