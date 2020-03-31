import os

from flask import Flask, request, jsonify, render_template

from src.model import generate_caption
from src.utils import base64_to_pil
from src.utils import fix_image_orientation


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "a super secret key")


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/caption', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        img = base64_to_pil(request.json)
        img = fix_image_orientation(img)
        return jsonify(caption=generate_caption(img), status="OK")
    return None
