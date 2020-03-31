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

@app.route('/caption', methods=['POST'])
def upload_web():
    img = base64_to_pil(request.json)
    img = fix_image_orientation(img)
    greedy, beam_k3, beam_k5 = generate_caption(img)
    return jsonify(greedy=greedy, beam_k3=beam_k3, beam_k5=beam_k5, status="OK")

@app.route('/android', methods=['POST'])
def upload_android():
    img = base64_to_pil(request.data, substitute_prefix=False)
    img = fix_image_orientation(img)
    greedy, beam_k3, beam_k5 = generate_caption(img)
    return jsonify(greedy=greedy, beam_k3=beam_k3, beam_k5=beam_k5, status="OK")
