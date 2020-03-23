# Image Captioning REST API

## Requirements
Python 3.6 or 3.7

## Setup

1. `git clone https://github.com/teamfortyone/rest-api`
2. Create a virtualenv and activate it
3. `pip install -r requirements.txt`

## Usage

Navigate to root of the project directory (`.../rest-api/`)

Run the server using below command:
`gunicorn -b :5000 src.app:app`
