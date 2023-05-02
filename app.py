import json
import subprocess
from flask import Flask, jsonify
from waitress import serve

app = Flask(__name__)

@app.route('/')
def index():
    # Call the other Python script and capture its output
    result = subprocess.check_output(['python', 'ABSA.py'])

    # Convert the output to a string
    result_str = result.decode('utf-8')

    # Parse the string as JSON
    result_json = json.loads(result_str)

    # Return the JSON as the response to the webpage
    return jsonify(result_json)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)