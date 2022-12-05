from flask import Flask, abort, jsonify, request
import openai
import pandas as pd
from io import StringIO
import tensorflow as tf
from transformers import GPT2Tokenizer
import math
from dotenv import load_dotenv
import os
from flask_cors import cross_origin

'''
pip3 install openai
pip3 install -q git+https://github.com/huggingface/transformers.git
pip3 install tensorflow-macos
pip3 install flask
pip3 install python-dotenv
pip3 install flask-cors
'''

app = Flask(__name__)

load_dotenv()
API_KEY = os.getenv("OPENAPI_SECRET")
PASSWORD = os.getenv("PASSWORD")
openai.api_key = API_KEY

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def getTokenCount(string):
    input_ids = tokenizer.encode(string, return_tensors='tf')
    token_count = input_ids.numpy().size
    return token_count


def completeString(prompt, string):
    query = prompt + "\n\n" + string

    input_token_count = getTokenCount(query)
    # for converting data, there will probably be the
    # same amount of tokens in the output, but the response
    # includes the input AND the output.
    output_token_count = math.ceil((2 * input_token_count) * 1.1)

    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=query,
        max_tokens=output_token_count
    )
    text = response.choices[0].text
    return text


def convertStringToDf(raw):
    markdown = completeString("Turn this data into a Markdown table.", raw)
    csv = completeString(
        "Turn this Markdown table into a CSV, preserving the table headers.", markdown)
    return {
        "markdown": markdown,
        "csv": csv
    }


@app.route("/convert", methods=['POST'])
@cross_origin()
def convert():
    if request.method == 'POST':
        request_data = request.get_json()
        raw_string = request_data.get('raw_string')
        password = request_data.get('password')

        if password == PASSWORD:
            response = convertStringToDf(raw_string)
            return response
        else:
            return abort(403)
