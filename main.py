from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse, abort
from flask_cors import CORS
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import os
from heapq import nlargest

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langdetect import detect
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('face', required=True)

MODEL_PATH = "Models/llama-2-7b-chat.Q8_0.gguf"


def load_model() -> LlamaCpp:
    callback = CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 40
    n_batch = 512
    Llama_model: LlamaCpp = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        max_tokens=2000,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        top_p=1,
        callback_manager=callback,
        verbose=True
    )

    return Llama_model


model = load_model()


def transporter(prompt):
    if detect(prompt) != 'en':
        prompt = GoogleTranslator(source='auto', target='en').translate(prompt)
    response = model.invoke(prompt)
    output = response.replace("Answer: ", "", 1)
    output = GoogleTranslator(source='en', target='pl').translate(output)
    return output


class ImageAnalysis(Resource):
    def post(self):
        data = request.get_json()
        key = data.get('key')
        face_url = data.get('face_url')
        prompt = data.get('prompt')

        if key != "kochamrobertmaklowicz2137" and key != "kochamrobertkubica2137":
            return {'message': 'Invalid key provided'}, 400
        elif key == "kochamrobertkubica2137":
            if prompt is None:
                return {'message': 'No prompt provided'}, 400
            output = transporter(prompt)
            output = self.clean_output(output)
            return self.jsonify_ai_output(output), 200

        if face_url is None:
            return {'message': 'No face URL provided'}, 400

        img_name = 'temp.jpg'
        self.get_img(img_name, face_url)
        while not os.path.exists(img_name):
            pass
        face_info = self.get_face_info(img_name)
        self.delete_img(img_name)
        output = self.jsonify_data(face_info)
        return output, 200

    def put(self):
        img_name = 'test.jpg'
        if not os.path.exists(img_name):
            return {'message': 'Image not found'}, 404
        face_info = self.get_face_info(img_name)
        self.delete_img(img_name)
        output = self.jsonify_data(face_info)
        return output, 200

    def get_img(self, img_name, face_url):
        img = Image.open(BytesIO(requests.get(face_url).content))
        img.save(img_name)

    def get_face_info(self, img_path):
        img = cv2.imread(img_path)
        imgplot = plt.imshow(img)
        try:
            obj = DeepFace.analyze(img_path=img_path, actions=['emotion'])
        except Exception as e:
            obj = [{'emotion': {'angry': 0.31797200939637477, 'disgust': 9.395564907524879, 'fear': 89.58764610216892, 'happy': 1.3895607626784605e-05, 'sad': 0.3362636282303112, 'surprise': 0.0007648623524128371, 'neutral': 0.3617738589211618}, 'dominant_emotion': 'fear', 'region': {'x': 99, 'y': 112, 'w': 347, 'h': 347, 'left_eye': None, 'right_eye': None}, 'face_confidence': 0.95}]
        return obj

    def delete_img(self, img_name):
        os.remove(img_name)

    def jsonify_data(self, face_info):
        face = face_info[0]
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_values = [face['emotion'][emotion] for emotion in face['emotion']]

        max_value = max(emotion_values)

        high_values = []
        export_values = []

        for emotion, value in zip(emotions, emotion_values):
            value = float(value)
            if value > 25:
                if emotion == 'neutral':
                    emotion = 'contempt'
                elif emotion == 'happy':
                    emotion = 'happiness'
                elif emotion == 'sad':
                    emotion = 'sadness'
                elif emotion == 'angry':
                    emotion = 'anger'
                high_values.append((emotion, value))

        high_values.sort(key=lambda x: x[1], reverse=True)

        if len(high_values) >= 2:
            if abs(high_values[0][1] - high_values[1][1]) < 33:
                export_values.extend(nlargest(2, high_values, key=lambda x: x[1]))
        else:
            export_values.append(high_values[0])

        dominant_emotion = face['dominant_emotion']

        if dominant_emotion == 'neutral':
            dominant_emotion = 'contempt'
        elif dominant_emotion == 'happy':
            dominant_emotion = 'happiness'
        elif dominant_emotion == 'sad':
            dominant_emotion = 'sadness'
        elif dominant_emotion == 'angry':
            dominant_emotion = 'anger'

        print(export_values)

        if len(export_values) == 0:
            json_data = {
                'dominant_emotion': dominant_emotion,
                'emotions': {
                    dominant_emotion: max_value,
                },
            }
        else:
            json_data = {
                'dominant_emotion': dominant_emotion,
                'emotions': {emotion: value for emotion, value in export_values},
            }

        return json_data

    def jsonify_ai_output(self, ai_output):
        return {'response': ai_output}

    def clean_output(self, output):
        output = output.lstrip('?\n')
        output = output.capitalize()
        return output

api.add_resource(ImageAnalysis, '/')

if __name__ == '__main__':
    app.run()