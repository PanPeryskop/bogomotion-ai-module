from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse, abort
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import os
from heapq import nlargest

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('face', required=True)


class ImageAnalysis(Resource):
    def post(self):
        data = request.get_json()
        key = data.get('key')
        face_url = data.get('face_url')

        if key != "kochamrobetmaklowicz2137":
            return {'message': 'Invalid key provided'}, 400

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
            obj = [{'emotion': {'angry': 0.31797200939637477, 'disgust': 9.395564907524879, 'fear': 0.3617738589211618, 'happy': 1.3895607626784605e-05, 'sad': 0.3362636282303112, 'surprise': 0.0007648623524128371, 'neutral': 89.58764610216892}, 'dominant_emotion': 'neutral', 'region': {'x': 99, 'y': 112, 'w': 347, 'h': 347, 'left_eye': None, 'right_eye': None}, 'face_confidence': 0.95}]
        return obj

    def delete_img(self, img_name):
        os.remove(img_name)

    def jsonify_data(self, face_info):
        face = face_info[0]
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_values = [face['emotion'][emotion] for emotion in face['emotion']]

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
            }
        else:
            json_data = {
                'dominant_emotion': dominant_emotion,
                'emotions': {emotion: value for emotion, value in export_values},
            }

        return json_data


api.add_resource(ImageAnalysis, '/')

if __name__ == '__main__':
    app.run()