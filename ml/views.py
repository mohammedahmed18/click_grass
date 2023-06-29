from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from rest_framework.decorators import api_view
from PIL import Image
import os
from rest_framework import status

modelPath = os.path.join(os.getcwd(), "\models\egrot.h5")


class PredictImage(APIView):
    def post(self, request, format=None):
        # Load the trained model
        model = load_model(modelPath)

        # Get the image data from the request
        img_data = request.FILES['image']

        # Preprocess the image
        img = image.load_img(img_data, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make the prediction
        predictions = model.predict(x)
        results = {
            'class_1': str(predictions[0][0]),
            'class_2': str(predictions[0][1]),
            'class_3': str(predictions[0][2]),
        }

        # Return the results as a JSON response
        return Response(results)


@api_view(['POST'])
def predict(request):
    model = load_model(modelPath)
    image_file = request.FILES['image']
    image = Image.open(image_file).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    max_prediction = np.argmax(prediction)

    if max_prediction == 1:
        result = 'unhealthy'
    else:
        result = 'healthy'
    print(f"<{max_prediction}>")
    return Response({'result': result})


# # Load the ML model
# model = load_model(r'F:\full-stack-python\graduation\graduation_project\src\ml\models\dataset.h5')
# # Assuming your model expects input images of size 224x224
# image_size = (224, 224)

# def make_predictions(image):
#     image = image.resize(image_size)
#     image = np.array(image) / 255.0  # Normalize the image
#     image = image.reshape(1, 224, 224, 3)
#     preds = model.predict(image)
#     return preds

# def fresh_or_rotten(image):
#     preds = make_predictions(image)
#     print(preds)
#     if preds <= 0.5:
#         return "It's Fresh! Eat ahead."
#     else:
#         return "It's Rotten. I won't recommend it."

# class FruitClassificationView(APIView):
#     def post(self, request):
#         # Check if an image file was provided
#         if 'image' not in request.FILES:
#             return Response({'error': 'No image file provided'}, status=status.HTTP_400_BAD_REQUEST)

#         # Load the image
#         image_file = request.FILES['image']
#         image = Image.open(image_file)
#         print(dir(image_file))
#         # Get the fresh/rotten prediction
#         prediction_label = fresh_or_rotten(image)
#         prediction2=make_predictions(image)
#         return Response({'prediction': prediction_label,'pre2':prediction2}, status=status.HTTP_200_OK)



from rest_framework.views import APIView
from rest_framework.response import Response
from keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image


model = load_model(modelPath)

class FruitClassificationAPI(APIView):
    def post(self, request):
        image_file = request.FILES['image']
        image = Image.open(image_file)
        image = image.resize((224, 224))  # Resize the image to the desired size
        image = image_utils.img_to_array(image)
        image = image.reshape(1, 224, 224, 3)
        image = preprocess_input(image)
        preds = model.predict(image)
        if preds <= 0.5:
            result = "It's Fresh! Eat ahead."
        else:
            result = "It's Rotten. I don't recommend!"
        return Response({"result": result})
