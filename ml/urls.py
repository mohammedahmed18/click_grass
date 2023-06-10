from django.urls import path
from .views import PredictImage,predict,FruitClassificationAPI

urlpatterns = [
    path('predict1/', predict, name='predict_image'),
    path('classify-fruit/', FruitClassificationAPI.as_view(), name='classify'),
    # path('predict2/', predict2, name='predict2_image'),
    
    
]