from django.urls import path
from . import views

app_name = 'medicareai'

urlpatterns = [
    path('upload/', views.upload_image, name='upload'),
] 