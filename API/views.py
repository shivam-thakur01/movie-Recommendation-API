from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse

import numpy as np
import pandas as pd
import tensorflow as tf
from .test import support


@api_view(['post'])
def reccommendationModel(request):
    # print(request.data["rating"])
    Recommendations=support(request.data["rating"],request.data["number_of_movie"])
    movies=Recommendations['Recommendations'].numpy().tolist()
    # print(movies)
    return Response({'status':200,'prediction':'success',"data":movies})

# {"rating":{"Maya Lin: A Strong Clear Vision (1994)":4,"Twister (1996)":5,"Godfather: Part II, The (1974)":2,"Good, The Bad and The Ugly, The (1966)":3,"Groundhog Day (1993)":4}}
# {"rating":{"4":5,"100":2,"50":4,"800":3,"700":4,"300":1,"200":4,"1":4,"2":5,"3":5,"5":4,"51":3,"27":3,"601":5,"101":1,"102":2,"105":3},"number_of_movie":3}