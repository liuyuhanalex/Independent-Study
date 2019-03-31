from django.conf.urls import url
from vcapp import views
from django.urls import path

urlpatterns = [
    # path('',views.index,name='index'),
    path('',views.application,name='applacation'),
]
