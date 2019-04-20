from django.conf.urls import url
from vcapp import views
from django.urls import path

urlpatterns = [
    # path('',views.index,name='index'),
    path('',views.application,name='application'),
    path('en.html',views.applicationEn,name='application'),
    path('application.html',views.application,name='application'),
    path('result.html',views.result,name='application'),
]
