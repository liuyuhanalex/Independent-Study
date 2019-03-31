from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from . import forms
import os
from vcserver.settings import *
# Create your views here.
from convert.convert import conversion

def index(request):
    my_dict = {'insert_me':"Hello i am from views.py !"}
    return render(request,'index.html',context=my_dict)


def application(request):

    context = {}
    if request.method =='POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        global current_datafile
        current_datafile=fs.save(os.path.join(SAVEFILE_DIR,uploaded_file.name),uploaded_file)
        if current_datafile:
            context['name'] = uploaded_file.name

    if(request.GET.get('btn')):
        direction = request.GET.get('optradio')
        context['start'] = 1
        context['finish'] = 0
        if current_datafile is None:
            context['datafile']=0
        else:
            context['datafile']=1
        context['output_file'] = conversion(os.path.join(CONVERT_DIR,"model"),'sf1_tm1.ckpt',
        current_datafile,
        direction,
        os.path.join(CONVERT_DIR,"output"))
    return render(request,'application.html',context)
