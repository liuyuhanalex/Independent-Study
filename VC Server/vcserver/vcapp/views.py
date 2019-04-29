from django.shortcuts import render
from django.shortcuts import redirect
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from . import forms
import os
from vcserver.settings import *
# Create your views here.
from convert.convert import conversion


def index(request):
    my_dict = {}
    return render(request,'index.html',context=my_dict)


def application(request):

    context = {}
    if request.method =='POST':
        #Upload file
        if request.FILES:
            #If user choose a file before upload
            uploaded_file = request.FILES['document']
            fs = FileSystemStorage()
            request.session['file'] =fs.save(os.path.join(SAVEFILE_DIR,uploaded_file.name),uploaded_file)
            context['name'] = uploaded_file.name

        else:
            # User do not choose a file to upload
            # Show prompt message
            context['NoSelectedFile'] = 1
            return render(request,'./application.html',context)

    if request.method == 'GET':

        if (request.GET.get('optradio') is None) and (request.session.get('file') is None):
            # Empty or fisrt time
            return render(request,'./application.html',context)

        if request.GET.get('optradio')==None:
            # User forget to choose a direction
            context['NoDirection']=1
            return render(request,'./application.html',context)

        if request.session.get('file') is None:
            # User do not upload file
            context['Nodatafile']=1
            return render(request,'./application.html',context)

        else:
            print("I am going to transfer"+request.session.get('file'))
            direction = request.GET.get('optradio')
            context['output_file'] = conversion(os.path.join(CONVERT_DIR,"model"),'sf1_tm1.ckpt',
            request.session.get('file'),
            direction,
            os.path.join(CONVERT_DIR,"output"))
            os.remove(request.session['file'])
            request.session['file']=None

    return render(request,'application.html',context)

def applicationEn(request):

    context = {}
    if request.method =='POST':
        #Upload file
        if request.FILES:
            #If user choose a file before upload
            uploaded_file = request.FILES['document']
            fs = FileSystemStorage()
            request.session['file'] =fs.save(os.path.join(SAVEFILE_DIR,uploaded_file.name),uploaded_file)
            context['name'] = uploaded_file.name

        else:
            # User do not choose a file to upload
            # Show prompt message
            context['NoSelectedFile'] = 1
            return render(request,'./en.html',context)

    if request.method == 'GET':

        if (request.GET.get('optradio') is None) and (request.session.get('file') is None):
            # Empty or fisrt time
            return render(request,'./en.html',context)

        if request.GET.get('optradio')==None:
            # User forget to choose a direction
            context['NoDirection']=1
            return render(request,'./en.html',context)

        if request.session.get('file') is None:
            # User do not upload file
            context['Nodatafile']=1
            return render(request,'./en.html',context)

        else:
            print("I am going to transfer"+request.session.get('file'))
            request.session['file'] = None
            print(request.session.get('file'))
            direction = request.GET.get('optradio')
            context['output_file'] = conversion(os.path.join(CONVERT_DIR,"model"),'sf1_tm1.ckpt',
            request.session.get('file'),
            direction,
            os.path.join(CONVERT_DIR,"output"))
            os.remove(request.session['file'])
            request.session['file']=None

    return render(request,'./en.html',context)

def result(request):
    return render(request,'./result.html')

def resultEn(request):
    return render(request,'./resultEn.html')
