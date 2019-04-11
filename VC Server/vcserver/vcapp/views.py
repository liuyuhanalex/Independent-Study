from django.shortcuts import render
from django.shortcuts import redirect
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from . import forms
import os
from vcserver.settings import *
# Create your views here.
from convert.convert import conversion

current_datafile = None
redirectB = False

def index(request):
    my_dict = {'insert_me':"Hello i am from views.py !"}
    return render(request,'index.html',context=my_dict)


def application(request):

    context = {}
    context['Nodatafile'] = 0
    global redirectB
    if redirectB ==True:
        print("redict！")
        context['Nodatafile']=1
        redirectB = False
        return render(request,'application.html',context)

    if request.method =='POST':
        if request.FILES:
            uploaded_file = request.FILES['document']
            fs = FileSystemStorage()
            global current_datafile
            current_datafile=fs.save(os.path.join(SAVEFILE_DIR,uploaded_file.name),uploaded_file)
            if current_datafile:
                context['name'] = uploaded_file.name
        else:
            redirectB = True
            context['Error2'] = 0
            return redirect('/')

    if(request.GET.get('btn')):
        print(current_datafile)
        direction = request.GET.get('optradio')
        if current_datafile is None:
            # User do not upload file
            redirectB = True
            context['Error2']=1
            print("Error2")
            return redirect('/')
        else:
            redirectB = False
            context['Nodatafile']=0
            context['output_file'] = conversion(os.path.join(CONVERT_DIR,"model"),'sf1_tm1.ckpt',
            current_datafile,
            direction,
            os.path.join(CONVERT_DIR,"output"))
            current_datafile = None
    return render(request,'application.html',context)
