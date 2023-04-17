from django.shortcuts import render
from django.core.files.storage import default_storage
from .utils import predict_label

def home(request):
    if request.method == 'POST':
        image=request.FILES.get("images")
        file_name = default_storage.save(str(image), image)
        path= r"C:\Users\kamte\OneDrive\Documents\Desktop\TB Detection\TB_Detection\public\static"
        path+=r"/"
        path+=str(image)
        result=predict_label(path)
        print(result)
        return render(request,'index.html', {"result":result})
    return render(request,'index.html') 

