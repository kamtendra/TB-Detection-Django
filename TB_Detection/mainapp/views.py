from django.shortcuts import render
from django.core.files.storage import default_storage
from .utils import predict_label, predict_label1, predict_label2, predict_label3

def home(request):
    if request.method == 'POST':
        image=request.FILES.get("images")
        file_name = default_storage.save(str(image), image)
        path= r"C:\Users\kamte\OneDrive\Documents\Desktop\TB Detection\TB_Detection\public\static"
        path+=r"/"
        path+=str(image)
        ans=[]
        ans.append(predict_label(path))
        ans.append(predict_label1(path))
        ans.append(predict_label2(path))
        ans.append(predict_label3(path))
        print(ans)
        if ans.count('Tuberculosis'):
            result = 'Tuberculosis'
        else:
            result = 'Normal'    
        return render(request,'index.html', {"result":result})
    return render(request,'index.html') 

