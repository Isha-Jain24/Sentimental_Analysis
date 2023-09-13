from django.shortcuts import render
import subprocess

def index(request):
    result = None
    if request.method == 'POST':
        # Get the YouTube video link from the form
        video_link = request.POST['video_link']
        command = f"python sentiment_new.py {video_link}"
        result = subprocess.getoutput(command)

    return render(request, 'index.html', {'result': result})
