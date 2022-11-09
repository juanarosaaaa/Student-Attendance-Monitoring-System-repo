from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User

from SAMS.settings import BASE_DIR

import os
from django.contrib.auth.decorators import login_required


# Create your views here.
def login(request):

	return render(request, 'recognition/home.html')

@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		print("admin")
		return render(request, 'recognition/adminDashboard.html')
	else:
		print("not admin")
		return render(request,'recognition/attendance.html')