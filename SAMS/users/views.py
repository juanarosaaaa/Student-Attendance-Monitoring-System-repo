from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages

# Create your views here.

@login_required
def registerFacilitator(request):
	if request.user.username!='admin':
		return redirect('not-authorized')
	if request.method=='POST':
		form=UserCreationForm(request.POST)
		if form.is_valid():
			form.save() ###add user to database
			messages.success(request, f'Facilitator registered successfully!')
			return redirect('dashboard')
	else:
		form=UserCreationForm()
	return render(request,'users/registerFacilitator.html', {'form' : form})

@login_required
def addStudent(request):
	if request.user.username!='admin':
		return redirect('not-authorized')
	if request.method=='POST':
		form=UserCreationForm(request.POST)
		if form.is_valid():
			form.save() ###add user to database
			messages.success(request, f'Student added successfully!')
			return redirect('dashboard')
	else:
		form=UserCreationForm()
	return render(request,'users/addStudent.html', {'form' : form})