"""SAMS URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from users import views as users_views
from recognition import views as recog_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path('register/',users_views.registerFacilitator, name='register-facilitator'),
    path('add-student/', users_views.addStudent, name = 'add-student'),
    path('home/', recog_views.home, name='home'),
    path('login/',auth_views.LoginView.as_view(template_name='users/login.html'),name='login'),
    path('logout/',auth_views.LogoutView.as_view(template_name='recognition/home.html'),name='logout'),
    path('dashboard/', recog_views.dashboard, name = 'dashboard'),
    path('not-authorized/', recog_views.not_authorized, name='not-authorized'),
    path('attendance/', recog_views.attendance,name = 'attendance'),
    path('attendanceIn/', recog_views.attendanceIn,name='attendanceIn'),
    path('attendanceOut/', recog_views.attendanceOut,name='attendanceOut'),
    path('train/', recog_views.train, name='train'),
    path('add-photos', recog_views.add_photos, name='add-photos'),
    path('about-us/', recog_views.aboutUs, name = 'about-us')
]
