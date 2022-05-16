"""NaturalDisaster URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
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
from users import views
from django.views.generic import TemplateView
from NaturalDisaster import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('',TemplateView.as_view(template_name="main.html"),name='main'),
    path('adminlogin/',TemplateView.as_view(template_name="adminlogin.html"),name='adminlogin'),
    path('adminlogincheck/',views.AdminLoginCheck.as_view(),name='adminlogincheck'),
    path('userslogin/',TemplateView.as_view(template_name="userlogin.html"),name='userslogin'),
    path('usersregistarion/',TemplateView.as_view(template_name="usersregistarion.html"),name='usersregistarion'),
    path('adhome/',TemplateView.as_view(template_name="adminhome.html")),
    path('user_details_save/', views.User_details_save.as_view(), name='user_details_save'),
    path('admin_logout/', views.Admin_logout.as_view(), name='admin_logout'),
    path('user_requests/', views.User_requests.as_view(), name='user_requests'),
    path('approve_user<int:id>/', views.Approve_user.as_view(), name='approve_user'),
    path('decline_user<int:id>/', views.Decline_user.as_view(), name='decline_user'),
    path('user_info/', views.User_info.as_view(), name='user_info'),
    path('RainfallAnalysis/', views.RainfallAnalysis.as_view(), name='RainfallAnalysis'),
    path('login/', views.User_Login_Validate.as_view()),
    path('logistic/', views.Logistic.as_view(), name='logistic'),
    path('tree/', views.Tree.as_view(), name='tree'),
    path('forest/', views.Forest.as_view(), name='forest'),
    path('ann/', views.Ann_al.as_view(), name='ann'),
    path('flood_prediction/', views.FloodPrediction.as_view(), name='flood_prediction'),
    path('userhome/', views.Userhome.as_view(), name='userhome'),
    path('predict_result/', views.Predict_result.as_view(), name='predict_result'),
    path('result_info/', views.Result_info.as_view(), name='result_info'),
    path('delete<int:id>/', views.Delete.as_view(), name='delete'),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
