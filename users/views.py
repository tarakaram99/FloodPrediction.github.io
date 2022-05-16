from django.shortcuts import render,redirect
from  django.views.generic import View,ListView,TemplateView,DetailView
from django.contrib import messages
from users.models import *
from  users.rain_analysis import analysis
from users.Algorithms import *

class AdminLoginCheck(View):
    def post(self,req):
        loginid=req.POST.get("loginid")
        pword=req.POST.get("pword")
        if loginid=="admin" and pword=="admin":
            return render(req,"adminhome.html")
        else:
            messages.success(req,"Invalid Details")
            return redirect('adminlogin')

class User_details_save(View):
    def post(self,request):
        uname=request.POST.get("uname")
        pword=request.POST.get("pword")
        email=request.POST.get("email")
        address=request.POST.get("address")
        status="pending"
        UserRegistrationModel.objects.create(user_name=uname,email=email,password=pword,current_address=address,status=status)
        messages.success(request,"User Registered Successfully")
        return redirect('usersregistarion')


class Admin_logout(View):
    def get(self,req):
        messages.success(req,"Successfully Logged Out")
        return redirect('main')

class User_requests(ListView):
    model = UserRegistrationModel
    queryset = UserRegistrationModel.objects.filter(status="pending")
    template_name = 'adminhome.html'
    context_object_name ="Udata"

class Approve_user(View):
    def get(self,request,id):
        qs=UserRegistrationModel.objects.filter(idno=id)
        qs.update(status="approved")
        return render(request,"adminhome.html",{"data":"User Approved"})
class Decline_user(View):
    def get(self,request,id):
        qs=UserRegistrationModel.objects.filter(idno=id)
        qs.update(status="declined")
        return render(request,"adminhome.html",{"data":"User Declined"})
class User_info(ListView):
    model = UserRegistrationModel
    template_name = 'adminhome.html'
    context_object_name ="uidata"

class User_Login_Validate(View):
    def post(self,request):
        uname=request.POST.get("uname")
        pword=request.POST.get("pword")
        qs=UserRegistrationModel.objects.filter(user_name=uname,password=pword)
        for x in qs:
            status=x.status
            mail=x.email
        if qs and status=="approved":
            request.session["mail"]=mail
            return render(request,"user_home.html",{"name":uname})
        elif qs and (status=="pending" or status=="declined"):
            messages.success(request, "Your details need to approve by admin..please wait until approve..!")
            return redirect('userslogin')
        else:
            messages.success(request,"Invalid Details")
            return redirect('userslogin')


class RainfallAnalysis(View):
    def get(self,req):
        g=analysis()
        mail = req.session["mail"]
        ob = UserRegistrationModel.objects.get(email=mail)
        return render(req, g, {"name": ob.user_name})


class Logistic(View):
    def get(self,req):
        score,fs=logistic()
        score=float(score)*100
        mail = req.session["mail"]
        ob = UserRegistrationModel.objects.get(email=mail)
        return render(req,"user_home.html",{"score":score,"fs":str(fs),"name":ob.user_name})
class Tree(View):
    def get(self,req):
        score,fs=decision()
        score=float(score)*100
        mail = req.session["mail"]
        ob = UserRegistrationModel.objects.get(email=mail)
        return render(req,"user_home.html",{"score":score,"fs":str(fs),"name":ob.user_name})

class Forest(View):
    def get(self,req):
        score,fs=randomforestal()
        score=float(score)*100
        mail = req.session["mail"]
        ob = UserRegistrationModel.objects.get(email=mail)
        return render(req,"user_home.html",{"score":score,"fs":str(fs),"name":ob.user_name})

class Ann_al(View):
    def get(self,req):
        score,fs=ann_al()
        print(score,fs,"============================>+++++++++")
        score=float(score)*100
        mail = req.session["mail"]
        ob = UserRegistrationModel.objects.get(email=mail)
        return render(req,"user_home.html",{"score":score,"fs":str(fs),"name":ob.user_name})


class FloodPrediction(View):
    def get(self,req):
        mail = req.session["mail"]
        ob = UserRegistrationModel.objects.get(email=mail)
        return render(req,"flood_prediction.html",{"name":ob.user_name})


class Userhome(View):
    def get(self,req):
        mail = req.session["mail"]
        ob = UserRegistrationModel.objects.get(email=mail)
        return render(req,"user_home.html",{"name":ob.user_name})


class Predict_result(View):
    def post(self,req):
        rain1=float(req.POST.get("rain1"))
        rain2=float(req.POST.get("rain2"))
        rain3=float(req.POST.get("rain3"))
        rain4=float(req.POST.get("rain4"))
        re=prediction_results(rain1,rain2,rain3,rain4)
        mail = req.session["mail"]
        ob = UserRegistrationModel.objects.get(email=mail)
        if (int(re[0]) == 1):
            result="Result Predicted As:"+str(re[0])+",So there is a possibility of  severe flood"
        else:
            result="Result Predicted As:"+str(re[0])+",So there is No Chance of  severe flood"
        ResultInformationModel.objects.create(user=mail,inputs_given=(str(rain1)+','+str(rain1)+','+str(rain1)+','+str(rain1)+'.'),predicted_result=result)
        return render(req,"flood_prediction.html",{"result":result,"name":ob.user_name})


class Result_info(View):
    def get(self,req):
        mail = req.session["mail"]
        ob = UserRegistrationModel.objects.get(email=mail)
        qs=ResultInformationModel.objects.filter(user=mail)
        return render(req,"result_info.html",{"qs":qs,"name":ob.user_name})


class Delete(View):
    def get(self,req,id):
        ResultInformationModel.objects.filter(rid=id).delete()
        ri=Result_info()
        return ri.get(req)