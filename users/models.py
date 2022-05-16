from django.db import models

class UserRegistrationModel(models.Model):
    idno=models.AutoField(primary_key=True)
    user_name=models.CharField(max_length=100,unique=True)
    current_address=models.CharField(max_length=100)
    email=models.EmailField(unique=True)
    password=models.CharField(max_length=100)
    status=models.CharField(max_length=50,default=True)


class ResultInformationModel(models.Model):
    rid=models.AutoField(primary_key=True)
    user=models.CharField(max_length=100)
    inputs_given=models.TextField()
    predicted_result=models.TextField()