# Generated by Django 2.2.4 on 2021-06-04 19:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ResultInformationModel',
            fields=[
                ('rid', models.AutoField(primary_key=True, serialize=False)),
                ('user', models.CharField(max_length=100)),
                ('inputs_given', models.TextField()),
                ('predicted_result', models.TextField()),
            ],
        ),
    ]