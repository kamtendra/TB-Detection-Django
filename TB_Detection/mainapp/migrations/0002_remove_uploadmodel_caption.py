# Generated by Django 4.0.4 on 2023-04-17 10:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='uploadmodel',
            name='caption',
        ),
    ]