# Generated by Django 4.0.4 on 2023-04-17 11:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0002_remove_uploadmodel_caption'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadmodel',
            name='date',
            field=models.DateTimeField(auto_now_add=True, default=None),
            preserve_default=False,
        ),
    ]
