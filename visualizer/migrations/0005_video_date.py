# Generated by Django 3.1.2 on 2020-10-30 11:03

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('visualizer', '0004_video_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='video',
            name='date',
            field=models.DateField(default=django.utils.timezone.now, max_length=100),
            preserve_default=False,
        ),
    ]