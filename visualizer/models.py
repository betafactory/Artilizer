from django.db import models
import nltk

nltk.download('stopwords')

# Create your models here.
class Video(models.Model):
    name = models.CharField(max_length=100)
    data = models.CharField(max_length=10000)
    videofile = models.FileField(upload_to='videos/', null=True, verbose_name="")

