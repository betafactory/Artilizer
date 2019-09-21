from django.db import models

# Create your models here.
class Video(models.Model):
    data = models.CharField(max_length=10000)
    videofile = models.FileField(upload_to='videos/', null=True, verbose_name="")

    def __str__(self):
        return self.name + ": " + str(self.videofile)
