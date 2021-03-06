from __future__ import absolute_import
import os
import django
from celery import Celery
from django.conf import settings

# set the default Django settings module for the 'celery' program.
settings_file_name = os.environ.get('SETTINGS_FILE')
if settings_file_name:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings_file_name)
else:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'creditRiskModelling.settings')
django.setup()

app = Celery('creditRiskModelling')

# Using a string here means the worker will not have to pickle the object when using Windows.
app.config_from_object('django.conf:settings')
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)
