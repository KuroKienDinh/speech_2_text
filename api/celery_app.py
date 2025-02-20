#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    celery_app.py
# @Author:      Kuro
# @Time:        2/18/2025 9:04 PM
import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery('worker', broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(worker_concurrency=1, worker_max_tasks_per_child=10, task_track_started=True, result_expires=1800)
# Autodiscover tasks in the 'api' package
celery_app.autodiscover_tasks(['api'])
