#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    config.py
# @Author:      Kuro
# @Time:        1/18/2025 11:05 AM
import yaml


class Config:
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
        audio_model = cfg['audio_model']
        ffmpeg_path = cfg['ffmpeg_path']
        threshold = cfg['threshold']
        sample_rate = cfg['sample_rate']
        face_model_name = cfg['face_model_name']
        distance_metric = cfg['distance_metric']
