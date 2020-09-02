# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:59:14 2019

@author: Fabricio
"""

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
