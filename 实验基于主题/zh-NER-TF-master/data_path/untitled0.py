# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 19:18:05 2019

@author: Gu Yi
"""
import os
data = []
with open('train_data', encoding='utf-8') as fr:
    lines = fr.readlines()
sent_, tag_ = [], []
for line in lines:
    if line != '\n':
        [char, label] = line.strip().split()
        sent_.append(char)
        tag_.append(label)
    else:
        data.append((sent_, tag_))
        sent_, tag_ = [], []
        print(lines)


