# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:46:23 2019

@author: Gabriela
"""

class Sample:
	def __init__(self, image, eye_descript, eye_coord):
		self.image = image
		self.eye_descript = eye_descript
		self.eye_coord = eye_coord
    
    
	def __str__(self):
		return  str([descript for descript in self.eye_descript]) + "\n"