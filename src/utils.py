from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from datetime import datetime
import csv


class StatusPrinter:

	def __init__(self):

		self.elements = {}

	def add_bar(self, name: str, statement: str, max_value: int,
				num_blocks: int = 30, value: int = 0, bold: bool = False):

		self.elements[name] = ("\033[1m"*bold+statement+"\033[0m", ProgressBar(max_value, num_blocks, value), "bar")

	def add_counter(self, name: str, statement: str, max_value: int, value: int = 0, bold: bool = False):

		self.elements[name] = ("\033[1m"*bold+statement+"\033[0m", ProgressBar(max_value, 1, value), "counter")

	def increment_and_print(self, name: str):

		if self.elements[name][2] == "counter":
			self.elements[name][1].increment()
			print(self.elements[name][0]+": {}/{}".format(self.elements[name][1].value,
														  self.elements[name][1].max_value))

		elif self.elements[name][2] == "bar":
			prev_b = self.elements[name][1].blocks
			self.elements[name][1].increment()
			curr_b = self.elements[name][1].blocks

			if (self.elements[name][1].value == 1 or
				prev_b != curr_b or
				self.elements[name][1].value == self.elements[name][1].max_value):

					print("  "+str(self.elements[name][1]), end="\r")

	def print_statement(self, name: str):

		print(self.elements[name][0]+":")

	def reset_element(self, name: str):

		self.elements[name][1].reset()

	def done_element(self, name: str):

		if self.elements[name][2] == "counter":
			print("\033[32m"+self.elements[name][0]+": {}/{}".format(self.elements[name][1].value,
				  self.elements[name][1].max_value)+"\033[0m")

		elif self.elements[name][2] == "bar":
			self.elements[name][1].set_value(self.elements[name][1].max_value)
			print("  "+str(self.elements[name][1]), end="\n")


class ProgressBar:

	def __init__(self, max_value: int, num_blocks: int, value: int = 0):

		self.max_value = max_value
		self.num_blocks = num_blocks
		self.block_size = max_value / num_blocks
		self.value = value
		self.blocks = int(self.value / self.block_size)

	def increment(self):
		self.value += 1
		self.blocks = int(self.value / self.block_size)

	def set_value(self, value):
		self.value = value
		self.blocks = int(self.value / self.block_size)

	def reset(self):
		self.value = 0
		self.blocks = 0

	def __str__(self):
		return "|"+"█"*self.blocks+" "*(self.num_blocks - self.blocks)+"|"


class Parameters:

	def __init__(self, path: str = None, is_help: bool = False):

		self.fixed = False

		if not is_help:
			self.help = Parameters(is_help=True)

		if path:
			self.load_from_file(path)

	def fix(self):
		self.fixed = True
		self.help.fixed = True

	def load_from_file(self, path: str):

		with open(path) as f:
			data = json.load(f)
		for key in data.keys():
			setattr(self, key, data[key][0])
			setattr(self.help, key, data[key][1])

	def overload(self, other, ignore=[]):
		for arg in vars(other):
			if arg not in ignore and getattr(other, arg):
				setattr(self, arg, getattr(other, arg))

	def as_dict(self):
		return {key: val for key, val in self.__dict__.items() if (key != "fixed" and key != "help")}

	def __setattr__(self, name: str, value):
		if name != 'fixed' and self.fixed and name in self.__dict__:
			raise TypeError("Parameters are already fixed. Not allowed to change them.")

		else:
			self.__dict__[name] = value

	def toTable(self):
		out = "<table> <thead> <tr> <th> Parameter </th> <th> Value </th> </tr> </thead> <tbody>"
		for name in self.__dict__.keys():
			if name != 'help' and name != 'fixed':
				out += " <tr> <td> "+name+" </td> <td> {} </td> </tr> ".format(getattr(self, name))
		out += "</tbody> </table>"
		return out

	def __str__(self):
		size_name = max([len(name) for name in self.__dict__.keys() if name != 'help' and name != 'fixed'])+2
		size_val = max([len(str(val)) for name, val in self.__dict__.items() if name != 'help' and name != 'fixed'])+2
		out = " "+"_"*(size_name + size_val+1)+" \n"
		out += "|"+"Parameter".center(size_name)+"|"+"Value".center(size_val)+"|\n"
		out += " "+"-"*(size_name + size_val+1)+" \n"
		for name in self.__dict__.keys():
			if name != 'help' and name != 'fixed':
				out += "|"+name.center(size_name)+"|"+str(getattr(self, name)).center(size_val)+"|\n"
		return out

	def __repr__(self):
		return str(self)

class AverageValueMeter:

	def __init__(self, value: int = 0, counter: int = 0):
		self.value = value
		self.counter = counter

	def __add__(self, other):

		if isinstance(other, float) or isinstance(other, int):
			value = self.value + (other - self.value)/(self.counter+1)
			return AverageValueMeter(value=value, counter=self.counter+1)

	def __str__(self):
		return str(self.value)

	def __repr__(self):
		return str(self)

	def reset(self):
		self.value = 0
		self.counter = 0


class CSVLogger:

	def __init__(self, filepath, header=['data'], log_time=False):
		
		self.filepath = filepath
		self.log_time = log_time		

		if self.log_time:
			header.append('time')

		with open(filepath, 'w', encoding='UTF8') as f:
			writer = csv.writer(f)
			writer.writerow(header)

	def log(self, data):
		
		if self.log_time:
			now = datetime.now()
			data.append(now.strftime("%H:%M:%S"))

		with open(self.filepath, 'a', encoding='UTF8') as f:
			writer = csv.writer(f)
			writer.writerow(data)
