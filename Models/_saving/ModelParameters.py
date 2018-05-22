import os
import six, csv
from keras.callbacks import Callback

# Class for writing the model parameters
class ModelParameters(Callback):
	def __init__(self, filename, row_dict, separator=',', append=False):
		self.sep = separator
		self.row_dict = row_dict		
		self.filename = filename
		self.append = append
		self.writer = None
		self.keys = row_dict.keys()
		self.append_header = True
		self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
		super(ModelParameters, self).__init__()

	def on_train_begin(self, logs=None):
		if self.append:
			if os.path.exists(self.filename):
				with open(self.filename, 'r' + self.file_flags) as f:
					self.append_header = not bool(len(f.readline()))
			self.csv_file = open(self.filename, 'a' + self.file_flags)
		else:
			self.csv_file = open(self.filename, 'w' + self.file_flags)

		if self.keys is None:
			self.keys = ['dataset', 'partition', 'loss', 'lr', 'date']

		if not self.writer:
					class CustomDialect(csv.excel):
						delimiter = self.sep

					self.writer = csv.DictWriter(self.csv_file,
												 fieldnames= self.keys, dialect=CustomDialect)
					if self.append_header:
						self.writer.writeheader()

		self.writer.writerow(self.row_dict)
		self.csv_file.flush()        

		self.csv_file.close()
		self.writer = None 