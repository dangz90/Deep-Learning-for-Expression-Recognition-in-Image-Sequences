import mne
import csv
import glob, os
import time, datetime

import numpy as np
import pandas as pd
import itertools as IT

import matplotlib
import plotly.plotly as py
import matplotlib.pyplot as plt

mne.set_log_level('CRITICAL')

###############################################################################################
#
#Reading eeg files. Each files contain informatiom for one trial (watching 60 images 
# wth 2 followong questions). Split file to events correspoding to watching specific 
#type of imae (see triggers). Assumme delay for video display for 0.5 sec
#
#OUT: "_____.bdf". if first symbol is 2, then the event had dublicate in the trial (can't identify an image)
#
###############################################################################################

triggers = {7:'disgust', 11:'surprise', 15:'sadness', 19:'anger', 23:'fear', 27:'neutral', 31:'happiness',
			103:'disgust', 107:'surprise', 111:'sadness', 115:'anger', 119:'fear', 123:'neutral', 127:'happiness',
			39:'disgust', 43:'surprise', 47:'sadness', 51:'anger', 55:'fear', 59:'neutral', 63:'happiness',
			71:'disgust', 75:'surprise', 79:'sadness', 83:'anger', 87:'fear', 91:'neutral', 95:'happiness'}

def valid(file, experiment):
	print(file)
	flag = 0
	with open(file, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		orders = list(reader)

		d = [row[2::2] for row in orders if experiment in row]
		experiments = [row[0::40] for row in orders]
		
		if flag == 0:
			header = orders[0][1::2]
			flag = 1

		df = pd.DataFrame(d, columns=header)
	csvfile.close()
	return df, experiments

def obtain_variables_from_video(subdir, files, experiment):
	for file in files:
		if 'sound' not in file and 'audio' not in file and file.endswith('.txt'):
			df, experiments = valid(os.path.join(subdir, file), experiment)

			return df, experiments

def split_dataset(data_path, path_to_save):
	if (not os.path.isdir(data_path)):
		assert False
	if (not os.path.isdir(path_to_save)):
		os.mkdir(path_to_save)

	for v in triggers.values():
		if not os.path.exists(os.path.join(path_to_save, v)):
			os.mkdir(os.path.join(path_to_save, v))

	'''
	Obtain Subdirs
	'''
	subdirs = [x[0] for x in os.walk(data_path)][1:]

	for subdir in sorted(subdirs):
		files = os.walk(subdir).next()[2]

		experiment = 'EckmanFaces'
		df, experiments = obtain_variables_from_video(subdir, files, experiment)

		'''
		Obtain .bdf file
		'''
		for f in sorted(files):
			if f.endswith('.bdf'):
				
				print('Reading', os.path.join(subdir, f))

				raw= mne.io.read_raw_edf(os.path.join(subdir, f), stim_channel='Status')
				picks = mne.pick_types(raw.info, eeg=True)
				events = mne.find_events(raw)
				epoch = mne.Epochs(raw, events, picks=picks)
				
				saved = []
				rate =  raw.info['sfreq'] #get sample frequency 


				epoch = mne.Epochs(raw, events, picks=picks)
				data = epoch.get_data()

				'''
				Iterate over events
				'''
				for e, (event, d) in enumerate(zip(events, data)):
					emotions = triggers[event[2]]
					
					# print(d.shape, emotions, experiments[e])


				# 	binary= bin(event[2]) 
				# 	coded = binary[-9:-1] #extract 8 digits with trigger information
				# 	if coded not in ['10000000','01000000']: # not first and second rensponse
				# 		try:
				# 			stimul = triggers[coded[-3:]] # stimul class
				# 			start_time = int(event[0]/rate - 0.5) 
				# 			stop_time = int(event[0]/rate + 2) #video was displaying for 1.5 sec + assuming the delay of trigger for 0.5 sec
				# 			repetition = '1_'
				# 			if coded in saved:
				# 				repetition = '2_' #the second image set has a bag in coding (some triggers codes are nor unique)
				# 			new_name = repetition + stimul+ '_' + str(coded) + '_' + f.split('.')[0] + '.bdf'
				# 			raw.save(path_to_save + stimul + '/' + new_name, tmin = start_time, tmax = stop_time, overwrite=True)
				# 			saved.append(coded)

				# 			print('saved')
				# 		except:
				# 			error_out.write('file: ' + f + ' ,' + binary + ' ,'  + coded + '\n')

def plot_channels(file, name):
	raw = mne.io.read_raw_edf(file)
	# raw.plot(block=True)
	event = mne.find_events(raw, stim_channel='Status')
	picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
	names = raw.info['ch_names']
	epoch = mne.Epochs(raw, event, tmax=1, picks=picks)
	# evoked = epoch.average()
	data = epoch.get_data()
	length = data.shape[2]

	for pick in range(data.shape[1]):
		y = np.reshape(data[0,pick, :], (length))

		plt.subplot(15,41,pick+1)
		plt.plot(y)
		# plt.title(names[pick])
		# plt.xlabel('time')
		# plt.ylabel('Hz')
	plt.subplots_adjust(hspace=1)
	plt.show()
	plt.savefig(name)
	plt.close()

def plot_filtered(file):
	raw = mne.io.read_raw_fif(file)
	event = mne.find_events(raw)
	picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
	epoch = mne.Epochs(raw, event, tmax=1, picks=[5])
	evoked = epoch.average()
	evoked.plot()
	data = epoch.get_data()
	plt.show()

def check_original_files(path):
	uniq = []
	s = {'a' : 0, 'b' : 1}
	files = os.listdir(path)
	trials = {}
	for f in files:
		id = f[4:]
		image_set = s[f[2]]

		if id not in uniq:
			uniq.append(id)
			trials[id] = [0,0]
		try:
			trials[id][image_set-1] += 1
		except:
			print(f)
	print('Number of subjects: ', len(uniq))
	print('Number of original files: ', len(files))
	print(trials)

def save_as_numpy(path_to_dataset, file):
	raw = mne.io.read_raw_edf( file )

	event = mne.find_events(raw, stim_channel='Status')

	picks = mne.pick_types(raw.info, eeg=True)

	epoch = mne.Epochs(raw, event, picks=picks)

	data = epoch.get_data()
	# if data.shape[0] != 1:
	# 	print(file, event, data.shape)
		
	outfile = os.path.splitext(file)[0] + '.npy'
	np.save(outfile, data )

'''
Main: EGG
'''
if __name__ == "__main__":
	print 'Starting:', time.ctime(), '\n' 

	import argparse

	parser = argparse.ArgumentParser(description='Change folder order for OULU-CASIA dataset')
	parser.add_argument('-dataset', 	type=str, default='Blocked_raw', 
											help='Name of the dataset to use')
	parser.add_argument('-participant', type=str, default='', 
											help='Name of the participant')	
	args = parser.parse_args()

	path_to_dataset = os.path.join('..', '_Dataset', args.dataset, args.participant)
	path_to_save = os.path.join('..', '_Dataset', 'Blocked')

	"""
	Convert EGG file into numpy array.
	"""
	split_dataset(path_to_dataset, path_to_save)

	# for file in files:
	# 	save_as_numpy( path_to_dataset, file )
		# split_dataset(path_to_dataset, path_to_save)

	print '\nEnding:', time.ctime()	