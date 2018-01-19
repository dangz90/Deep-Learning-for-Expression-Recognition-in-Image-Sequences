import os
import mne
import numpy as np

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

def split_dataset(data_path, path_to_save):
	error_out = open('errors.txt','w')
	if (not os.path.isdir(data_path)):
		assert False
	if (not os.path.isdir(path_to_save)):
		os.mkdir(path_to_save)
	triggers = {'001':'ntr', '010':'sex','100':'veri','011':'posM', '101':'negM'}
	for v in triggers.values():
		os.mkdir(path_to_save+v)
	for f in os.listdir(data_path):
		raw= mne.io.read_raw_edf(data_path+ f, preload=False)
		events = mne.find_events(raw)
		saved = []
		rate =  raw.info['sfreq'] #get sample frequency 
		for event in events:
			binary= bin(event[2]) 
			coded = binary[-9:-1] #extract 8 digits with trigger information
			if coded not in ['10000000','01000000']: # not first and second rensponse
				try:
					stimul = triggers[coded[-3:]] # stimul class
					start_time = int(event[0]/rate - 0.5) 
					stop_time = int(event[0]/rate + 2) #video was displaying for 1.5 sec + assuming the delay of trigger for 0.5 sec
					repetition = '1_'
					if coded in saved:
						repetition = '2_' #the second image set has a bag in coding (some triggers codes are nor unique)
					new_name = repetition + stimul+ '_' + str(coded) + '_' + f.split('.')[0] + '.bdf'
					raw.save(path_to_save + stimul + '/' + new_name, tmin = start_time, tmax = stop_time, overwrite=True)
					saved.append(coded)
				except:
					error_out.write('file: ' + f + ' ,' + binary + ' ,'  + coded + '\n')
	error_out.close()


def plot_channels(file, name):
	raw = mne.io.read_raw_fif(file)
	# raw.plot(block=True)
	event = mne.find_events(raw)
	picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
	names = raw.info['ch_names']
	epoch = mne.Epochs(raw, event, tmax=1, picks=picks)
	# evoked = epoch.average()
	data = epoch.get_data()
	length = data.shape[2]
	for pick in range(data.shape[1]):
		y = np.reshape(data[:,pick, :], (length))
		plt.subplot(8,6,pick+1)
		plt.plot(y)
		# plt.title(names[pick])
		# plt.xlabel('time')
		# plt.ylabel('Hz')
	plt.subplots_adjust(hspace=1)
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

def save_as_numpy(path, where_to_save):
	for sub in os.listdir(path):
		for file in os.listdir(os.path.join(path, sub)):

			print(os.path.join(path, sub, file))

			raw = mne.io.read_raw_edf( os.path.join(path, sub, file) )

			event = mne.find_events(raw)
			picks = mne.pick_types(raw.info, eeg=True)
			epoch = mne.Epochs(raw, event, picks=picks)
			data = epoch.get_data()
			if data.shape[0] != 1:
				print(sub, file, event, data.shape)
				
			np.save(where_to_save + sub +'/'+ file.split('.')[0] + '.npy', np.reshape(data, (data.shape[1], data.shape[2])))

'''
___init___
'''

# check_original_files(os.path.join('dataset', 'EGG'))

# plot_filtered('dataset\EGG\h1a3ki42.bdf')

save_as_numpy( os.path.join('dataset'), os.path.join('dataset','numpy_data') )

print('\nEND')