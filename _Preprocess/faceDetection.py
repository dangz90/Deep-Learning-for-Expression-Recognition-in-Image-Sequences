import numpy as np
import cv2
import os

'''
FUNCTION: obtain a list of files from a directory
'''
def list_files(dir):                                                                                                  
	r = []
	r_subdir = []
	r_file = []
	subdirs = [x[0] for x in os.walk(dir)]
	for subdir in subdirs:
		files = os.walk(subdir).__next__()[2]
		if (len(files) > 0):
			for file in files:
				if file.endswith('.jpg'):
					r.append(subdir + "/" + file)
					r_subdir.append(subdir)
					r_file.append(file)
	return r, r_subdir, r_file

'''
MAIN: Detect face patches and save them as jpg
'''

# Get list of files in dataset
r, r_subdir, r_file = list_files('dataset\\')

print(r)

# # Obtain XML with Haar Features
# face_cascade = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

# # Cycle through the files
# for subdir, file in zip(r_subdir, r_file):
# 	print(subdir +'\\'+ file)

# 	img = cv2.imread(subdir +'\\'+ file)
# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 	# Classifier for detecting faces
# 	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# 	for (x,y,w,h) in faces:
# 		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# 		# Create the face patch
# 		detected_color = img[y:y+h, x:x+w]
# 		detected_color = cv2.resize(detected_color, (224, 224))
# 		# Save patch
# 		split_subdir = subdir.split('\\')
# 		cv2.imwrite(os.path.join('patches\\'+split_subdir[1]+'\\'+split_subdir[2]+'\\', file + '.jpg'), detected_color)     # save frame as JPEG file