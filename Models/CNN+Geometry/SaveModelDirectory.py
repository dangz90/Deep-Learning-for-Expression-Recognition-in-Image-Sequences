import os
import sys
import tensorflow as tf

def create_path(__model__, dataset):

	''' 
	Directory base folder
	'''
	export_path_base = os.path.dirname(sys.argv[0])

	'''
	Define model version
	'''

	# Obtain most recent version
	export_base_path = os.path.join(export_path_base, '_versions', dataset, __model__)

	# Verify a folder already exists and assign version number
	if os.path.isdir(export_base_path):
		all_subdirs = [int(d) for d in os.listdir(export_base_path) if os.path.isdir(export_base_path)]
		__version__ = int(max(all_subdirs)) + 1
	else:
		__version__ = 1

	# Assign FLAG version number
	tf.app.flags.DEFINE_integer('model_version', __version__, 'Version number of the model.')
	FLAGS = tf.app.flags.FLAGS

	'''
	Create version folder
	'''
	export_path = os.path.join(	export_base_path,
								str(FLAGS.model_version))

	print('Exporting model to', export_path)
	builder = tf.saved_model.builder.SavedModelBuilder(export_path)

	return export_path

if __name__ == '__main__':
    create_path()	