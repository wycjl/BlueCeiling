import shutil
import os

def remove_dir(dir_name):
	if os.path.exists(dir_name):
		shutil.rmtree(dir_name)
	else:
		print('no such dir:%s'%dir_name)

remove_dir('tile')
remove_dir('out')
remove_dir('temp')
remove_dir('coord')
