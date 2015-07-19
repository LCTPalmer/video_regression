#setting up ipython profile
to make the HOF ipython profile run
	ipython profile create hof
at the terminal

copy the startup folder stored in ipython_profile_files 
to the relevant place in new system

change directory in 01_setdirs.py in startup folder to 
the relevant location (location of this file/code!)

#tested with opencv 2.4.10
install opencv
copy opencv/python/x86-64/cv2.pyd to site-packages library

#to work with video
copy opencv/sources/3rdparty/ffmpeg/opencv_ffmpeg.dll to 
python installation directiory (here its C:/Anaconda), and rename
to opencv_ffmpegXYZ.dll where opencv is version X.Y.Z (here 2410)