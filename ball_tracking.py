# USAGE
# pre-recorded video:
	# python ball_tracking.py --video 'Movie on 1-27-21 at 1.02 PM.mov'
	# python ball_tracking.py --video 'Tennis ball parabola.mp4'

# live video feed:
	# python ball_tracking.py
		# use with macos terminal not vs

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the BGR color space, then initialize the
# list of tracked points and X/Y coordinates for graph
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

pts = deque(maxlen=args["buffer"])

run_x = []
rise_y = []

# initialize function variable and R^2 (error) value
funct = None
r_squared = None

# get the starting time and initialize last capture variable
t = None
last_capture = None
print(last_capture)

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	t = time.localtime()

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid and update X/Y coordinates
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		run_x.append(int(M["m10"] / M["m00"]))
		rise_y.append(int(M["m01"] / M["m00"]))

		# update last capture variable
		last_capture = time.strftime("%M.%S", t)

		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
	pts.appendleft(center)

	# initialize point ammount 
	pts_ammount = 0

	# loop over the set of tracked points
	for i in range(1, len(pts)):

		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line, draw
		# the connecting lines, and add to point ammount
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
		pts_ammount += 1
		
	# check if there are enough points to evaluate accurately
	if pts_ammount >= 5:

		# uncomment to create scatter plot
		# plt.scatter(run_x, rise_y)

		# polynomial fit with degree = 2
		funct = np.polyfit(run_x, rise_y, 2)
		model = np.poly1d(funct)

		# create a correlation matrix of the y points previously
		# recorded and the predicted y points, then evaluate the 
		# r^2 (error) value
		correlation_matrix = np.corrcoef(rise_y, np.polyval(funct, run_x))
		correlation_xy = correlation_matrix[0,1]
		r_squared = correlation_xy**2

		# print the r^2 (error) value
		# print(r_squared)

		# uncomment to add fitted polynomial line to scatterplot
		# polyline = np.linspace(1, 600, 1000)
		# plt.scatter(run_x, rise_y)
		# plt.plot(polyline, model(polyline))

		# uncomment if you want to see graph in seperate window
		# plt.show()

		# print the function for the graph
		print(model)

		pts_ammount = 0		

	# create a list of x points to evaluate at and initialize
	# future points list
	future_x_pts = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650)
	future_pts = []

	# draw the parabola when the function has low enough error
	# to be accurate
	if r_squared is not None:

		# loop through the x points, evaluate the corresponding
		# y values and add them to the future_pts list as a touple
		for i in future_x_pts:
			future_y_pts = np.polyval(funct, i)
			new_point = (i, int(future_y_pts))
			future_pts.append(new_point)

		# loop through the future_pts
		for i in range(1, len(future_pts)):

			# if either of the tracked points are None, ignore
			# them
			if future_pts[i - 1] is None or future_pts[i] is None:
				continue

			# otherwise, compute the thickness of the line and draw
			# the connecting lines
			thickness = int(2)
			cv2.line(frame, future_pts[i - 1], future_pts[i], (255, 0, 0), thickness)

	# when five seconds pass since the last capture, the X and
	# Y points will reset for better live video and multiple 
	# object recognition
	if last_capture is not None and float(last_capture) - float(time.strftime("%M.%S", t)) <= -0.02:
		run_x = []
		rise_y = []
		pts = deque(maxlen=args["buffer"])

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
