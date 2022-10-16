import jetson.utils
import argparse
import sys


# create video sources & outputs
input = jetson.utils.videoSource('csi://0')
output = jetson.utils.videoOutput()

# capture frames until user exits
while output.IsStreaming():
	image = input.Capture(format='rgb8')
	output.Render(image)
	output.SetStatus("Video Viewer | {:d}x{:d} | {:.1f} FPS".format(image.width, image.height, output.GetFrameRate()))
