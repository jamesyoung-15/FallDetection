import argparse

def get_args():
    """ Parse arguments. Returns user/default set arguments """
    parser = argparse.ArgumentParser(description="Pose estimate program")
    # user settings
    parser.add_argument("--src", type=str, default='/dev/video0', help="Video file location eg. /dev/video0")
    parser.add_argument("--show", type=int, default=1, help="Whether to show camera to screen, 0 to hide 1 to show.")
    parser.add_argument("--width", type=int, default=640, help="Input video width.")
    parser.add_argument("--height", type=int, default=480, help="Input video height")
    parser.add_argument("--port", type=int, default=5002, help="Port to run video feed")
    
    args = parser.parse_args()
    return args