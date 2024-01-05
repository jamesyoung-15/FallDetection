from ultralytics import YOLO
import argparse
import cv2
import utils
import my_defs
par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
par.add_argument('-i', '--image', default=None,  help='Source of camera or video file path. Eg. /dev/video0 or ./videos/myvideo.mp4')
args = par.parse_args()
# print(args)
img_src = args.image

if img_src is None:
    raise Exception("No image specified")

# load pretrained model
model = YOLO("yolo-weights/yolov8n-pose.pt")

def get_xy(keypoint):
    """ Convert Yolo tensor keypoint data to array and returns (x,y)  """
    try:
        return int(keypoint[0].item()), int(keypoint[1].item())
    except:
        raise Exception("unable to get keypoint coordinate")

# # run predict
# results = model(source=args.video, show=bool(int(to_show)))
frame = cv2.imread(img_src)
frame = cv2.resize(frame, (640,640))

results = model.predict(frame, conf=0.5)
for result in results:
    keypts = result.keypoints
    # print(f'Keypoints: \n{kpts}')
    num_people = keypts.shape[0]
    num_pts = keypts.shape[1]
    
    if num_pts !=0:
        for i in range(num_people):
            # extract relevant keypts
            left_shoulder = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['left_shoulder']])
            right_shoulder = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['right_shoulder']])
            left_hip = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['left_hip']])
            right_hip = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['right_hip']])
            left_knee = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['left_knee']])
            right_knee = get_xy(keypts.xy[i, my_defs.KEYPOINT_DICT['right_knee']])
            
            # check whether left/right keypt exist, if both exist get midpoint
            shoulder = utils.get_mainpoint(left_shoulder, right_shoulder, "shoulder")
            hips = utils.get_mainpoint(left_hip, right_hip, "hips")
            knees = utils.get_mainpoint(left_knee, right_knee, "knees")
            shoulder_exist = False
            hips_exist = False
            knees_exist = False
            
            # if relevant keypt exist draw pt
            if shoulder!=(0,0):
                shoulder_exist = True
                frame = utils.draw_keypoint(frame, shoulder)
                # print(f'Shoulder: {shoulder}')
            if hips!=(0,0):
                hips_exist = True
                frame = utils.draw_keypoint(frame, hips)
                # print(f'Hips: {hips}')
            if knees!=(0,0):
                knees_exist = True
                frame = utils.draw_keypoint(frame, knees)
                # print(f'Knees: {knees}')
                
            # if keypts exist draw line to connect them, calculate vector
            spine_vector = (0,0)
            legs_vector = (0,0)
            if shoulder_exist and hips_exist:
                spine_vector = utils.calculate_vector(hips, shoulder)
                # utils.draw_keypoint_line(frame, shoulder, hips)
                utils.draw_vector(frame, hips, spine_vector)
                # print(f'Spine Vector: {spine_vector}')
            if hips_exist and knees_exist:
                legs_vector = utils.calculate_vector(hips, knees)
                # utils.draw_keypoint_line(frame, hips, knees)
                utils.draw_vector(frame, hips, legs_vector)
                # print(f'Leg Vector: {legs_vector}')
            
            # calculate vector if all 3 main pts exist
            spine_leg_theta = -1 # angle between spine (vector between shoulder and hips) and legs (vector between hips and knees)
            spine_x_axis_phi = -1 # angle between spine (vector between shoulder and hips) and x_axis along hip point
            standing = None
            sitting = None
            lying_down = None
            if shoulder_exist and hips_exist and knees_exist:
                spine_leg_theta = utils.angle_between(spine_vector, legs_vector)
                hips_x_axis = utils.calculate_vector(hips, (hips[0]+20, hips[1]))
                # utils.draw_vector(frame, hips, hips_x_axis, color=(255,255,255))
                # spine_x_axis_phi = utils.calculate_angle_with_x_axis(spine_vector)
                spine_x_axis_phi = utils.angle_between(spine_vector, hips_x_axis)
                print(f'Theta: {spine_leg_theta}, Phi: {spine_x_axis_phi}')
                state = utils.action_state(spine_leg_theta, spine_x_axis_phi)
                print(f'State: {state}')

            

cv2.imshow('Yolo Pose Test', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()