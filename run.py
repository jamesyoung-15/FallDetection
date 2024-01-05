import utils
import yolo_action_detect


if __name__ == '__main__':
    # get user passed args
    args = utils.get_args()
    media_source = args.src
    show_frame = args.show
    manual_move = bool(int(args.manual_frame))
    conf_threshold = args.conf_score
    vid_width = args.width
    vid_height = args.height
    is_image = bool(int(args.type))
    interval = args.interval
    
    if is_image:
        yolo_action_detect.image_inference(img_src=media_source, width=vid_width, height=vid_height, show_frame=show_frame)
    else:
        yolo_action_detect.video_inference(vid_source=media_source, vid_width=vid_width, vid_height=vid_height, 
                                           show_frame=show_frame, manual_move=manual_move, interval=interval)