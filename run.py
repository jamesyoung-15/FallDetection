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
    debug = bool(args.debug)
    save_video = bool(args.save_vid)
    
    # image inference
    if is_image:
        yolo_action_detect.image_inference(img_src=media_source, width=vid_width, height=vid_height, show_frame=show_frame, debug=debug)
    # threaded webcam inference
    elif "/dev/" in media_source:
        yolo_action_detect.stream_inference(vid_source=media_source, vid_width=vid_width, vid_height=vid_height, 
                                            show_frame=show_frame, manual_move=manual_move, interval=interval, debug=debug)
    # normal video inference
    else:
        yolo_action_detect.video_inference(vid_source=media_source, vid_width=vid_width, vid_height=vid_height, 
                                           show_frame=show_frame, manual_move=manual_move, interval=interval, debug=debug)