import utils
import old_code


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
        old_code.image_inference(img_src=media_source, width=vid_width, height=vid_height, show_frame=show_frame, debug=debug)
    # threaded webcam inference
    elif "/dev/" in media_source:
        old_code.stream_inference(vid_source=media_source, vid_width=vid_width, vid_height=vid_height, 
                                            show_frame=show_frame, manual_move=manual_move, interval=interval, debug=debug, conf_threshold=conf_threshold, save_video=save_video)
    # normal video inference
    else:
        old_code.video_inference(vid_source=media_source, vid_width=vid_width, vid_height=vid_height, 
                                           show_frame=show_frame, manual_move=manual_move, interval=interval, debug=debug, conf_threshold=conf_threshold, save_video=save_video)