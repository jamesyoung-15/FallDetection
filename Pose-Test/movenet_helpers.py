import cv2


def draw_debug(
    image,
    keypoint_score_th,
    keypoints_list,
    scores_list,
    bbox_score_th,
    bbox_list,
):
    debug_image = copy.deepcopy(image)

    # 0:鼻 1:左目 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手首
    # 10:右手首 11:左股関節 12:右股関節 13:左ひざ 14:右ひざ 15:左足首 16:右足首
    for keypoints, scores in zip(keypoints_list, scores_list):
        # Line：鼻 → 左目
        index01, index02 = 0, 1
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：鼻 → 右目
        index01, index02 = 0, 2
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左目 → 左耳
        index01, index02 = 1, 3
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：右目 → 右耳
        index01, index02 = 2, 4
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：鼻 → 左肩
        index01, index02 = 0, 5
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：鼻 → 右肩
        index01, index02 = 0, 6
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左肩 → 右肩
        index01, index02 = 5, 6
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左肩 → 左肘
        index01, index02 = 5, 7
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左肘 → 左手首
        index01, index02 = 7, 9
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：右肩 → 右肘
        index01, index02 = 6, 8
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：右肘 → 右手首
        index01, index02 = 8, 10
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左股関節 → 右股関節
        index01, index02 = 11, 12
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左肩 → 左股関節
        index01, index02 = 5, 11
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左股関節 → 左ひざ
        index01, index02 = 11, 13
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：左ひざ → 左足首
        index01, index02 = 13, 15
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：右肩 → 右股関節
        index01, index02 = 6, 12
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：右股関節 → 右ひざ
        index01, index02 = 12, 14
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
        # Line：右ひざ → 右足首
        index01, index02 = 14, 16
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv2.line(debug_image, point01, point02, (0, 0, 0), 2)

        # Circle：各点
        for keypoint, score in zip(keypoints, scores):
            if score > keypoint_score_th:
                cv2.circle(debug_image, keypoint, 6, (255, 255, 255), -1)
                cv2.circle(debug_image, keypoint, 3, (0, 0, 0), -1)

    # バウンディングボックス
    for bbox in bbox_list:
        if bbox[4] > bbox_score_th:
            cv2.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (255, 255, 255), 4)
            cv2.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 0, 0), 2)

    return debug_image