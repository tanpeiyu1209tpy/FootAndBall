import numpy as np
import cv2

from data.augmentation import PLAYER_LABEL, BALL_LABEL

# pylint: disable=too-few-public-methods
class Color:
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)


def draw_bboxes(img, bboxes: np.array, color: tuple, width: int = 2):
    """
    Draws bounding boxes on given image

    Parameters
    ----------
    img :
        Image to draw bounding boxes on
    bboxes :
        A two dimensional array of corner points where each row is in format
        [x1, y1, x2, y2]

    Returns
    -------
    Image with drawn bounding boxes
    """

    img_bbox = img.copy()

    if len(bboxes.shape) != 2:
        raise ValueError(f"bboxes must be 2-dimensional, are {len(bboxes.shape)}")

    if bboxes.shape[1] != 4:
        raise ValueError(f"bboxes second dimension must be 4, is {bboxes.shape[1]}")

    for bbox in bboxes:
        cv2.rectangle(
            img_bbox,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            width,
        )

    return img_bbox


def draw_bboxes_on_detections(image, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, score in zip(
        detections["boxes"], detections["labels"], detections["scores"]
    ):
        if label == PLAYER_LABEL:
            x1, y1, x2, y2 = box
            color = (255, 0, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                image,
                f"{score:0.2f}",
                (int(x1), max(0, int(y1) - 10)),
                font,
                1,
                color,
                2,
            )

        elif label == BALL_LABEL:
            x1, y1, x2, y2 = box
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            color = (0, 0, 255)
            radius = 25
            cv2.circle(image, (int(x), int(y)), radius, color, 2)
            cv2.putText(
                image,
                f"{score:0.2f}",
                (max(0, int(x - radius)), max(0, (y - radius - 10))),
                font,
                1,
                color,
                2,
            )

    return image
