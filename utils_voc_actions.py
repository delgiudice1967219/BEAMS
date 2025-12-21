import os
import numpy as np
import xml.etree.ElementTree as ET

VOC_ACTIONS = [
    "jumping",
    "phoning",
    "playinginstrument",
    "reading",
    "ridingbike",
    "ridinghorse",
    "running",
    "takingphoto",
    "usingcomputer",
    "walking",
]


def parse_voc_xml(xml_path):
    """
    Parses VOC XML to find people objects that have at least one active action.
    Returns a list of dicts: {'bbox': [xmin, ymin, xmax, ymax], 'actions': [...]}
    """
    if not os.path.exists(xml_path):
        return []

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    root = tree.getroot()
    people = []

    for obj in root.findall("object"):
        if obj.find("name").text != "person":
            continue

        # 1. Check Actions FIRST
        active_actions = set()
        act_node = obj.find("actions")
        if act_node is not None:
            for act in VOC_ACTIONS:
                # Check if the tag exists and is set to "1"
                val = act_node.find(act)
                if val is not None and val.text == "1":
                    active_actions.add(act)

        # 2. FILTER: If no actions are active, skip this person
        if len(active_actions) == 0:
            continue

        # 3. Get BBox only if they passed the filter
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        people.append({"bbox": [xmin, ymin, xmax, ymax], "actions": active_actions})

    return people


def compute_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two boxes.
    Boxes are expected in format [xmin, ymin, xmax, ymax].
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def mask_to_bbox(mask):
    """
    Converts a binary mask to a bounding box [xmin, ymin, xmax, ymax].
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return [xmin, ymin, xmax, ymax]
