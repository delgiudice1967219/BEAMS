import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# --- CONFIGURATION ---
# Target Image
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005572.jpg"

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


def get_human_location(bbox, img_w, img_h):
    """
    Returns a string describing the location (e.g., 'Bottom-Left', 'Center').
    bbox: [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # Horizontal Position
    if cx < img_w / 3:
        h_loc = "Left"
    elif cx < 2 * img_w / 3:
        h_loc = "Center"
    else:
        h_loc = "Right"

    # Vertical Position
    if cy < img_h / 3:
        v_loc = "Top"
    elif cy < 2 * img_h / 3:
        v_loc = "Center"
    else:
        v_loc = "Bottom"

    if h_loc == "Center" and v_loc == "Center":
        return "Center"
    return f"{v_loc}-{h_loc}"


def parse_voc_xml(xml_path):
    if not os.path.exists(xml_path):
        print(f"❌ Error: XML file not found at {xml_path}")
        return []

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        print(f"❌ Error: Could not parse XML.")
        return []

    root = tree.getroot()
    people = []

    for i, obj in enumerate(root.findall("object")):
        if obj.find("name").text != "person":
            continue

        # 1. Get BBox
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        # 2. Get Actions
        active_actions = []
        act_node = obj.find("actions")
        if act_node is not None:
            for act in VOC_ACTIONS:
                # Some XMLs use 'playinginstrument', others 'playing_instrument'.
                # We check loosely.
                val = act_node.find(act)
                if val is not None and val.text == "1":
                    active_actions.append(act)

        # 3. Check for specific flags useful for debugging
        difficult = obj.find("difficult")
        is_difficult = difficult is not None and difficult.text == "1"

        truncated = obj.find("truncated")
        is_truncated = truncated is not None and truncated.text == "1"

        people.append(
            {
                "id": i,
                "bbox": [xmin, ymin, xmax, ymax],
                "actions": active_actions,
                "difficult": is_difficult,
                "truncated": is_truncated,
            }
        )

    return people


def main():
    print(f"--- DEBUGGING IMAGE: {os.path.basename(IMG_PATH)} ---")

    # 1. Check Image
    if not os.path.exists(IMG_PATH):
        print("❌ Image file not found.")
        return

    # FIX: Use numpy to read unicode paths on Windows
    try:
        # Read file as byte stream to handle 'à' correctly
        with open(IMG_PATH, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Image decoding failed")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"❌ Critical Error reading image: {e}")
        return

    h, w, _ = img.shape
    print(f"Image Size: {w}x{h}")

    # 2. Parse XML
    xml_path = IMG_PATH.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
    people = parse_voc_xml(xml_path)

    print(f"Found {len(people)} Ground Truth 'person' objects.\n")

    # 3. Print Text Report
    print(
        f"{'ID':<4} | {'Location':<15} | {'Size (WxH)':<12} | {'Difficult':<10} | {'Actions'}"
    )
    print("-" * 85)

    for p in people:
        bbox = p["bbox"]
        loc = get_human_location(bbox, w, h)
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        diff_str = "YES" if p["difficult"] else "no"
        act_str = ", ".join(p["actions"]) if p["actions"] else "None (Background?)"

        print(f"{p['id']:<4} | {loc:<15} | {bw}x{bh:<9} | {diff_str:<10} | {act_str}")

    # 4. Visualize
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    ax = plt.gca()

    # Colors for distinct boxes
    colors = ["red", "cyan", "lime", "magenta", "yellow", "orange"]

    for i, p in enumerate(people):
        bbox = p["bbox"]
        color = colors[i % len(colors)]

        # Draw Box
        rect = plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=3,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Draw Label (ID + Actions)
        label_text = f"ID {p['id']}: {', '.join(p['actions'])}"
        if p["difficult"]:
            label_text += " (DIFF)"

        ax.text(
            bbox[0],
            bbox[1] - 5,
            label_text,
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="none"),
        )

    plt.title(f"Ground Truth Debug: {os.path.basename(IMG_PATH)}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
