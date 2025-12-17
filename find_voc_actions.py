# import os
# import xml.etree.ElementTree as ET
# import argparse

# # Le azioni ufficiali di PASCAL VOC
# VOC_ACTIONS = [
#     "jumping",
#     "phoning",
#     "playinginstrument",
#     "reading",
#     "ridingbike",
#     "ridinghorse",
#     "running",
#     "takingphoto",
#     "usingcomputer",
#     "walking",
# ]


# def find_contrast_images(voc_root):
#     ann_dir = os.path.join(voc_root, "Annotations")
#     print(f"Scansione di {ann_dir}...")

#     found_images = []

#     for filename in os.listdir(ann_dir):
#         if not filename.endswith(".xml"):
#             continue

#         path = os.path.join(ann_dir, filename)
#         try:
#             tree = ET.parse(path)
#         except:
#             continue

#         root = tree.getroot()

#         # Dizionario per salvare le azioni presenti in questa immagine
#         # Esempio: {'running': 1, 'walking': 2}
#         actions_in_image = {}
#         people_count = 0

#         for obj in root.findall("object"):
#             if obj.find("name").text != "person":
#                 continue

#             people_count += 1
#             act_node = obj.find("actions")

#             if act_node:
#                 for act in VOC_ACTIONS:
#                     val = act_node.find(act)
#                     # Se l'azione è 1 (true), la registriamo
#                     if val is not None and val.text == "1":
#                         actions_in_image[act] = actions_in_image.get(act, 0) + 1

#         # CRITERI DI SELEZIONE:
#         # 1. Almeno 2 persone
#         # 2. Almeno 2 azioni DIVERSE presenti (es. una corre, una cammina)
#         if people_count >= 2 and len(actions_in_image) >= 2:
#             print(f"--> TROVATA! {filename}")
#             print(f"    Azioni: {list(actions_in_image.keys())}")
#             found_images.append(filename)

#             if len(found_images) >= 5:  # Fermiamoci dopo 5 per il test
#                 break

#     return found_images


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # Punta alla root del tuo dataset VOC (dove ci sono JPEGImages e Annotations)
#     parser.add_argument("--voc_root", type=str, required=True)
#     args = parser.parse_args()

#     find_contrast_images(args.voc_root)


import os
import xml.etree.ElementTree as ET
import argparse
import csv
from collections import Counter

# --------------------------------------------------
# VOC ACTIONS
# --------------------------------------------------
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


# --------------------------------------------------
def parse_person_actions(obj):
    """Ritorna la lista di azioni attive (==1) per una persona"""
    actions = []
    act_node = obj.find("actions")
    if act_node is None:
        return actions

    for act in VOC_ACTIONS:
        node = act_node.find(act)
        if node is not None and node.text == "1":
            actions.append(act)
    return actions


def analyze_image(xml_path):
    """
    Analizza una singola annotation XML.
    Ritorna:
      - None se non valida
      - dict con info se multi-azione
    """
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return None

    root = tree.getroot()

    persons = []
    for obj in root.findall("object"):
        if obj.find("name").text != "person":
            continue

        acts = parse_person_actions(obj)
        persons.append(acts)

    if len(persons) < 2:
        return None

    # Raccogliamo tutte le azioni presenti
    flat_actions = [a for p in persons for a in p]
    unique_actions = set(flat_actions)

    if len(unique_actions) < 2:
        return None

    return {
        "num_persons": len(persons),
        "person_actions": persons,
        "unique_actions": sorted(unique_actions),
    }


def find_multi_action_images(voc_root, out_csv, max_images=50):
    ann_dir = os.path.join(voc_root, "Annotations")
    results = []

    print(f"Scansione: {ann_dir}")

    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".xml"):
            continue

        xml_path = os.path.join(ann_dir, fname)
        info = analyze_image(xml_path)

        if info is None:
            continue

        results.append(
            {
                "image": fname.replace(".xml", ".jpg"),
                "num_persons": info["num_persons"],
                "unique_actions": ",".join(info["unique_actions"]),
                "person_actions": " | ".join(
                    [",".join(p) if p else "none" for p in info["person_actions"]]
                ),
            }
        )

        print(f"[FOUND] {fname} -> {info['unique_actions']}")

        if len(results) >= max_images:
            break

    # Salva CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "num_persons",
                "unique_actions",
                "person_actions_per_instance",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["image"],
                    r["num_persons"],
                    r["unique_actions"],
                    r["person_actions"],
                ]
            )

    print(f"\nTrovate {len(results)} immagini multi-azione")
    print(f"CSV salvato in: {out_csv}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voc_root",
        type=str,
        required=True,
        help="Path to VOC2012 root (Annotations/, JPEGImages/)",
    )
    parser.add_argument(
        "--out_csv", type=str, default="voc_multi_action_candidates.csv"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=50,
        help="Numero massimo di immagini da trovare",
    )
    args = parser.parse_args()

    find_multi_action_images(
        voc_root=args.voc_root,
        out_csv=args.out_csv,
        max_images=args.max_images,
    )
