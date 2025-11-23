# config_classes.py
"""
Class ontology for advanced iterative object detection.
Maps target classes to synonyms for embedding fusion and defines background classes.
"""

TARGET_CLASSES = {
    "bread": ["bread", "loaf", "slice of bread", "bun", "toast"],
    "knife": ["knife", "cutlery", "blade", "kitchen knife", "utensil"],
    "towel": ["towel", "napkin", "cloth", "dish towel", "rag"],
    "cooking_pan": ["cooking pan", "skillet", "frying pan", "wok", "pot", "saucepan"],
    "window": ["window", "glass pane", "window frame", "glass window"],
    "car": ["car", "automobile", "vehicle", "sedan", "SUV", "truck"],
    "building": ["building", "skyscraper", "edifice", "structure", "office building"],
    "house": ["house", "home", "cottage", "residence", "detached house"],
    "person": ["person", "human", "man", "woman", "child", "pedestrian", "face"],
    "cat": ["cat", "kitten", "feline", "domestic cat", "pet"],
    "dog": ["dog", "puppy", "canine", "domestic dog", "pet"],
    "bear": ["bear", "grizzly", "polar bear", "wild animal"],
    "rabbit": ["rabbit", "bunny", "hare"],
    "goat": ["goat", "ram", "billy goat", "farm animal"],
    "apple": ["apple", "red apple", "green apple", "fruit"],
    "reading_glasses": ["reading glasses", "spectacles", "eyeglasses", "sunglasses"],
    "spoon": ["spoon", "cutlery", "tablespoon", "teaspoon", "soup spoon"],
    "wallet": ["wallet", "purse", "billfold", "card holder"],
    "plant": ["plant", "potted plant", "houseplant", "flowerpot", "vase"],
    "cup": ["cup", "mug", "teacup", "glass", "coffee cup", "beverage"],
    "jar": ["jar", "bottle", "glass jar", "container", "mason jar"]
}

# Backgrounds for suppression (Indoor + Outdoor contexts)
BACKGROUND_CLASSES = [
    "background", "noise", "blur", "text", "watermark",
    "floor", "wall", "ceiling", "carpet", "tile", "wood",  # Indoor
    "grass", "sky", "ground", "pavement", "road", "dirt", "tree", "clouds"  # Outdoor
]
