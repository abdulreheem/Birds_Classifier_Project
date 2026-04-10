FEATURES = [
    "gender",
    "body_mass",
    "beak_length",
    "beak_depth",
    "fin_length",
]

CLASS_LABEL_COLUMN = "bird category"
CLASS_VALUES = ["A", "B", "C"]
CLASS_PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]

GENDER_MAP = {
    "female": 0.0,
    "male": 1.0,
}
