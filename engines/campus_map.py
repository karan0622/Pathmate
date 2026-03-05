# camous ke layout ko graph ki tarah define kar rhe hai
# location is a node and edge is the distance 
CAMPUS_MAP = {
    # Entrance to the BLOCKS
    "ENTRANCE": [("BLOCK-A", 10), ("GROUND", 5)],
    "GROUND": [("ENTRANCE", 5)],
    "BLOCK-A": [("ENTRANCE", 10), ("BLOCK-B-DOOR-1", 10)],
    "BLOCK-B-DOOR-1": [("BLOCK-A", 10), ("BLOCK-B-DOOR-2", 5)],
    "BLOCK-B-DOOR-2": [("BLOCK-B-DOOR-1", 5), ("BLOCK-B-STAIRS", 6), ("BLOCK-E", 10)],
    "BLOCK-E": [("BLOCK-B-DOOR-2", 10)],
    "BOYS-HOSTEL": [],

    # Block B ke andr
    "BLOCK-B-STAIRS": [("BLOCK-B-DOOR-2", 6), ("BLOCK-B-FLOOR-2", 15)],
    "BLOCK-B-FLOOR-2": [("BLOCK-B-STAIRS", 15), ("BLOCK-B-CORRIDOR-2", 5)],
    "BLOCK-B-CORRIDOR-2": [("BLOCK-B-FLOOR-2", 5), ("B-201", 5), ("B-202", 8), ("BLOCK-B-LABS", 20)],
    "B-201": [("BLOCK-B-CORRIDOR-2", 5)],
    "B-202": [("BLOCK-B-CORRIDOR-2", 8)],
    "BLOCK-B-LABS": [("BLOCK-B-CORRIDOR-2", 20)],
}