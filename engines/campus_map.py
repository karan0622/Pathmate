# camous ke layout ko graph ki tarah define kar rhe hai
# location is a node and edge is the distance 
CAMPUS_MAP = {
    # Outdoor
    "ENTRANCE":     [("RECEPTION", 10), ("GROUND", 5)],
    "GROUND":       [("ENTRANCE", 5)],
    "RECEPTION":    [("ENTRANCE", 10), ("BLOCK B", 10)],
    "BLOCK B":      [("RECEPTION", 10), ("STAIRS", 5), ("BLOCK E", 10)],
    "BLOCK E":      [("BLOCK B", 10)],
    "HOSTEL":       [],

    # Block B Indoor
    "STAIRS":       [("BLOCK B", 5), ("FLOOR TWO", 15)],
    "FLOOR TWO":    [("STAIRS", 15), ("CORRIDOR", 5)],
    "CORRIDOR":     [("FLOOR TWO", 5), ("B 201", 5), ("B 202", 8), ("LAB", 20)],
    "B 201":        [("CORRIDOR", 5)],
    "B 202":        [("CORRIDOR", 8)],
    "LAB":          [("CORRIDOR", 20)],
}