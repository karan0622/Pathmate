#bss ush map ko call kiya hai
from campus_map import CAMPUS_MAP 

# BFS function becoz It always finds the shortest path in terms of number of connections
def find_path(start, end):
  queue = [(start, [start])]
  