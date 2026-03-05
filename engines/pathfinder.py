#bss ush map ko call kiya hai
from campus_map import CAMPUS_MAP 

# BFS function becoz It always finds the shortest path in terms of number of connections
def find_path(start, end):
  queue = [(start, [start])] # first thing to explore is start and path so far is start is line a mtlb ye hain
  visited = set() # jissh place pe hum phle the like imagining a map and crossing off locations on a map
  while queue:
    current, path = queue.pop(0) # take the first item from the queue split it into where i am "current" and how i got here "path"

    if current == end:
      return path
      