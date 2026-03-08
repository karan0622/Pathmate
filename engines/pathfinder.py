from campus_map import CAMPUS_MAP

def find_path(start, end):
    queue = [(start, [start])] # means start from start and path we have covered so far [start] ish queue ke andr 2 cheeze hai hum kya explore karna hai and how we got there
    visited = set() # ish set ke andr voh hogi joh hum visit kara chuke hai
    while queue: # keep exploring until queue is empty
        current, path = queue.pop(0) # phle item lo aur phir define karo 

        if current == end:
            return path

        if current not in visited: # baar baar same position pe na aye ishliye
            visited.add(current) # current ko visited mein daalo and move forward
            for neighbour, distance in CAMPUS_MAP.get(current, []):  # check all neighbours 
                queue.append((neighbour, path + [neighbour])) # neighbours ko daalo aur updated paths ko and repeat

    return None  # if no path found

def generate_directions(path):
  directions = [] # empty list where we'll keep adding steps one by one
  for i in range(len(path) - 1): # 2 steps, always one less than total 3 locations ke liye
    current = path[i]
    next_place = path[i + 1]

    # ish function mein hum, kya kar rhe hai 
    """What this does:

    Looks up current location in the map
    Finds the neighbour that matches next_place
    Gets the distance
    Creates a sentence and adds it to directions list
"""
    
    for neighbour, distance in CAMPUS_MAP[current]:
      if neighbour == next_place:
        step = f"Walk from {current} to {next_place} - about {distance} steps"
        directions.append(step)
  
  directions.append(f"You have arrived at {path[-1]}")
  return directions

if __name__ == "__main__":
  path = find_path("ENTRANCE", "B-202")
  if path:
    directions = generate_directions(path)
    for step in directions:
      print(step)
  else:
    print("No path found")