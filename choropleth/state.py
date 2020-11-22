from copy import deepcopy


class MapState:
    colors = {"green", "red", "blue", "yellow", "cyan", "black"}

    def __init__(self, geomap, ncolored=0, depth=0, parent=None):
        self.map = geomap
        self.ncolored = ncolored
        self._depth = depth
        self.parent = parent

    def __hash__(self):
        hash_s = ""
        for i, key in enumerate(self.map, 1):
            if self.map[key][0] is not None:
                hash_s += str(i)

        if hash_s == "":
            return 0

        return int(hash_s)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_moves(self):
        result = []
        for region in self.map:
            if self.map[region][0] is None:
                avail = self.available_colors(region)
                for color in avail:
                    move = deepcopy(self)
                    move.map[region][0] = color
                    result.append(move)

            return result

    def available_colors(self, region):
        neighbors = self.map[region][1]
        neigcolors = set()
        for neighbor in neighbors:
            neigcolors.add(self.map[neighbor][0])

        return self.colors - neigcolors

    def valid_color(self, region, color):
        neighbors = self.map[region][1]
        for neighbor in neighbors:
            neigcolor = self.map[neighbor][0]
            if (neigcolor is not None) and (color == neigcolor):
                return False

        return True

    def is_goal(self):
        colors = [self.valid_color(region, self.map[region][0])
                  for region in self.map]
        return (self.ncolored == len(self.map)) and all(colors)
