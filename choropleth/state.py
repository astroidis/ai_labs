class MapState:
    colors = {"green", "red", "blue", "yellow", "cyan"}  # , "pink", "black"}

    def __init__(self, geomap, ncolored=0, depth=0, parent=None):
        self.map = geomap
        self.ncolored = ncolored
        self._depth = depth
        self.parent = parent

    def __hash__(self):
        hash_s = ""
        for i, val in enumerate(self.map.values(), 1):
            if val[0] is not None:
                hash_s += str(i)

        if hash_s == "":
            return 0

        return int(hash_s)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_next_move(self):
        # get first noncolored region
        # and fill it with one of available color
        for region in self.map:
            if self.map[region][0] is None:
                avail = self.available_colors(region)
                for color in avail:
                    break
                return (region, color)

    def apply_move(self, move):
        self.map[move[0]][0] = move[1]

    def available_colors(self, region):
        _, neighbors = self.map[region]
        neigcolors = set()
        for neighbor in neighbors:
            neigcolors.add(self.map[neighbor][0])

        return self.colors - neigcolors

    def valid_color(self, region, color):
        _, neighbors = self.map[region]
        for neighbor in neighbors:
            neigcolor = self.map[neighbor][0]
            if (neigcolor is not None) and (color == neigcolor):
                return False
        return True

    def is_goal(self):
        colors = [self.valid_color(region, self.map[region][0])
                  for region in self.map]
        return (self.ncolored == len(self.map)) and all(colors)
