# mountain car has 2 dimensions:
#
# position :  -1.2 < x < 0.5
# velocity : -0.07 < v < 0.07
#
#
# Use velocity along x axis and position on y axis to create a grid, and get a vector which gives all the grids
# a point is in.
#
import numpy as np

class TileRange:

    def __init__(self, low, high, sections=10, offset_proportion=0):
        """ Non inclusive range - expects   low < val < high

        Divides into sections and allows an offset which is a proportion (e.g. 0.5 off sets by half a section)
        0 <= offset_proportion <= 1
        """
        if low < high:
            self.low = low
            self.high = high
        else:
            self.high = low
            self.low = high
        self.sections = sections
        self.graduation = self.span() / (self.sections - 1)
        if offset_proportion < 0 or offset_proportion > 1:
            raise ValueError(f"offset_proportion {offset_proportion} is outside the range: 0 <= offset_proportion <= 1")
        self.offset = offset_proportion * self.graduation

    def span(self):
        return self.high - self.low

    def section_for(self, val):
        # determine which section val falls into
        if val <= self.low:
            # print(f"Value {val} is outside the range {self.low} < val < {self.high}")
            val = self.low
        if val >= self.high:
            # print(f"Value {val} is outside the range {self.low} < val < {self.high}")
            val = self.high

        return int((val + self.offset - self.low) / self.graduation)


class TileGrid:

    def __init__(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range
        self.grids = self.x_range.sections * self.y_range.sections

    def extract_feature(self, x, y):
        feature = np.zeros(self.grids)
        # work out location of x,y in the feature
        x_section = self.x_range.section_for(x)
        y_section = self.y_range.section_for(y)

        grid = x_section + (y_section * self.x_range.sections)
        feature[grid] = 1

        return feature


def main():
    position_range = TileRange(-1.2, +0.5)
    velocity_range = TileRange(-0.07, +0.07)

    tile = TileGrid(position_range, velocity_range)

    x = tile.extract_feature(-1.2, -0.07)


if __name__ == '__main__':
    main()