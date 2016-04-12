"""
@author: Erfan Noury, https://github.com/erfannoury
"""
import numpy as np
from numpy.random import random as rnd


class PoissonDiskSampler(object):
    def __init__(self, width, height, radius, k=20):
        """
        This class is for sampling points in a 2-D region
        such that no two points are closer to each other
        than `radius`.
        This is code is based on Bostocks implementation (http://bl.ocks.org/mbostock/19168c663618b7f07158) which in turn is based on (https://www.jasondavies.com/poisson-disc/)

        Parameters
        ==========
        width: int
            width of the 2-D region
        height: int
            height of the 2-D region
        radius: float
            minimum distance between two arbitrary sampled points
        k: int
            maximum number of samples before rejection
        """
        self.width = width
        self.height = height
        self.radius = radius
        self.k = k
        self.cell_size = self.radius * 1.0 / np.sqrt(2.0)
        self.grid_width = int(np.ceil(self.width / self.cell_size))
        self.grid_height = int(np.ceil(self.height / self.cell_size))
        self.grid = [-1] * (self.grid_height * self.grid_width)
        self.queue = []
        self.samples = []

    def get_sample(self):
        """
        Returns an array of sample points sampled in the specified region using Bridson's Poisson-disk sampling algorithm

        Returns
        =======
        samples: list of tuples of two ints
            A list containing the coordinates sampled on a 2-d region such that no two samples points have distance less than `radius`.
        """
        # initialize with a seed point
        self.__sample__(rnd() * self.width, rnd() * self.height)
        while len(self.queue) > 0:
            idx = int(rnd() * len(self.queue))
            p = self.queue[idx]
            new_inserted = False
            for j in xrange(self.k):
                theta = 2 * np.pi * rnd()
                # radius <= r <= 2 * radius
                r = np.sqrt(3 * rnd() * self.radius**2 + self.radius**2)
                x = p[0] + r * np.cos(theta)
                y = p[1] + r * np.sin(theta)
                if (0 <= x < self.width) and (0 <= y < self.height) and self.__far__(x, y):
                    self.__sample__(x, y)
                    new_inserted = True
                    break
            # remove point from active list
            if not new_inserted:
                self.queue = self.queue[:idx] + self.queue[idx+1:]
                self.samples.append(p)

        return self.samples

    def __far__(self, x, y):
        i = int(y / self.cell_size)
        j = int(x / self.cell_size)
        i0 = np.max([i - 2, 0])
        j0 = np.max([j - 2, 0])
        i1 = np.min([i + 3, self.grid_height])
        j1 = np.min([j + 3, self.grid_width])

        for j in xrange(j0, j1):
            for i in xrange(i0, i1):
                if self.grid[i * self.grid_width + j] != -1:
                    dx = self.grid[i * self.grid_width + j][0] - x
                    dy = self.grid[i * self.grid_width + j][1] - y
                    if (dx**2 + dy**2) < (self.radius**2):
                        return False
        return True

    def __sample__(self, x, y):
        p = (x, y)
        self.queue.append(p)
        idx = int(self.grid_width * np.floor(y / self.cell_size) + np.floor(x / self.cell_size))
        self.grid[idx] = p
