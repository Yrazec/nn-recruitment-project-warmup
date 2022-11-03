"""File for storing K-Means feature."""

import math
import random
import numpy as np


class KMeans:
    """The class responsible for carrying out all the functionality related to the K-Means algorithm."""

    def __init__(self, k: int, data: np.ndarray):
        """
        Standard KMeans constructor.

        :param int k: number of centroids/clusters
        :param np.ndarray data: points to be clustered/grouped
        """

        self.k = k
        self.iterations = 0
        self.stop_flag = False
        self.number_of_points = len(data)

        self.centroids = {}
        self._create_random_centroids_object(data=data)

        self.points = {}
        self.points = self._create_points_object(data=data)

    def run_kmeans(self) -> None:
        """Method that runs all the functionality of the K-Means algorithm."""

        while not self.stop_flag:
            self.iterations += 1
            self._measure_distance()
            self._relocate_centroids()
            self._within_cluster_sum_of_squares()

    def _create_random_centroids_object(self, data: np.ndarray) -> dict:
        """
        Method that draws the centroids (so that they fit into the grid of points) and creates an object (dictionary)
        responsible for storing information about centroids.

        :param np.ndarray data:
            points to be clustered/grouped (in this case necessary to define the boundaries of the centroids)
        :return: object (dictionary) responsible for storing information about centroids
        """

        x_coordinate, y_coordinate = data[:, 0], data[:, 1]
        min_x, max_x, min_y, max_y = min(x_coordinate), max(x_coordinate), min(y_coordinate), max(y_coordinate)
        for i in range(1, self.k + 1):
            self.centroids[i] = {}
            self.centroids[i]['x'] = random.uniform(min_x, max_x)
            self.centroids[i]['y'] = random.uniform(min_y, max_y)
            self.centroids[i]['wcss'] = None
        return self.centroids

    def _create_points_object(self, data: np.ndarray) -> dict:
        """
        Method that creates an object (dictionary) responsible for storing information about points.

        :param np.ndarray data: points to be clustered/grouped
        :return: object (dictionary) responsible for storing information about points
        """

        x_coordinate, y_coordinate = data[:, 0], data[:, 1]
        for i, (x_coordinate, y_coordinate) in enumerate(zip(x_coordinate, y_coordinate), start=1):
            self.points[i] = {}
            self.points[i]['x'] = x_coordinate
            self.points[i]['y'] = y_coordinate
            self.points[i]['closest_centroid'] = {}
            self.points[i]['closest_centroid']['id'] = None
            self.points[i]['closest_centroid']['x'] = None
            self.points[i]['closest_centroid']['y'] = None
        return self.points

    def _measure_distance(self) -> None:
        """
        Method that measures the distance between each point relative to each centroids
        (distance = sqrt(pow(x2-x1, 2) + pow(y2-y1, 2))). This method includes a key check - if no point
        has changed the centroids, it means that we have reached the end of the algorithm.
        """

        unchanged_points_counter = 0
        for point_id, data in self.points.items():
            distances = {}
            for centroid_id, centroid_coordinates in self.centroids.items():
                distance = math.sqrt(
                    math.pow(centroid_coordinates['x'] - data['x'], 2) +
                    math.pow(centroid_coordinates['y'] - data['y'], 2)
                )
                distances[centroid_id] = distance

            closest = min(distances, key=distances.get)

            if self.points[point_id]['closest_centroid']['id']:
                if self.points[point_id]['closest_centroid']['id'] == closest:
                    unchanged_points_counter += 1

            self.points[point_id]['closest_centroid']['id'] = closest
            self.points[point_id]['closest_centroid']['x'] = self.centroids[closest]['x']
            self.points[point_id]['closest_centroid']['y'] = self.centroids[closest]['y']

        if unchanged_points_counter == self.number_of_points:
            self.stop_flag = True

    def _relocate_centroids(self) -> None:
        """
        Method that updates the location of the centroids.
        For example for centroid C1 the new location is C1 = ( (x1+...+xn)/n , (y1+...+yn)/n )
        """

        for centroid_id, _ in self.centroids.items():
            new_x = 0.0
            new_y = 0.0
            counter = 0
            for _, data in self.points.items():
                if data['closest_centroid']['id'] == centroid_id:
                    new_x += data['x']
                    new_y += data['y']
                    counter += 1
            try:
                new_x = new_x / counter
                new_y = new_y / counter
                self.centroids[centroid_id]['x'] = new_x
                self.centroids[centroid_id]['y'] = new_y
            except ZeroDivisionError:
                # Leave the old values if it is not possible to change the position of the centroid
                pass

    def _within_cluster_sum_of_squares(self) -> None:
        """
        Method that calculates Within-Cluster Sum of Squares.
        """

        for centroid_id, centroid_coordinates in self.centroids.items():
            wcss = 0.0
            for point_id, data in self.points.items():
                if data['closest_centroid']['id'] == centroid_id:
                    wcss += math.pow(math.sqrt(
                        math.pow(centroid_coordinates['x'] - data['x'], 2) +
                        math.pow(centroid_coordinates['y'] - data['y'], 2)
                    ), 2)
            self.centroids[centroid_id]['wcss'] = wcss
