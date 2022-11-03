"""File for storing Graphical User Interface feature."""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from libs.k_means import KMeans


class GUI:
    """
    Class responsible for launching the functionality of the Graphical User Interface and some algorithms
    (like K-Means algorithm).
    """

    @staticmethod
    def print_title(title: str) -> None:
        """
        Static method responsible for displaying the title with Streamlit.

        :param str title: the title
        """

        st.title(title)

    @staticmethod
    def print_subheader(subheader: str) -> None:
        """
        Static method responsible for displaying the subheader with Streamlit.

        :param str subheader: the subheader
        """

        st.subheader(subheader)

    @staticmethod
    def run_and_draw_kmeans() -> None:
        """
        Static method responsible for starting the drawing of the graph and performing calculations for the
        K-Means algorithm.
        """

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        fig, axes = plt.subplots()

        random_points = st.slider('Number of random points', 10, 500, 300)
        random_array = np.random.rand(random_points, 2)
        clusters = st.slider('Number of clusters', 2, 6, 4)

        GUI.print_subheader(subheader=f"Clusters: {clusters}, points: {len(random_array)}.")

        kmeans = KMeans(k=clusters, data=random_array)
        kmeans.run_kmeans()

        GUI.print_subheader(f"The number of iterations necessary to achieve a sufficient result: {kmeans.iterations}.")

        x_coordinate = [centroid_coordinates['x'] for centroid_id, centroid_coordinates in kmeans.centroids.items()]
        y_coordinate = [centroid_coordinates['y'] for centroid_id, centroid_coordinates in kmeans.centroids.items()]
        axes.plot(x_coordinate, y_coordinate, 'ks')

        for _, point_data in kmeans.points.items():
            x_coordinate = point_data['x']
            y_coordinate = point_data['y']
            if point_data['closest_centroid']['id']:
                centroid_color_index = int(point_data['closest_centroid']['id']) - 1
            else:
                centroid_color_index = 6
            axes.plot(x_coordinate, y_coordinate, f'{colors[centroid_color_index]}o')

        st.pyplot(fig)
