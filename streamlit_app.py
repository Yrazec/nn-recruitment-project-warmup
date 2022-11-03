"""Runner file for K-Means wrapped in GUI."""

from gui.gui import GUI


def main():
    """Standard main function."""

    GUI.print_title(title='K-Means')
    GUI.run_and_draw_kmeans()


if __name__ == "__main__":
    main()
