import warnings
import os
import matplotlib.pyplot as plt


def save_fig(figure, path_name):
    """
    Helper function to save figure to the specified location - path_name

    :param figure: matplot
    :param path_name:
    :return:
    """

    # if path_name ends with "/" - do not save, e.g. "/home/user/dir1/dir2/"
    if os.path.basename(path_name):

        # get the directory path (exclude file_name in the end of path_name)
        dir_path = os.path.dirname(path_name)

        # if dir *path_name* does not exist - create
        # os.path.dirname(path_name) - make sure not to create a dir if path_name="myfile.png"
        if not os.path.exists(dir_path) and os.path.dirname(path_name):
            warnings.warn("The path - {} does not exist. But we will create this "
                          "directory for you and store the figure there.".format(dir_path))

            # get the dirname to create "/home/dir1/myfile.png" -> "/home/dir1"
            dir_create = os.path.dirname(path_name)
            os.makedirs(dir_create)
    else:
        warnings.warn("The location name - {} you gave us ends with \"/\". "
                      "Please give us the name of the file like /home/dir1/myfile.png "
                      "or to save in the current directory just the name *myfile.png*".format(path_name))
        return

    # save the figure to the file
    figure.savefig(path_name, bbox_inches=0, dpi=100)
