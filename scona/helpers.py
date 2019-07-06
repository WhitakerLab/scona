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

    # If path_name ends with "/" - do not save, e.g. "/home/user/dir1/dir2/"
    if os.path.basename(path_name):

        # get the directory path (exclude file_name in the end of path_name)
        # For example: "/home/dir1/myfile.png" -> "/home/dir1"
        dir_path = os.path.dirname(path_name)

        # Create the output directory (dir_path) if it does not already exist
        # or if the path_name is not a directory
        if not os.path.exists(dir_path) and os.path.dirname(path_name):
            warnings.warn('The path "{}" does not exist.\n'
                          "We will create this directory for you "
                          "and store the figure there.\n"
                          "This warning is just to make sure that you aren't "
                          "surprised by a new directory appearing!".format(dir_path))

            # Make the directory
            dir_create = os.path.dirname(path_name)
            os.makedirs(dir_create)
    else:
        warnings.warn('The file name you gave us "{}" ends with \"/\". '
                      "That is a directory rather than a file name."
                      "Please run the command again with the name of the file,"
                      "for example: '/home/dir1/myfile.png'"
                      "or to save the file in the current directory you can just pass 'myfile.png'".format(path_name))
        return

    # save the figure to the file
    figure.savefig(path_name, bbox_inches=0, dpi=100)
