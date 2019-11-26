# -------------------------- Write fixtures ---------------------------
# To regression test our wrappers we need examples. This script
# generates files. We save these files once, and regression_test.py
# re-generates these files to tests them for identicality with the
# presaved examples (fixtures). If they are found not to be identical
# it throws up an error.
#
# The point of this is to check that throughout the changes we make to
# scona the functionality of this script stays the same
#
# Currently the functionality of write_fixtures is to generate corrmat
# and network_analysis data via the functions
# corrmat_from_regionalmeasures and network_analysis_from_corrmat.
# ---------------------------------------------------------------------
import os
import scona as scn
import scona.datasets as datasets


def recreate_correlation_matrix_fixture(folder):
    # generate a correlation matrix in the given folder using
    # the Whitaker_Vertes dataset
    regionalmeasures, names, covars, centroids = (
        datasets.NSPN_WhitakerVertes_PNAS2016._data())
    corrmat_path = os.path.join(folder, 'corrmat_file.txt')
    scn.wrappers.corrmat_from_regionalmeasures(
        regionalmeasures,
        names_file=names,
        output_name=corrmat_path)


def recreate_network_analysis_fixture(folder, corrmat_path):
    # generate network analysis in the given folder using the #####
    # data in example_data and the correlation matrix given   #####
    # by corrmat_path                                         #####
    regionalmeasures, names, covars, centroids = (
        datasets.NSPN_WhitakerVertes_PNAS2016._data())
    # It is necessary to specify a random seed because
    # network_analysis_from_corrmat generates random graphs to
    # calculate global measures
    scn.wrappers.network_analysis_from_corrmat(
        corrmat_path,
        names,
        centroids,
        os.path.join(os.getcwd(), folder, 'network-analysis'),
        cost=10,
        n_rand=10,  # this is not a reasonable
        # value for n, we generate only 10 random
        # graphs to save time
        edge_swap_seed=2984
        )


def write_fixtures(folder='temporary_test_fixtures'):
    # Run functions corrmat_from_regionalmeasures and               ##
    # network_analysis_from_corrmat to save corrmat in given folder ##
    # --------------------------------------------------------------##
    # if the folder does not exist, create it
    if not os.path.isdir(os.path.join(os.getcwd(), folder)):
        os.makedirs(os.path.join(os.getcwd(), folder))
    # generate and save the correlation matrix
    print("generating new correlation matrix")
    recreate_correlation_matrix_fixture(folder)
    # generate and save the network analysis
    print("generating new network analysis")
    corrmat_path = os.path.join(folder, 'corrmat_file.txt')
    recreate_network_analysis_fixture(folder, corrmat_path)


def delete_fixtures(folder):
        import shutil
        print('\ndeleting temporary files')
        shutil.rmtree(os.getcwd()+folder)


def hash_folder(folder='temporary_test_fixtures'):
    hashes = {}
    for path, directories, files in os.walk(folder):
        for file in sorted(files):
            hashes[os.path.join(path, file)] = hash_file(
                os.path.join(path, file))
        for dir in sorted(directories):
            hashes.update(hash_folder(os.path.join(path, dir)))
        break
    return hashes


def hash_file(filename):
    import hashlib
    m = hashlib.sha256()
    with open(filename, 'rb') as f:
        while True:
            b = f.read(2**10)
            if not b:
                break
            m.update(b)
    return m.hexdigest()


def generate_fixture_hashes(folder='temporary_test_fixtures'):
    # generate the fixtures
    write_fixtures(folder=folder)
    # calculate the hash
    hash_dict = hash_folder(folder=folder)
    # delete the new files
    delete_fixtures("/"+folder)
    # return hash
    return hash_dict


def pickle_hash(hash_dict):
    import pickle
    with open("tests/.fixture_hash", 'wb') as f:
        pickle.dump(hash_dict, f)


def unpickle_hash():
    import pickle
    # import fixture relevant to the current python, networkx versions
    print('loading test fixtures')
    with open("tests/.fixture_hash", "rb") as f:
        pickle_file = pickle.load(f)
    return pickle_file


if __name__ == '__main__':
    if (input("Are you sure you want to update scona's test fixtures? (y/n)")
            == 'y'):
        hash_dict = generate_fixture_hashes()
        pickle_hash(hash_dict)
