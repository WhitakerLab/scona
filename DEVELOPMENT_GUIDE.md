# Development Guide

The contributing guidelines above have dealt with getting involved, asking questions, making pull requests, etcetera.
The [development guide](#development-guide) deals with the specifics of contributing code to the `scona` codebase, and ends with a worked example to guide you through the process of writing docstrings and tests for new sections of code.

* [Installing](#installing-in-editable-mode)
* [Linting](#linting)
* [Docstrings](#writing-docstrings)
* [Building Sphinx docs](#building-sphinx-docs)
* [Tutorials](#tutorials)
* [Testing](#testing)
* [A worked example](#worked-example)


### Installing in editable mode

Use `pip install -e git+https://github.com/WhitakerLab/scona.git` to install scona in editable mode. This means that the python install of scona will be kept up to date with any changes you make, including switching branches in git.

### Linting

scona uses the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/).
You can use [flake8](http://flake8.pycqa.org/en/latest/) to lint code.
We're quite a young project (at time of writing in January 2019) and so we aren't going to be super hardcore about your linting!
Linting should make your life easier, but if you're not sure how to get started, or if this is a barrier to you contributing to `scona` then don't worry about it or [get in touch](CONTRIBUTING.md#how-to-get-in-touch) and we'll be happy to help you.

### Writing docstrings

We at scona love love LOVE documentation 😍 💖 😘 so any contributions that make using the various functions, classes and wrappers easier are ALWAYS welcome.

`scona` uses the `sphinx` extension [`napoleon`](http://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) to generate code from numpy style docstrings. See the [numpydoc guide](https://numpydoc.readthedocs.io/en/latest/) for details on syntax.
For an example of how docstrings are written in scona, checkout the [docstrings section](#step-1-docstrings) in our [code example](#worked-example) below.

`sphinx` can automatically create links to crossreference other packages. If set up correctly ``:class:`package-name.special-class` `` renders as `package-name.special-class` with a link to the `special-class` documentation in `package-name`'s online documentation. If the package is scona, the package name can be omitted, so that
``:class:`networkx.Graph` `` becomes [`networkx.Graph`](https://networkx.github.io/documentation/stable/reference/classes/graph.html#networkx.Graph), and ``:func:`create_corrmat` `` becomes [`create_corrmat`](https://whitakerlab.github.io/scona/build/html/scona.html#scona.make_corr_matrices.create_corrmat).

Crossreferencing is currently set up for the python standard library, networkx, pandas, numpy and python-louvain. It is possible to set this up for other python packages by adding
```python
    'package-name': ('https://package.documentation.url/', None)
```
to the `intersphinx_mapping` dictionary in [docs/source/conf.py](docs/source/conf.py) (the handy dandy configuration file for sphinx).

### Building Sphinx docs

When [docstrings](https://en.wikipedia.org/wiki/Docstring#Python) are updated, `sphinx` can automatically update the docs (and ultimately our website). Unfortunately this is [not yet an automated process](https://github.com/WhitakerLab/scona/issues/79). For the time being somebody needs to build those pages. If you're comfortable doing this you can follow the instructions below, but if it's going to be a barrier to you submitting a pull request then please just prepare the docstrings and the maintainers of scona will build the html files for you 😃.
You might also use these instructions to build documenatation locally while you're still writing, for example to check rendering.

You will need `sphinx` (`pip install sphinx`) and `make` (depends on your distribution) installed.
In a terminal, navigate to the docs folder and run `make html`. You should be able to view the new documentation in your browser at `file:///local/path/to/scona/docs/build/html/scona.html#module-scona`

### Tutorials

You may also want to show off the functionality of some new (or old) code. Please feel free to add a tutorial to the tutorials folder. You may find it helpful to use the `NSPN_WhitakerVertes_PNAS2016` data as an example dataset, as demonstrated in [tutorials/tutorial.ipynb](#tutorials/tutorial.ipynb).

### Testing

Tests don't need to be exhaustive or prove the validity of a function. The aim is only to alert us when something has gone wrong. Testing is something most people do naturally whenever they write code. If you think about the sorts of things you would try running in a command line or jupyter notebook to test out a function you have just defined, these are the sorts of things that can go in unit tests.

scona uses pytest to run our test suite. pytest runs test discovery on all modules with names ending `_test.py`, so if you make a new test module, make sure the filename conforms to this format.
Use [`py.test`](https://docs.pytest.org/en/latest) to run tests, or `py.test --cov=scona` if you also want a test coverage report.

For an example of how tests are written in scona, checkout the [testing section](#step-2-testing) in our [code example](#worked-example) below.

### Random seeds

Sometimes you want a random process to choose the same pseudo-random numbers each time so that the process returns the same result each time. This is particularly useful for testing and reproducibility. To do this we set a [random seed](https://www.tutorialspoint.com/python/number_seed.htm).

There is currently no way to seed the random graph generators in scona except by setting the global seed. For more discussion on this subject see [issue #77](https://github.com/WhitakerLab/scona/issues/77). To set the global random seed put the following lines near the top of your test.

```python
import random
random.seed(42)
```
Where 42 is your integer of choice, see https://docs.python.org/3.7/library/random.html#random.seed


### Worked Example

A lot of the developer guidelines above are a little hard to apply in the abstract, so this section is going to apply them to a sample piece of code. We'll start with a working function and show you how to [add a docstring](#writing-docstrings) and [add some tests](#testing).

We'll start with a new function to calculate the proportion of interhemispheric edges *leaving* each module of a graph. This is a somewhat silly idea given that we have no guarantee that a module is entirely within one hemisphere, but it is only intended for the purpose of demonstration.

```python
def calc_leaving_module_interhem(G, M):
  # Assign a binary variable "interhem" to each edge in G
  # Assign a "hemisphere" label to each node in G
  # N.B this function relies on G having nodal attribute "centroids" defined
  assign_interhem(G)

  leaving_interhem = {}
  # Loop over the modules in M
  for module, nodeset in M.items():
    # Calculate the proportion of edges from a node inside module to a node outside of module that are interhemispheric
    # N.B interhem is a 0, 1 variable indicating if an edges is interhemispheric, so it is possible to sum over the interhem value of valid edges.
    leaving_interhem[module] = np.mean([G.[node1][node2]['interhem'] for node1 in nodeset for node2 in nx.all_neighbors(node1) if node2 not in nodeset])
  return leaving_interhem
```

Now suppose we decide to add this back into the scona source code.

#### step 1: docstrings

The key things to have in the docstring are a short description at the top,
an explanation of the function parameters, and a description of what the
function returns. If, say, the function returns nothing, or has no parameters, you can leave those out. For `calc_leaving_module_interhem` we might write something like this:

```python
def calc_leaving_module_interhem(G, M):
'''
Calculate the proportion of edges leaving each module that are
interhemispheric

Parameters
----------
G : :class:`networkx.Graph`
M : dict
  M maps module names to vertex sets. M represents a nodal
  partition of G

Returns
-------
dict
  a dictionary mapping a module name to the proportion of interhemispheric
  leaving edges

See Also
--------
:func:`assign_interhem`
'''
```

Let's say we add this to [`scona.graph_measures`](scona/graph_measures.py). If you followed the [instructions to build sphinx documentation locally](#building-sphinx-docs) you would be able to view this function in your browser at `file:///local/path/to/scona/docs/build/html/scona.html#module-scona.graph_measures.calc_leaving_module_interhem`

#### step 2: Testing

Now we need to write some tests for this function to [tests/graph_measures_test.py](scona/graph_measures.py)
Tests don't need to be exhaustive or prove the validity of a function. The aim is simply to alert us when something has gone wrong. Testing is something most people do naturally when they write code. If you think about the sorts of sanity checks you would try running in a command line or jupyter notebook to make sure everything is working properly when you have just defined a function, these are the sorts of things that should go in unit tests.

Examples of good tests for `calc_leaving_module_interhem` might be:
* checking that `calc_leaving_module_interhem` raises an error when run on a graph where the nodal attribute "centroids" is not defined.
* checking that `calc_leaving_module_interhem(G, M)` has the same dictionary keys as M for some partition M.
* given a partition M with only two modules, check that the values of `calc_leaving_module_interhem(G, M)` are equal as they are evaluating the same set of edges. (There is no third module to connect to, so the set of leaving A is actually the set of edges from A -> B, which is the same set of edges from B -> A, and these are precisely the edges leaving B)
* given a partition M with only one module, check that the values of `calc_leaving_module_interhem(G, M)` are 0, as there are no leaving edges.

*
    ```
    G :       0-----1
              |     |
              |     |
              2-----3
    ```
    We can define a simple square graph G with nodes 0 and 2 in the right hemisphere and 2 and 3 in the left hemisphere. Using this simple graph let's calculate the expected result for a couple of partitions.
    * If M = `{0: {0,2}, 1: {1,3}}` then `calc_leaving_module_interhem(G, M)` is `{0:1.0, 1:1.0}` since all leaving edges are interhemispheric
    * If M = `{0: {0,1}, 1: {2,3}}` then `calc_leaving_module_interhem(G, M)` is `{0:0.0, 1:0.0}` since no leaving edges are interhemispheric
    * If M = `{0: {0}, 1: {1,2,3}}` then `calc_leaving_module_interhem(G, M)` is `{0:0.5, 1:0.5}`
    These are all sanity checks we can use to check our function is working as expected.


Let's set up a `LeavingModuleInterhem` test case. We use the `setUpClass` method to define some variables which we will use repeatedly throughout.

```python
class LeavingModuleInterhem(unittest.TestCase):
  @classmethod
  def setUpClass(self):
      self.G_no_centroids = nx.erdos_renyi_graph(20, 0.2)
      self.G_random_graph = self.G_no_centroids.copy()
      # assign random centroids to G_random_graph
      scn.assign_node_centroids(
        self.G_random_graph,
        [tuple(np.subtract((.5, .5, .5), np.random.rand(3))) for x in range(20)])
      # Define a trivial partition for G1
      self.M_one_big_module = {0: set(G_random_graph.nodes)}
      # Create a second partition of G1 with two modules
      nodeset1 = set(np.random.choice(self.G_random_graph.nodes))
      nodeset2 = set(self.G_random_graph.nodes).difference(nodeset1)
      self.M_two_random_modules = {0: nodeset1, 1: nodeset2}
      # Create graph G2
      # G2 :      0-----1
      #           |     |
      #           |     |
      #           2-----3
      self.G_square_graph = nx.Graph()
      self.G_square_graph.add_nodes_from([0, 1, 2, 3])
      self.G_square_graph.add_edges_from([(0, 2), (2, 3), (1, 3), (0, 1)])
      scn.assign_node_centroids(
        self.G_square_graph, [(1, 0, 0), (-1, 0, 0), (1, 0, 0), (-1, 0, 0)])
```
Now we have defined the set up for testing, we can move on to the testing methods. These should be short methods making one or two assertions about the behaviour of our new function. We try to make the function names descriptive so that if they error during testing we can tell at a glance which behaviour they were testing.

```python
  def centroids_must_be_defined(self):
    # This function should fail on a graph where no centroids are defined
    with self.assertRaises(Exception):
      scn.calc_leaving_module_interhem(
          self.G_no_centroids, self.M_one_big_module)

  def result_keys_are_modules(self):
    # check that `calc_leaving_module_interhem(G, M)` has the same
    # dictionary keys as M
    result = scn.calc_leaving_module_interhem(
        self.G_random_graph, self.M_one_big_module)
    assert result.keys() == self.M_one_big_module.keys()

  def trivial_partition_values_equal_zero(self):
    # check that the values of `calc_leaving_module_interhem(G, M)` are 0,
    # as there are no leaving edges
    result = scn.calc_leaving_module_interhem(
        self.G_random_graph, self.M_one_big_module)
    for x in result.values():
      assert x == 0

  def partition_size_two_modules_have_equal_values(self):
    # check that the values of `calc_leaving_module_interhem(G, M)` are equal,
    # as they are evaluating the same edges.
    L2 = scn.calc_leaving_module_interhem(
        self.G_random_graph, self.M_two_random_modules)
    assert L2[0] == L2[1]

  def G2_modules_are_hemispheres_values_are_1(self):
    # the leaving interhem values should be one for each module, since since
    # all leaving edges are interhemispheric
    result = scn.calc_leaving_module_interhem(
        self.G_square_graph, {0: {0, 2}, 1: {1, 3}})
    assert result == {0: 1.0, 1: 1.0}

  def G2_modules_are_split_across_hemispheres_values_0(self):
    # the leaving interhem values should be zero for each module, since since
    # none of the leaving edges are interhemispheric
    result = scn.calc_leaving_module_interhem(
        self.G_square_graph, {0: {0, 1}, 1: {2, 3}})
    assert result == {0: 0.0, 1: 0.0}

  def G2_test_module_M5(self):
    result = scn.calc_leaving_module_interhem(
        self.G_square_graph, {0: {0}, 1: {1, 2, 3}})
    assert  == {0: .5, 1: .5}
```

And now you're ready to roll! :tada:

Thank you for reading this far through scona's contributing guidelines :sparkles::hibiscus::tada:.
As always, if you have any question, see any typo's, or have suggestions or corrections for these guidelines don't hesitate to [let us know](#how-to-get-in-touch):heart_eyes:.
