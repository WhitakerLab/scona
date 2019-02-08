<img align="right" src="https://media.giphy.com/media/3ohhwwL4kj5z1Id6uI/giphy.gif">

# scona

[![Join the chat at https://gitter.im/WhitakerLab/scona](https://badges.gitter.im/WhitakerLab/scona.svg)](https://gitter.im/WhitakerLab/scona?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/WhitakerLab/scona.svg?branch=master)](https://travis-ci.org/WhitakerLab/scona)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/WhitakerLab/scona/blob/master/LICENSE)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/WhitakerLab/scona/master)

Welcome to the `scona` GitHub repository! :sparkles:

* [Get started](#get-started)
* [What are we doing?](#what-are-we-doing)
    * [Why do we need another package?](#why-do-we-need-another-package)
    * [Who is scona built for?](#who-is-scona-built-for)
* [Get involved](#get-involved)

## Get Started

If you don't want to bother reading the whole of this page, here are three places you can get your hands dirty and explore `scona`:

* To install `scona` as a python package with pip

```
pip install git+https://github.com/WhitakerLab/scona.git
```

* Check out our [tutorial](tutorials/tutorial.ipynb) for examples of basic functionality.
  Or run it interactively [in Binder](https://mybinder.org/v2/gh/whitakerlab/scona/master?filepath=tutorials%2Ftutorial.ipynb).

* Read the docs: https://whitakerlab.github.io/scona

## What are we doing?

`scona` is a toolkit to perform **s**tructural **co**variance brain **n**etwork **a**nalyses using python.

`scona` takes regional cortical thickness data obtained from structural MRI and generates a matrix of correlations between regions over a cohort of subjects.
The correlation matrix is used alongside the [networkx package](https://networkx.github.io/) to generate a variety of networks and network measures.

The `scona` codebase was first developed by Dr Kirstie Whitaker for the Neuroscience in Psychiatry Network publication "Adolescence is associated with genomically patterned consolidation of the hubs of the human brain connectome" published in PNAS in 2016 [(Whitaker*, Vertes* et al, 2016](http://dx.doi.org/10.1073/pnas.1601745113)).
In 2017, Isla Staden was awarded a [Mozilla Science Lab mini-grant](https://whitakerlab.github.io/resources/Mozilla-Science-Mini-Grant-June2017) to develop Kirstie's [original code](https://github.com/KirstieJane/NSPN_WhitakerVertes_PNAS2016) into a
documented, tested python package that is easy to use and re-use.

### Why do we need another package?

There are lots of tools to analyse brain networks available.
Some require you to link together lots of different packages, particularly when you think about visualising your results.
Others will do the full analyses but require you to work with very specifically formatted datasets.
Our goal with `scona` is to balance these two by providing a package that can take in data pre-processed in multiple different ways and run your analysis all the way to the end.

The code is modular, extendable and well documented.
You can run a standard analysis from beginning to end (including exporting "ready for publication" figures) with just a few clicks of a button.
We welcome any contributions to extend these "standard analysis" into a suite of possible investigations of structural covariance brain network data.

### Who is scona built for?

`scona` ties together the excellent graph theoretical analyses provided by [`NetworkX`](https://networkx.github.io) along with some neuroscience know-how.
For example, we've incorporated methods to include anatomical locations of your regions of interest, double edge swapping to preserve degree distribution in random graphs, and some standard reporting mechanisms that would be expected in a neuroimaging journal article.

Our target audience are researchers who have structural brain imaging data and would like to run quite standard structural covariance network analyses.
We don't want experts in neuroimaging to have to also become expert in building reproducible pipelines and well tested software.
(Although we'd love to answer questions if you are interested in improving your development skills!)
`scona` is available to help researchers get started (and publish) their analyses quickly and reliably.

We would also like to encourage network ninjas to incorporate their methods and ideas into the package either as code or by submitting an issue and a recommendation.

## Get involved

`scona` is openly developed and welcomes contributors.

If you're thinking about contributing (:green_heart: you are loved), our [roadmap](https://github.com/WhitakerLab/scona/issues/12) and our [contributing guidelines](CONTRIBUTING.md) are a good place to start.
You don't need advanced skills or knowledge to help out.
Newbies to Python, neuroscience, network analyses, git and GitHub are all welcome.

The only rule you absolutely have to follow when you're contributing to `scona` is to act in accordance with our [code of conduct](CODE_OF_CONDUCT.md).
We value the participation of every member of our community and want to ensure that every contributor has an enjoyable and fulfilling experience.

Our detailed development guide can be found at [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md).
Once you've read the [contributing guidelines](CONTRIBUTING.md), and if you think you need the additional information on linting, testing, and writing and building the documentation, please check out those instructions too.

If you have questions or want to get in touch you can join our [gitter lobby](https://gitter.im/WhitakerLab/scona), tweet [@Whitaker_Lab](https://twitter.com/whitaker_lab) or email Isla at [islastaden@gmail.com](mailto:islastaden@gmail.com).


## Other Stuff

To view our (successful) Mozilla Mini-Grant application, head [here](https://github.com/WhitakerLab/WhitakerLabProjectManagement/blob/master/FUNDING_APPLICATIONS/MozillaScienceLabMiniGrant_June2017.md).

In October 2017 scona ran a MozFest [session](https://github.com/MozillaFoundation/mozfest-program-2017/issues/724)
