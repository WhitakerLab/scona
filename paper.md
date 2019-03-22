---
title: 'scona: A Python package for network analysis of structural covariance networks'
tags:
  - Python
  - neuroimaging
  - network analysis
  - structural covariance networks
authors:
  - name: Kirstie Whitaker
    orcid:
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Isla Staden
    orcid:
    affiliation: 3
affiliations:
 - name:
   index: 1
 - name: Institution 2
   index: 2
 - name: Queen Mary University of London
   index: 3
date: some time 2019
bibliography: paper.bib
---

# Summary

[first paragraph - explanation of structural covariance networks]

`scona` is a toolkit to perform **s**tructural **co**variance brain **n**etwork **a**nalyses using python.
`scona` takes regional cortical thickness data obtained from structural MRI and generates a matrix of correlations between regions over a cohort of subjects. The correlation matrix is used alongside the [networkx package](https://networkx.github.io/) to generate a variety of networks and network measures.
There are lots of tools to analyse brain networks available.
Some require you to link together lots of different packages, particularly when you think about visualising your results.
Others will do the full analyses but require you to work with very specifically formatted datasets.
Our goal with `scona` is to balance these two by providing a modular, extendable and well documented package that can take in data pre-processed in multiple different ways and run your analysis all the way to the end.


### Who is scona built for?

Our target audience are researchers who have structural brain imaging data and would like to run quite standard structural covariance network analyses.
We don't want experts in neuroimaging to have to also become expert in building reproducible pipelines and well tested software.
`scona` is available to help researchers get started (and publish) their analyses quickly and reliably.



The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

``Gala`` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for ``Gala`` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. ``Gala`` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the ``Astropy`` package [@astropy] (``astropy.units`` and
``astropy.coordinates``).

``Gala`` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in ``Gala`` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike. The source code for ``Gala`` has been
archived to Zenodo with the linked DOI: [@zenodo]


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

# Acknowledgements
kirstie's paper
mozilla
gsoc

# References
