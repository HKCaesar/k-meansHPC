# K-Means openMPI

Is an implementation of K-means algorithm in Python.

## Getting Started

There are two files, the paralel code and the serial code, most of the Prerequisites and instalation steps are just applicable to the paralel code.

Install openMPI (Ubuntu):
```
$ sudo apt-install openmpi-bin openmpi-doc libopenmpi-dev
```

### Prerequisites

- [Anaconda (Python3)](https://anaconda.org/anaconda/python) ->  5.0.1
- Python -> 2.7.13
- openMPI -> 1.6.3



### Installing

For python you need something like Anaconda, miniconda or virtualenv for installing the packages just for the project.

```
$ conda create -n mpi python=2.7
$ conda install -n mpi numpy
$ conda install -n mpi mpi4pi
```

Then

```
$ source activate mpi
(mpi) $ mpiexec --version
> mpiexec (OpenRTE) 1.6.3
```

At the end of the definitions of the methods you should adapt this 2 variables:

```
k = <Number of clusters>
datasetLocation = <location of the dataset>
```

## Running the tests (Command)

```
(mpi) $ mpiexec -n <number of cores> python ./Paralel.py #Paralel

$ python ./Serial #Serial

```
### Running the tests  (Output both Serial or Paralel)

```
> Clusters: <Array of clusters>
Documents with no Relation: <Set of documents of no relation>
Time: <Time in seconds>
```
## Authors

* **Diego Alejandro Perez**
* **Edwin Montoya Jaramillo**

## Acknowledgments

* Edwin Nelson Montoya Múnera
* Juan David Pineda Cardenas
* Juan Francisco Cardona Mc'Cormick
* Daniel Hoyos Ospina
* Daniela Serna Escobar
* Daniel Rendon
