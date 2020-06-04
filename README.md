# Sequential-Forward-Floating-Selection


The aim of this project was to implement a feature selector algorithm - Sequential Forward Floating Selector (SFFS) from scratch in python. SFS is also implemented

The code also supports to choose from either a wrapper method or filter method to calculate the significance of features. For the wrapper method 1-NN algorithm is used and for the filter method Mahalanobis distance method is used.

## Parameters supported
The python code supports the below parameters,

| Option           | Description                               | Default Value                  |
| ---------------- | ----------------------------------------- | ------------------------------ |
| -h, --help       | show this help message and exit           |                                |
| --dataset        | path_to_dataset                           | mushroom.csv                   |
| --objective_type | wrapper/filter                            | wrapper                        |
| --features       | K-best features to select                 | 5                              |
| --folds          | K-Folds cross validation. Used in wrapper | 5                              |
| --floating       | Select SFFS or SFS                        | False. SFS by default is used  |

## Run
To run the Sequential Forward Selection (SFS) algorithm with wrapper method (1-NN) using 5 fold cross validation to select 10 best features execute,\
`python feature_selection.py --dataset mushroom.csv --objective_type wrapper --features 10 --folds 5`

To run the Sequential Forward Selection (SFS) algorithm with filter method (mahalanobis distance) to select 10 best features execute,\
`python feature_selection.py --dataset mushroom.csv --objective_type filter --features 10`

To run the Sequential Forward Floating Selection (SFFS) algorithm with wrapper method (1-NN) using 5 fold cross validation to select 10 best features execute,\
`python feature_selection.py --dataset mushroom.csv --objective_type wrapper --features 10 --folds 5 --floating`

To run the Sequential Forward Floating Selection (SFFS) algorithm with filter method (mahalanobis distance) to select 10 best features execute,\
`python feature_selection.py --dataset mushroom.csv --objective_type filter --features 10 --floating`


The implemetation is based on [this research paper](http://library.utia.cas.cz/separaty/historie/somol-floating%20search%20methods%20in%20feature%20selection.pdf). For explanation of the algorithm and results please check the [Report folder](https://github.com/ajaygk95/Sequential-Forward-Floating-Selection/tree/master/Report)
