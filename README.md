
# MovieLens Recommender

This project utilizes the MovieLens dataset to build a collaborative filtering system with the Alternating Least Squares model introduced by [Zhou et al.](https://link.springer.com/chapter/10.1007/978-3-540-68880-8_32) for the Netflix prize competition. This data set is rather large, with the full data set coming in at 20 million entries. Consequently, ALS was chosen for it's parallelizability as well as the fact that it is the only recommender system currently implemented in Spark's Mllib. 

Recommender systems differ from standard machine learning models in that they don't perform well under the model scoring paradigm that's generally used to train machine learning models. The reason for this is recommenders are often implemented to show the user something new rather than items that they have seen already. While data points can be withheld, other subjectivities of data sets such as popularity of few items make recommender systems hard to evaluate without a closed goal.

With that in mind, the goal of this project is to construct a deployable recommender system as a package built on top of the Spark framework that provides utilities for processing a data set, training models, evaluating models, and producing recommendations for end users rather than optimize a chosen scoring metric.

## To use this repository:
The dataset must be populated under `src/main/resources/ml-100k` to run the default implementations or the target directory must be properly set in the constructor of the `MovieLens` class and its children. 

The MovieLens datasets are freely available at:  
[https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)