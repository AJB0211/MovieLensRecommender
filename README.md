
# MovieLens Recommender

This project utilizes the MovieLens dataset to build a collaborative filtering system with the Alternating Least Squares model. This data set is rather large, with the full data set coming in at 20 million entries. Consequently, ALS was chosen for it's parallelizability as well as the fact that it is the only recommender system currently implemented in Spark's Mllib. 

Recommender systems differ from standard machine learning models in that they don't perform well under the model scoring paradigm that's generally used to train machine learning models. The reason for this is recommenders are often implemented to show the user something new rather than items that they have seen already. While data points can be withheld, other subjectivities of data sets such as popularity of few items make recommender systems hard to evaluate without a closed goal.

With that in mind, the goal of this project is to construct a deployable recommender system as a package built on top of the Spark framework that provides utilities for processing a data set, training models, evaluating models, and producing recommendations for end users rather than optimize a chosen scoring metric.