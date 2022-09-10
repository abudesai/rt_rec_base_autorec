AutoRecommender algorithm built in TensorFlow for Recommender - Base problem category as per Ready Tensor specifications.

- sklearn
- Tensorflow
- python
- pandas
- numpy
- scikit-optimize
- flask
- nginx
- uvicorn
- docker
- recommender system

This is a Recommender System that uses the autorec architecture implemented through Tensorflow. See paper here:
https://arxiv.org/abs/2007.07224

The recommender is equipped with early stopping: the model would stop training if there is no significant improvement in a perdetermined number of epochs, with default equals 3.

The data preprocessing step includes indexing and MinMax scaling. Also, the data is converted into sparse matrix for efficient memory consumption during training and prediction.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as jester, anime, book-crossing, modcloth, amazon electronics, and movies.

This Recommender System is written using Python as its programming language. Tensorflow and ScikitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.
