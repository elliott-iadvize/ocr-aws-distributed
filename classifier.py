#!/usr/bin/python
# -*- coding: utf-8 -*-

#Regular packages import
import os
import re
import sys
import json
import boto3
import itertools
import subprocess

#Spark packages import
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import LabeledPoint

class Classifier(object):
    """
    CONFIGURATION
    """

    def __init__(self):
        """
        Classifier allows to load, train and test features of images in order to class them
        Type of classification: 1 vs All
        Data: features extracted using the extract-features.py script on Oxford IIIT-Pet Dataset
        """

        #Set python environment variable when running in dev
        #os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"

        #Initialize the Spark context and retrieve Spark session
        self.sc = SparkContext()
        self.spark = SparkSession.builder.getOrCreate()

        #Fetch script argument and init variables
        self.classes = []
        self.best_models = {}
        self.classes_dup = []
        self.features_value_list = []
        self.bucket_name = sys.argv[1]

        #Import boto3 and s3 tools + connect to bucket
        self.s3 = boto3.resource("s3")
        self.oc_bucket = self.s3.Bucket(self.bucket_name)


    def load_classes_and_values_from_features(self):
        """
        Reads the bucket specified in argument and extracts both classes and values from features
        Input: path to features directory
        Output: list of features' classes and features' values
        """

        #Read bucket in order to extract classes and values
        for object in self.oc_bucket.objects.filter(Prefix="images/a"):
            feature_file = object.key.replace("images/", "")

            #Extract classes and append them in list
            new_class = re.sub(r'[0-9]', "", feature_file)
            new_class = new_class[:-9].strip("_")
            self.classes.append(new_class)

            file = self.s3.Object(self.bucket_name, object.key)
            values = file.get()["Body"].read().decode("utf-8").strip("[]").split(",")
            values = [float(x) for x in values]

            self.features_value_list.append([new_class, values])

        #Sort classes and create a duplicate
        self.classes = sorted(list(set(self.classes)))
        self.classes_dup = self.classes

    def convert_labels_to_float(self):
        """
        Convert strings label into float with an estimator
        Input: self.train_features_df, self.test_features_df
        Output: self.train_features_df, self.test_features_df with converted labels
        """

        self.label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
        self.label_indexer_transformer = self.label_indexer.fit(self.train_features_df)
        self.train_features_df = self.label_indexer_transformer.transform(self.train_features_df)
        self.test_features_df = self.label_indexer_transformer.transform(self.test_features_df)

    def load_features_in_dataframe_1_vs_All(self, class1):
        """
        Uses the features value extracted in list to create a RDD then a dataframe with values
        Input: self.features_value_list, class1
        Output: self.features_df containing the features value
        """

        #Init variable
        self.features_row_list = []

        print("#5: Creating Rows from features_value_list")
        for feature in self.features_value_list:
            if feature[0] == class1:
                features_row = Row(label=feature[0], features=feature[1])
            else:
                features_row = Row(label="All", features=feature[1])

            self.features_row_list.append(features_row)

        print("#6: Creating RDD from features_row_list")
        features_rdd = self.sc.parallelize(self.features_row_list)

        print("#7: Creating dataframe from RDD")
        self.features_df = self.spark.createDataFrame(features_rdd)

    def training(self, class1, class2):
        """
        Train classifier - Grid search on SVMWithSGD model parameters
        Train model for each combination and return best model parameters
        Input: class1 and class2 used in classification
        Output: Result of the classification in a file
        """

        print("#12: Begining training for %s vs. %s classifier" % (class1, class2))

        #Defining model parameters
        self.model_number = 0
        self.number_of_iterations = [10,50,100,300]  # number of iterations. (default: 100)
        self.step_sizes = [1] # step parameter used in SGD (default: 1.0)
        self.regularizer_parameters = [0.01] # regularizer parameter. (default: 0.01)

        #Init variable
        self.best_model_number = None
        self.best_model = None
        self.best_prediction_df = None
        self.best_accuracy = 0
        self.best_train_errors = 0
        self.best_errors = 0
        self.best_number_of_iteration = None
        self.best_step_size = None
        self.best_regularizer_parameter = None

        #Grid training
        for number_of_iteration, step_size, regularizer_parameter in itertools.product(self.number_of_iterations, self.step_sizes, self.regularizer_parameters):
            self.model_number += 1

            print("#13: Building model #%i" % (self.model_number))
            model = SVMWithSGD.train(self.train_features_labelpoints, number_of_iteration, step_size, regularizer_parameter)

            #Objective is to guess the labels on test data
            print("#14: Testing model #%i" % (self.model_number))
            self.prediction_labelpoints = self.test_features_labelpoints.map(lambda tflp : Row(label_index_predicted=model.predict(tflp.features), label_index=tflp.label))
            self.prediction_df = self.spark.createDataFrame(self.prediction_labelpoints)

            #Then the objective is to evaluate the model on the training data
            print("#15: Evaluating model #%i" % (self.model_number))
            self.accuracy = self.prediction_df.filter(self.prediction_df.label_index_predicted == self.prediction_df.label_index).count() / float(self.prediction_df.count())
            self.train_errors = self.prediction_df.filter(self.prediction_df.label_index_predicted != self.prediction_df.label_index).count() / float(self.prediction_df.count())
            self.errors = self.prediction_df.filter(self.prediction_df.label_index_predicted != self.prediction_df.label_index).count()

            #Finally compare those results with best_model in memory
            if self.accuracy > self.best_accuracy:
                self.best_model_number = self.model_number
                self.best_model = model
                self.best_prediction_df = self.prediction_df
                self.best_accuracy = self.accuracy
                self.best_train_errors = self.train_errors
                self.best_errors = self.errors
                self.best_step_size = step_size
                self.best_regularizer_parameter = regularizer_parameter
                self.best_number_of_iteration = number_of_iteration

            print("""
            Model #%i
    		Model trained with (number_of_iteration: %.2f, step_size = %.2f, regularizer_parameter = %.2f)
    		Model has an accuracy of %.3f (errors: %i / train_errors: %.3f) on test data
            """ % (self.model_number, number_of_iteration, step_size, regularizer_parameter, self.accuracy, self.errors, self.train_errors))

        #At last, print results for the best model
        print("""
        Best model is the model number #%i
        Best model was trained with (number_of_iteration: %.2f, step_size = %.2f, regularizer_parameter = %.2f)
        Best model has accuracy of %.3f (errors: %i / train_errors: %.3f) on test data
        """ % (self.best_model_number, self.best_number_of_iteration, self.best_step_size, self.best_regularizer_parameter, self.best_accuracy, self.best_errors, self.best_train_errors))

        #And add this best_model into best_models dictionnary
        self.best_models[("%s_vs_%s" % (class1, class2))] = {
            "accuracy":self.best_accuracy,
            "number_of_iteration":self.best_number_of_iteration,
            "step_size":self.best_step_size,
            "regularizer_parameter":self.best_regularizer_parameter,
            "errors":self.best_errors,
            "train_errors":self.best_train_errors}

    def main(self):
        """
        Execute the process of loading, training and testing features to class them
        Input: Nothing
        Output: Result file named result_classifier.json
        """

        print("#1: Reads the bucket specified in argument and extracts both classes and values from features")
        self.load_classes_and_values_from_features()

        print("#2: Processing 1 vs All classification")
        for class1 in self.classes:
            print("#3: Dataframes construction for classifier %s vs All" % (class1))

            print("#4: Loading features values into main dataframe")
            self.load_features_in_dataframe_1_vs_All(class1)

            print("#8: Spliting dataframe into train and test samples")
            self.train_features_df, self.test_features_df = self.features_df.randomSplit([0.5, 0.5])

            print("#8.1: %i training data" % (self.train_features_df.count()))
            print("#8.2: %i testing data" % (self.test_features_df.count()))

            print("#9: Convert strings label into float with an estimator")
            self.convert_labels_to_float()

            print("#10: Convert dataframe into labelpoints RDD")
            self.train_features_labelpoints = self.train_features_df.rdd.map(lambda row: LabeledPoint(row.label_index, row.features))
            self.test_features_labelpoints = self.test_features_df.rdd.map(lambda row: LabeledPoint(row.label_index, row.features))

            print("#11: Training classifier")
            self.training(class1, "All")

        print("#Final results: loading best_models dictionnary informations to best_classifiers.json file")
        with open("./best_classifiers.json", "w") as out:
            json.dump(self.best_models, out)

        #Explicitly ask to terminate the script in order to access to Spark Web UI for improvements/optimisation purposes (available here http://localhost:4040)
        #input("#User action asked: press ctrl+c to exit, remember Spark Web UI is available here: http://localhost:4040")

#Trigger
if __name__ == "__main__":
    dummy = Classifier()
    dummy.main()
