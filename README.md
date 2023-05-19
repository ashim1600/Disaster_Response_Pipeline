# Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)
## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installing](#installation)
	3. [Executing Program](#execution)
	4. [Additional Material](#material)
3.[License](#license)
4. [Acknowledgement](#acknowledgement)


<a name="descripton"></a>
## Description
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.
The major sections of this project are as follows:

1. Preparing data for storage in a SQLite database by establishing an ETL pipeline to extract data from the source.
2. Create a machine learning pipeline to educate the system that can categorize text messages.
3. Utilize a web application that displays model results in real time.
<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly

<a name="installation"></a>
### Installing
To clone the git repository:https://github.com/ashim1600/Disaster_Response_Pipeline
<a name="execution"></a>
### Executing Program:
1. You can run the following commands in the project's directory to set up the database, train model and save the model.

    - To run ETL pipeline to clean data and store the processed data in the database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db`
    - To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file
        `python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl`
        <a name="execution"></a>
2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
<a name="material"></a>
### Additional Material

In the **data** and **models** folder you can find two jupyter notebook that will help you understand how the model works step by step:
1. **ETL Preparation Notebook**: learn everything about the implemented ETL pipeline
2. **ML Pipeline Preparation Notebook**: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn
<a name="importantfiles"></a>
### Important Files
**app/templates/***: templates/html files for web app

**data/process_data.py**: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

**models/train_classifier.py**: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use

**run.py**: This file can be used to launch the Flask web app used to classify disaster messages
<a name="license"></a>
## MIT
<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model
<a name="authors"></a>
<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model

<a name="Summary"></a>
## Summary:
In this project, I applied my data engineering skills to analyze disaster data from Appen (formerly Figure 8). The objective was to build a machine learning pipeline that classifies disaster messages and enables their efficient routing to the appropriate relief agencies through an API.

I started by exploring and understanding the dataset, which consisted of real messages sent during disaster events. The dataset had various variables, and my goal was to predict the category of each message to ensure it reaches the relevant disaster relief agency promptly.

To ensure accurate predictions, I performed essential data cleaning and preprocessing tasks. This involved handling missing values, removing duplicates, and addressing any inconsistencies or errors in the data. I also focused on preprocessing the text data by removing stop words, performing stemming or lemmatization, and converting the text into numerical representations suitable for machine learning algorithms.

Next, I applied feature engineering techniques to extract relevant features from the text data. These features were crucial in improving the performance of the machine learning model. I used techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) and word embeddings like Word2Vec or GloVe to capture the semantic meaning of the messages effectively.

With the preprocessed data and engineered features, I developed a machine learning model using various algorithms, including Naive Bayes, Logistic Regression, Support Vector Machines (SVM), Random Forests, or Gradient Boosting. The model was trained to classify the messages into different categories based on their content.

To evaluate the model's performance, I used appropriate metrics such as accuracy, precision, recall, and F1 score. I also employed techniques like cross-validation to ensure a robust estimate of the model's performance. The final model achieved a high level of accuracy in classifying the disaster messages correctly.

In order to make the model predictions accessible to the community, I deployed the model as an API. This API allows users to submit new messages, which are then automatically assigned to the appropriate disaster relief agency based on the model's predictions. This streamlines the process of routing messages, ensuring that each message reaches the relevant agency quickly and efficiently.

Impact on the Community:
The application developed in this project has significant implications for disaster management and relief efforts. By accurately classifying disaster messages, it enables quick and efficient allocation of resources and aid. Here are the key ways this application can benefit people and organizations in the event of a disaster:

Timely Response: During a disaster, every minute counts. By automatically routing messages to the appropriate relief agencies, the application ensures that help and support can be provided promptly to those in need. This reduces response times and increases the chances of saving lives and minimizing the impact of the disaster.

Optimized Resource Allocation: Different disaster events require specific resources and expertise. By accurately categorizing messages, the application helps in optimizing resource allocation. Each agency can receive messages that are relevant to their expertise, allowing them to allocate their resources efficiently and effectively.

Improved Coordination: In the chaos of a disaster, coordination among various relief agencies is crucial. The application streamlines the process of message routing, ensuring that each agency receives the messages that fall within their purview. This enhances collaboration and coordination among agencies, leading to a more organized and effective response.

Reduced Information Overload: During a disaster, there is often a flood of information and messages. Manually sorting through this vast amount of data can be overwhelming and time-consuming. The application automates the categorization process, reducing the burden on human operators and enabling them to focus on critical tasks that require their expertise.

Overall, this application plays a pivotal role in enhancing disaster response and relief efforts. By leveraging machine learning and automation, it optimizes resource allocation, reduces response times, improves coordination among agencies, and minimizes the burden of information
