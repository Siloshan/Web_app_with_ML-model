# Lyrics Prediction Web App

This is a web application that utilizes a machine learning model to predict the categories of lyrics. Please note that this application has been created for educational purposes, and additional configurations may be required if the plan is to use it in mass production.

## System Architecture

The system is built using the following components:

1. Machine Learning Model: The ML model used in this application is implemented using scikit-learn (sklearn) library. It is trained to predict the categories of lyrics based on certain features.

2. Flask App: The Flask framework is used to develop the backend of the web application. It provides the necessary APIs to connect the ML model and the web app.

3. Web App: The web application is developed using Ajax, a technique that allows asynchronous communication between the web browser and the server. This enables real-time interaction with the Flask API to fetch predictions from the ML model.

## Configuration

To set up and configure the web app, follow the steps below:

1. Clone the repository from GitHub:

   ```
   git clone <repository_url>
   ```

2. Install the required dependencies by running the following command in your terminal:

   ```
   pip install -r requirements.txt
   ```

   This will install all the necessary libraries and packages mentioned in the requirements.txt file.

3. Train the machine learning model using your dataset. Make sure to preprocess the data and engineer the appropriate features for training.

4. Save the trained model to a file, such as `model.pkl`, using the pickle library:

   ```python
   import pickle
   
   # Assuming `model` is your trained machine learning model
   with open('model.pkl', 'wb') as f:
       pickle.dump(model, f)
   ```

5. Move the `model.pkl` file to the root directory of the web app.

6. Start the Flask app by running the following command:

   ```
   python app.py
   ```

   This will start the Flask development server.

7. Access the web app by opening your web browser and navigating to `http://localhost:5000`. You should see the web application interface.

8. Enter the necessary information or input data, and click the appropriate button to trigger the prediction process.

Please note that this is a basic configuration for running the web app in a development environment. For a production-ready deployment, additional considerations such as scalability, security, and performance optimizations should be taken into account.
