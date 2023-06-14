from flask import Flask
from flask_cors import CORS
from flask import Flask,jsonify, request
import asyncio
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, StringIndexer, IndexToString
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import webbrowser
import os
#global predict_csv=r"Others\readme.csv"
def delete_file9():
    my_file="readme.csv"

    # check if file exists 
    if os.path.exists(my_file):
        my_file="readme.csv"
        os.remove(my_file)

        # Print the statement once the file is deleted  
        print("The file: {} is deleted!".format(my_file))
    else:
        print("The file: {} does not exist!".format(my_file))

    with open(my_file, 'w') as f:
        f.write('id, lyrics\n')
        f.write('Create a new text file!')
    

# Load the saved model
spark2 = SparkSession.builder.master("local").appName("MyApp").getOrCreate()
#dtc_model = DecisionTreeClassificationModel.load("Others\DTmodel_new_for_Mendeley_dataset_CSV")
dtc_model = DecisionTreeClassificationModel.load("Others\DTmodel_new_for_Merged_dataset_CSV")
#spark.stop()
#dtc_model = DecisionTreeClassificationModel.load("DTmodel_new")
prepositions = ['aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'around', 'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'but', 'by', 'concerning', 'considering', 'despite', 'down', 'during', 'except', 'following', 'for', 'from', 'in', 'inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite', 'outside', 'over', 'past', 'per', 'plus', 'regarding', 'round', 'save', 'since', 'through', 'to', 'toward', 'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without']
stopwords = StopWordsRemover().getStopWords() + prepositions
tokenizer = RegexTokenizer(inputCol="lyrics", outputCol="words", pattern="\\W")
stopwords = StopWordsRemover(inputCol="words", outputCol="filtered")
prepositions = StopWordsRemover(inputCol="filtered", outputCol="clean", stopWords=prepositions)
vectorizer = HashingTF(inputCol="clean", outputCol="features", numFeatures=10000)
#sentenceDataFrame = spark2.createDataFrame([(0, input_lyrics)], ["id", "lyrics"])
pipeline2 = Pipeline(stages=[tokenizer, stopwords, prepositions, vectorizer])

url = "Others\dashboard.html"
webbrowser.open(url)

#def piplinepredict():
async def run_spark_prediction(lyrics_text):
    delete_file9()
    #lyrics_text='asfs'
    with open('readme.csv', 'w') as f:
        f.write('artist_name,track_name,release_date,genre,lyrics,ot\n')
        f.write('0,0,0,0,'+ ' '.join((lyrics_text.replace(",", "")).splitlines())+',0')
    
    #df_test = spark2.read.csv("test_ata.csv",header=True,inferSchema=True).select("artist_name","track_name","release_date","genre","lyrics")
    df_test = spark2.read.csv('readme.csv',header=True,inferSchema=True).select("artist_name","track_name","release_date","genre","lyrics")
    df_test.show()
    

    sentenceDataFrame_features = pipeline2.fit(df_test).transform(df_test)
    Prediction= dtc_model.transform(sentenceDataFrame_features)
    prediction_values=Prediction.collect()[0]['probability']
    result = await asyncio.sleep(5)
    return prediction_values






app= Flask(__name__)
CORS(app)
cors=CORS(app, resources={
    r"/*":{
    "origins":"*"
    }
})


@app.route('/fls', methods=['POST'])
def hello_world():    
    post_data=request.json['key1']
    #print(piplinepredict()[0])
    #return request.json['key1']
    return post_data
 

@app.route('/predict', methods=['POST'])
async def predict():
    #data = request.get_json()
    post_data=request.json['key1']
    prediction = await run_spark_prediction(post_data)
    print(str(prediction))
    return str(prediction)


if __name__ == '__main__':
    app.run()