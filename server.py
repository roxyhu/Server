import flask
import os
from flask import *
import requests
import difflib
import random
import mysql.connector
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import numpy
import pandas, numpy, textblob, string
# from keras.preprocessing import text, sequence
# from keras import layers, models, optimizers
from flask_cors import CORS
# def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
#     # fit the training dataset on the classifier
#     classifier.fit(feature_vector_train, label)
    
#     # predict the labels on validation dataset
#     predictions = classifier.predict(feature_vector_valid)
#     # print('feature_vector_valid', feature_vector_valid)


#     if is_neural_net:
        
#         predictions = predictions.argmax(axis=-1)
#     # print('模型預測', predictions)
#     # print('實際分類', valid_y)
#     return metrics.accuracy_score(predictions, valid_y)
# def train_model_test(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
#     tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
#     tfidf_vect.fit(trainDF['text'])
#     testdata_tfidf = tfidf_vect.transform(feature_vector_valid)
#     # fit the training dataset on the classifier
#     classifier.fit(feature_vector_train, label)
    
#     # predict the labels on validation dataset
#     predictions = classifier.predict(testdata_tfidf)
#     predictions2 = classifier.predict_proba(testdata_tfidf)
#     if is_neural_net:    
#         predictions = predictions.argmax(axis=-1)
#     prediction_dicts = {'library': predictions2[0][0], 'pool': predictions2[0][1], 'supermarket': predictions2[0][2], 'school': predictions2[0][3], 'park': predictions2[0][4]}
#     return prediction_dicts

# # load the dataset
# data = open('all_1000', 'rt', encoding='utf-8').read()
# labels, texts = [], []
# a=0
# for i, line in enumerate(data.split("\n")):
#     content = line.split()
#     a=a+1
#     # print(content)
#     labels.append(content[0])
#     texts.append(" ".join(content[1:]))
# # create a dataframe using texts and lables
# trainDF = pandas.DataFrame()
# trainDF['text'] = texts
# trainDF['label'] = labels

# # split the dataset into training and validation datasets 
# train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
# # label encode the target variable 
# encoder = preprocessing.LabelEncoder()
# train_y = encoder.fit_transform(train_y)
# valid_y = encoder.fit_transform(valid_y)






# #词语级tf-idf
# tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
# tfidf_vect.fit(trainDF['text'])
# xtrain_tfidf = tfidf_vect.transform(train_x)
# xvalid_tfidf = tfidf_vect.transform(valid_x)

# #特征为词语级别TF-IDF向量的朴素贝叶斯
# accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
# # print ("NB, WordLevel TF-IDF: ", accuracy)

# # print ("0 : library , 1 : pool , 2 : supermarket , 3 : school , 4 : park")

mydb = mysql.connector.connect(
  host="140.125.32.128",       # 数据库主机地址
  user="root",    # 数据库用户名
  passwd="",   # 数据库密码
  database="yunbot"
)
mycursor = mydb.cursor()
app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True
@app.route('/test',methods=['POST','GET'])
def testserver():
    if request.method =='POST':
        if request.values['send']=='test':
            sentences_list = []
            sentences_list.insert( 0, request.values['sentence'])
            out = train_model_test(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, sentences_list)
            local_1 = max(out, key=out.get)
            out[local_1] = 0
            local_2 = max(out, key=out.get)
            out[local_2] = 0 
            local_3 = max(out, key=out.get) 
            print("第一 : " + local_1)
            print("第二 : " + local_2)
            print("第三 : " + local_3)
            print(local_1)
            return render_template('index.html',model_result=local_1)
    return render_template('index.html',model_result="")
@app.route('/')
def index():
        sql = "SELECT location, status FROM yunbot_data WHERE uid ='1'"
        mycursor.execute(sql)
        output = mycursor.fetchall()
        #yunbot_location = 
        #print(output[0])
        if(output[0][1]==0):
            return "0"
        else:
            sql = "UPDATE yunbot_data SET status ='0' WHERE uid ='1'"
            mycursor.execute(sql)
            mydb.commit()
            return output[0][0]
@app.route('/2')
def index2():
        sql = "SELECT location, status FROM yunbot_data WHERE uid ='2'"
        mycursor.execute(sql)
        output = mycursor.fetchall()
        #yunbot_location = 
        #print(output[0])
        if(output[0][1]==0):
            return "0"
        else:
            sql = "UPDATE yunbot_data SET status ='0' WHERE uid ='2'"
            mycursor.execute(sql)
            mydb.commit()
            return output[0][0]
@app.route('/chongRobot')
def chongRobot():
        sql = "SELECT action, status FROM chongbot WHERE uid ='1'"
        mycursor.execute(sql)
        output = mycursor.fetchall()
        #yunbot_location = 
        #print(output[0])
        if(output[0][1]==0):
            return "0"
        else:
            sql = "UPDATE chongbot SET status ='0' WHERE uid ='1'"
            mycursor.execute(sql)
            mydb.commit()
            return output[0][0]
        
@app.route('/ScoreBot', methods=['POST'])
def ScoreBot():
    dict1 = {'Go straight': 0, 'Turn left': 7, 'Turn right': 0, 'Go forward': 0 ,'Go to the pool': 0 ,'Go ahead': 0 ,'Go to the library': 0 ,'Go to the school': 0 ,'Go to the park': 0 ,'Go to the supermarket': 0 }
    text = request.values.get('message')
    print("you:" + text )
    go_straight = difflib.SequenceMatcher(None, text, 'Go straight').quick_ratio()
    turn_left = difflib.SequenceMatcher(None, text, 'Turn left').quick_ratio()
    turn_right = difflib.SequenceMatcher(None, text, 'Turn right').quick_ratio()
    go_forward = difflib.SequenceMatcher(None, text, 'Go forward').quick_ratio()
    go_to_the_pool = difflib.SequenceMatcher(None, text, 'Go to the pool').quick_ratio()
    go_ahead = difflib.SequenceMatcher(None, text, 'Go ahead').quick_ratio()
    go_to_the_library = difflib.SequenceMatcher(None, text, 'Go to the library').quick_ratio()
    go_to_the_school = difflib.SequenceMatcher(None, text, 'Go to the school').quick_ratio()
    go_to_the_park = difflib.SequenceMatcher(None, text, 'Go to the park').quick_ratio()
    go_to_the_supermarket = difflib.SequenceMatcher(None, text, 'Go to the supermarket').quick_ratio()
    dict1['Go straight'] = round(go_straight,2)*100
    dict1['Turn left'] = round(turn_left,2)*100
    dict1['Turn right'] = round(turn_right,2)*100
    dict1['Go forward'] = round(go_forward,2)*100
    dict1['Go to the pool'] = round(go_to_the_pool,2)*100
    dict1['Go ahead'] = round(go_ahead,2)*100
    dict1['Go to the library'] = round(go_to_the_library,2)*100
    dict1['Go to the school'] = round(go_to_the_school,2)*100
    dict1['Go to the park'] = round(go_to_the_park,2)*100
    dict1['Go to the supermarket'] = round(go_to_the_supermarket,2)*100
    
    score = str(dict1[max(dict1, key=dict1.get)])
    similar_words = str(max(dict1, key=dict1.get))
    print(score)
    print(similar_words)
    t = {
        'score' : score,
        'similar_words': similar_words
    }
    return jsonify(t)
@app.route('/armbot', methods=['POST'])
def armbot():
    print(request.values)
    text = request.values.get('message')
    bot = request.values.get('bot')
    print("you:" + text )
    sql = "UPDATE chongbot SET action = %s ,status = %s WHERE uid = %s"
    val = ( text, "1", "1")
    mycursor.execute( sql, val)
    mydb.commit()
    t = {
        'similar_words': "收到"
    }
    return jsonify(t)

@app.route('/NLPBot', methods=['POST'])
def NLPBot():
    print(request.values)
    text = request.values.get('message')
    bot = request.values.get('bot')
    print("you:" + text )
    sentences_list = []
    sentences_list.insert( 0, text)
    out = train_model_test(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, sentences_list)
    local_1 = max(out, key=out.get)
    out[local_1] = 0
    local_2 = max(out, key=out.get)
    out[local_2] = 0 
    local_3 = max(out, key=out.get) 
    print("第一 : " + local_1)
    print("第二 : " + local_2)
    print("第三 : " + local_3)
    t = {
        'similar_words_1': local_1,
        'similar_words_2': local_2,
        'similar_words_3': local_3
    }
    return jsonify(t)
@app.route('/new_train_data', methods=['POST'])
def new_train_data():
    print(request.values)
    status = request.values.get('status')
    data = request.values.get('data')
    label = request.values.get('label')
    # print("you:" + new_train_data )
    t = {
        'return': "收到"
    }
    return jsonify(t)
@app.route('/SimBot', methods=['POST'])
def SimBot():
    dict1 = {'Go straight': 0, 'Turn left': 7, 'Turn right': 0, 'Go forward': 0 ,'Go to the pool': 0 ,'Go ahead': 0 ,'Go to the library': 0 ,'Go to the school': 0 ,'Go to the park': 0 ,'Go to the supermarket': 0 }
    text = request.values.get('message')
    print("you:" + text )
    go_straight = difflib.SequenceMatcher(None, text, 'Go straight').quick_ratio()
    turn_left = difflib.SequenceMatcher(None, text, 'Turn left').quick_ratio()
    turn_right = difflib.SequenceMatcher(None, text, 'Turn right').quick_ratio()
    go_forward = difflib.SequenceMatcher(None, text, 'Go forward').quick_ratio()
    go_to_the_pool = difflib.SequenceMatcher(None, text, 'Go to the pool').quick_ratio()
    go_ahead = difflib.SequenceMatcher(None, text, 'Go ahead').quick_ratio()
    go_to_the_library = difflib.SequenceMatcher(None, text, 'Go to the library').quick_ratio()
    go_to_the_school = difflib.SequenceMatcher(None, text, 'Go to the school').quick_ratio()
    go_to_the_park = difflib.SequenceMatcher(None, text, 'Go to the park').quick_ratio()
    go_to_the_supermarket = difflib.SequenceMatcher(None, text, 'Go to the supermarket').quick_ratio()
    dict1['Go straight'] = round(go_straight,2)*100
    dict1['Turn left'] = round(turn_left,2)*100
    dict1['Turn right'] = round(turn_right,2)*100
    dict1['Go forward'] = round(go_forward,2)*100
    dict1['Go to the pool'] = round(go_to_the_pool,2)*100
    dict1['Go ahead'] = round(go_ahead,2)*100
    dict1['Go to the library'] = round(go_to_the_library,2)*100
    dict1['Go to the school'] = round(go_to_the_school,2)*100
    dict1['Go to the park'] = round(go_to_the_park,2)*100
    dict1['Go to the supermarket'] = round(go_to_the_supermarket,2)*100
    
    score = str(dict1[max(dict1, key=dict1.get)])
    similar_words = str(max(dict1, key=dict1.get))
    scoreInt = float(score)
    print("scoreInt",scoreInt)
    print(similar_words)
    print(request.values)
    bot = request.values.get('bot')
    print("you:" + text )     

    if scoreInt >= 80:

        sql = "UPDATE data SET location = %s ,status = %s WHERE uid = %s"
        val = ( similar_words, "1", bot)
        mycursor.execute( sql, val)
        mydb.commit()
        print("00000000", similar_words)
    else:
        sql = "UPDATE data SET location = %s ,status = %s WHERE uid = %s"
        val = ( text, "1", bot)
        mycursor.execute( sql, val)
        mydb.commit()
        print(bot)

    word_array = text.split()
    print("word_array" ,word_array)
    word_array_num = list(set(word_array).symmetric_difference(set(similar_words.split())))
    print("word_array_num",word_array_num)
    t = {
        'score' : score,
        'similar_words': similar_words,
        'word_array': "word_array",
        'word_array_num': "word_array_num"
    }
    print("t", t )
    return jsonify(t)
@app.route('/YunBot', methods=['POST'])
def YunBot():
    dict1 = {'Go straight': 0, 'Turn left': 7, 'Turn right': 0, 'Go forward': 0 ,'Go to the pool': 0 ,'Go ahead': 0 ,'Go to the library': 0 ,'Go to the school': 0 ,'Go to the park': 0 ,'Go to the supermarket': 0 }
    text = request.values.get('message')
    print("you:" + text )
    go_straight = difflib.SequenceMatcher(None, text, 'Go straight').quick_ratio()
    turn_left = difflib.SequenceMatcher(None, text, 'Turn left').quick_ratio()
    turn_right = difflib.SequenceMatcher(None, text, 'Turn right').quick_ratio()
    go_forward = difflib.SequenceMatcher(None, text, 'Go forward').quick_ratio()
    go_to_the_pool = difflib.SequenceMatcher(None, text, 'Go to the pool').quick_ratio()
    go_ahead = difflib.SequenceMatcher(None, text, 'Go ahead').quick_ratio()
    go_to_the_library = difflib.SequenceMatcher(None, text, 'Go to the library').quick_ratio()
    go_to_the_school = difflib.SequenceMatcher(None, text, 'Go to the school').quick_ratio()
    go_to_the_park = difflib.SequenceMatcher(None, text, 'Go to the park').quick_ratio()
    go_to_the_supermarket = difflib.SequenceMatcher(None, text, 'Go to the supermarket').quick_ratio()
    dict1['Go straight'] = round(go_straight,2)*100
    dict1['Turn left'] = round(turn_left,2)*100
    dict1['Turn right'] = round(turn_right,2)*100
    dict1['Go forward'] = round(go_forward,2)*100
    dict1['Go to the pool'] = round(go_to_the_pool,2)*100
    dict1['Go ahead'] = round(go_ahead,2)*100
    dict1['Go to the library'] = round(go_to_the_library,2)*100
    dict1['Go to the school'] = round(go_to_the_school,2)*100
    dict1['Go to the park'] = round(go_to_the_park,2)*100
    dict1['Go to the supermarket'] = round(go_to_the_supermarket,2)*100
    
    score = str(dict1[max(dict1, key=dict1.get)])
    similar_words = str(max(dict1, key=dict1.get))
    print(score)
    print(similar_words)
    print(request.values)
    bot = request.values.get('bot')
    print("you:" + text )
    sql = "UPDATE yunbot_data SET location = %s ,status = %s WHERE uid = %s"
    val = ( text, "1", bot)
    mycursor.execute( sql, val)
    mydb.commit()
    t = {
        'score' : score,
        'similar_words': similar_words
    }
    return jsonify(t)

@app.route('/upload',methods=['GET','POST'])
def upload():
    if flask.request.method=='GET':
        return flask.render_template('upload.html')
    else:
        file=flask.request.files['file']
        print("0000000000000000000",file.filename)
        if file:
            file.save(file.filename)
            return jsonify(t = {'similar_words': "收到"})

@app.route('/yatest', methods=['POST'])
def yatest():
    print( " request.values",request.get_json())
    t = {
    'ans': "data",
    'type': "result"
    }
    return jsonify(t)

@app.route('/rasaBot', methods=['POST'])
def rasaBot():
    global r_animal
    bot = request.values.get('bot')
    try:
        url = 'http://140.125.32.141:5005/webhooks/rest/webhook'
        data = request.values.get('message')
        print(data)
        datajs = {"sender": bot,"message": data}
        datajs = json.dumps(datajs)
        x = requests.post(url, data = datajs)
        # print(x.json())
        emptylist = x.json()
        # print("empty",emptylist)
        result = x.json()[0]['text']
        print("you:" + data + "  ans:" + result)
        ans = "0"


    except:
        print("777777777777777777")
        data = request.values.get('message')
        print(data)
        datajs = {"sender": bot,"message": "ABC"}
        datajs = json.dumps(datajs)
        x = requests.post(url, data = datajs)
        # print(x.json())
        # print("x",x)
        emptylist = x.json()
        # print("empty",emptylist)
        result = x.json()[0]['text']
        if result == " ":
            result = "Sorry, I didn't get that. Could you speak a little more slowly,please?"
        print("you:" + data + "  ans:" + result)
        ans = "0"
    
    result_split = result.split("###",1)
    print("0000",result_split[0])
    
    final_result = result_split[0]
    print(type(final_result))

    update_users = "UPDATE data SET location = %s, status = %s, response = %s WHERE uid=%s "
    val = ( data, "1", result, bot)
    mycursor.execute(update_users,val)
    mydb.commit()
    print("bot",bot)

    # ****************判斷式********************
    # if(result != "" and data != ""):
        # print("12",result)
        # bot = request.values.get('bot')

      

    # if result == "" or data == "":
    #     result = "Sorry. Can you say it again?"
    #     print("1",result)




    # ****************結束********************
    t = {
        'ans': data,
        'type': final_result
    }
    return jsonify(t)

####評分機制成功
@app.route('/rasaBotTest', methods=['POST'])
def rasaBotTest():
    bot = request.values.get('bot')
    try:
        url = 'http://140.125.32.141:5005/webhooks/rest/webhook'
        text = request.values.get('message')
        datajs = {"sender": bot, "message": text}
        datajs = json.dumps(datajs)
        x = requests.post(url, data = datajs)

        result = x.json()[0]['text']
        print("you:" + text + "  ans:" + result)

    except:
        print("888888888888888888888")
        url = 'http://140.125.32.141:5005/webhooks/rest/webhook'
        text = request.values.get('message')
        datajs = {"sender": bot, "message": "ABC"}
        datajs = json.dumps(datajs)
        x = requests.post(url, data = datajs)

        result = x.json()[0]['text']
        print("you:" + text + "  ans:" + result)

    ###句子比對
    f = open("1.txt","r",encoding="utf-8") 
    lines = [line.rstrip() for line in f]
    dict1 = {
        key: round(difflib.SequenceMatcher(None, text, key).quick_ratio(), 2) * 100
        for key in lines
    }
    score_int = max(dict1.values())
    score  = str(max(dict1.values()))
    ###value和key反轉取值
    dict3 = {v:k for k, v in dict1.items()}
    similar_words = dict3[score_int]
    # print("score", type(score))
    # print("similar_words",type(similar_words))
    scoreInt = float(score_int)
    # print("scoreInt",scoreInt)
    bot = request.values.get('bot')
    print("bottt",bot)
    sql = "UPDATE data SET location = %s, status = %s, response = %s, score = %s, similar_words = %s WHERE uid=%s "
    val = ( text, "1", result, score, similar_words, bot)
    mycursor.execute( sql, val)
    mydb.commit()
    print("222",score)


    # if scoreInt >= 80:
    #     sql = "UPDATE data SET location = %s, status = %s, response = %s, score = %s WHERE uid=%s "
    #     val = ( similar_words, "1", result, score, bot)
    #     mycursor.execute( sql, val)
    #     mydb.commit()
    #     print("00000000", score)
    # else:
    #     sql = "UPDATE data SET location = %s, status = %s, response = %s, score = %s WHERE uid=%s "
    #     val = ( text, "1", result, score, bot)
    #     mycursor.execute( sql, val)
    #     mydb.commit()
    #     print("222",score)

    word_array = text.split()
    t = {
        'score' : score,
        'similar_words': similar_words,
        'word_array': "word_array",
        'word_array_num': "word_array_num",
        'type': result
    }
    # print("t", t )
    return jsonify(t)

@app.route('/test0413', methods=['POST'])
def test0413():
    r = request.values.get('bot')
    m = request.values.get('message')

    print("r--- ",r)
    print("m--- ",m)

        


    t = {
        'score' : 'score',
        'similar_words': 'similar_words',
        'word_array': "word_array",
        'word_array_num': "word_array_num",
        'type': 'result'
    }
    return jsonify(t)


if __name__ == "__main__":
    r_animal = random.randint(0,19)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)