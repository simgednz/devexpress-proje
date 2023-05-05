from rest_framework.response import Response
from rest_framework.decorators import api_view 

import pandas as pd
from sklearn.model_selection import train_test_split 

import pyodbc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle


@api_view(['GET'])
def TrainModel(request):
    cnxn = pyodbc.connect('Driver={SQL Server};'
                      'Server=.;'
                      'Database=DenemeBlazor;'
                      'Trusted_Connection=yes;')
    veri = pd.read_sql('SELECT * FROM dbo.KeyWord', cnxn)
    veri.head()
    x_set = veri['Name']
    y_set = veri['Problem']
    count_vect = CountVectorizer()
    text_counts= count_vect.fit_transform(x_set)
    X_train, X_test, y_train, y_test = train_test_split(text_counts, y_set, test_size=0.25, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    pickle.dump(model, open('trained_model.plk', 'wb'))
    pickle.dump(count_vect, open('vectorizer.plk', 'wb'))

    return Response(None)


@api_view(['GET'])
def SearchProblem(request):
    ticket = request.query_params.get('ticket')
    loaded_model = pickle.load(open('trained_model.plk', 'rb'))
    loaded_vectorizer = pickle.load(open('vectorizer.plk', 'rb'))
    
    ticket = [ticket]

    transformed = loaded_vectorizer.transform(ticket)
    problem = loaded_model.predict(transformed)

    return Response(problem)