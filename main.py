from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score
from train_test_maker import train_test

from fastapi import FastAPI



app = FastAPI()


@app.get("/isClickBait/{title}")
async def root(title : str):
    title = [title]

    X_train, y_train, X_test, y_test = train_test()

    classificador = Pipeline([
                            ('meu_vetorizador', CountVectorizer()),
                            ('meu_classificador', BernoulliNB())
                            ])


    classificador.fit(X_train, y_train)

    y_pred = y_test

    #sample = ["Top ten reasons why your boyfriend is cheating on you"]
    y_pred = classificador.predict(title)

    # acc = accuracy_score(y_test, y_pred)
    # print(acc)
    print(y_pred)

    if y_pred == 0:
        return {"message": "not clickbait"}
    else:
        return {"message": "clickbait"}

    # return {"message": y_pred}

 