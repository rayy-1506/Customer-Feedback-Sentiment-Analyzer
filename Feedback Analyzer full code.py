import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../DATA/airline_tweets.csv")
df.head()

sns.countplot(data=df,x='airline',hue='airline_sentiment')

sns.countplot(data=df,x='negativereason')
plt.xticks(rotation=90);

sns.countplot(data=df,x='airline_sentiment')

df['airline_sentiment'].value_counts()

#Features and labels
data = df[['airline_sentiment','text']]
data.head()

y = df['airline_sentiment']
X = df['text']

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(X_train)

X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
X_train_tfidf

#Model Comparison

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_tfidf,y_train)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression(max_iter=1000)
log.fit(X_train_tfidf,y_train)

from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train_tfidf,y_train)

from sklearn.metrics import plot_confusion_matrix,classification_report

def report(model):
    preds = model.predict(X_test_tfidf)
    print(classification_report(y_test,preds))
    plot_confusion_matrix(model,X_test_tfidf,y_test)
print("NB MODEL")
report(nb)

print("Logistic Regression")
report(log)

print('SVC')
report(svc)

#Finalizing pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([('tfidf',TfidfVectorizer()),('svc',LinearSVC())])
pipe.fit(df['text'],df['airline_sentiment'])

new_tweet = ['good flight']
pipe.predict(new_tweet)

new_tweet = ['bad flight']
pipe.predict(new_tweet)

new_tweet = ['ok flight']
pipe.predict(new_tweet)

