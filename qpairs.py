import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("./train.csv").fillna("")
df['q1len'] = df['question1'].str.len()
df['q2len'] = df['question2'].str.len()

df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))

def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(),
            row['question1'].split(" ")))

        w2 = set(map(lambda word: word.lower().strip(),
                row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
df['word_share'] = df.apply(normalized_word_share, axis=1)
scaler = MinMaxScaler().fit(df[['q1len', 'q2len', 'q1_n_words', 'q2_n_words',
    'word_share']])

X = scaler.transform(df[['q1len', 'q2len', 'q1_n_words', 'q2_n_words',
    'word_share']])
y = df['is_duplicate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
        random_state=42)
clf = LogisticRegression()
grid = {
            'C': [1e-6, 1e-3, 1e0],
                'penalty': ['l1', 'l2']
       }
cv = GridSearchCV(clf, grid, scoring='neg_log_loss', n_jobs=-1, verbose=1)
cv.fit(X_train, y_train)
for i in range(1, len(cv.cv_results_['params'])+1):
    rank = cv.cv_results_['rank_test_score'][i-1]
    s = cv.cv_results_['mean_test_score'][i-1]
    sd = cv.cv_results_['std_test_score'][i-1]
    params = cv.cv_results_['params'][i-1]
    print("{0}. Mean validation neg log loss: {1:.3f} (std:{2:.3f}) - {3}".format(
        rank,
        s,
        sd,
        params
    ))
print(cv.best_params_)
print(cv.best_estimator_.coef_)
dftest = pd.read_csv("./test.csv").fillna("")

dftest['q1len'] = dftest['question1'].str.len()
dftest['q2len'] = dftest['question2'].str.len()

dftest['q1_n_words'] = dftest['question1'].apply(lambda row: len(row.split(" ")))
dftest['q2_n_words'] = dftest['question2'].apply(lambda row: len(row.split(" ")))
dftest['word_share'] = dftest.apply(normalized_word_share, axis=1)
print(dftest.head())
retrained = cv.best_estimator_.fit(X, y)

X_submission = scaler.transform(dftest[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_share']])
y_submission = retrained.predict_proba(X_submission)[:,1]
submission = pd.DataFrame({'test_id': dftest['test_id'], 'is_duplicate': y_submission })
print(submission.head())
sns.distplot(submission.is_duplicate[0:2000])
submission.to_csv("submission.csv", index=False)
