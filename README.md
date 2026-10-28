# COVID-19 Tweet Sentiment Classification

Classifying 44,955 English COVID-19 tweets (March to April 2020) as Extremely Negative, Negative, Neutral, Positive or Extremely Positive, and a simpler 3-class version (Negative / Neutral / Positive). This began as an NLP exercise in my Certified Data Scientist training (IABAC / AFRICDSA); I rebuilt it with a baseline, cross-validated tuning, and an untouched test set. The write-up and code were produced with AI assistance and I reviewed the results.

## Results (scored once on the provided test set of 3,798 tweets)

| Task | Majority-class baseline | Logistic regression | Linear SVM |
|---|---|---|---|
| **3-class**: accuracy / macro-F1 | 40.7% / 0.19 | **78.4% / 0.752** | 78.4% / 0.751 |
| **5-class**: accuracy / macro-F1 | 24.9% / 0.08 | **54.2% / 0.551** | 53.2% / 0.545 |

Cross-validated macro-F1 on the training data (0.75 for 3-class, 0.56 for 5-class) matches the test scores, so the results are not a lucky split.

Per-class F1, logistic regression: 3-class: Negative 0.80, Positive 0.82, Neutral 0.64. 5-class: Neutral 0.62, Extremely Positive 0.58, Extremely Negative 0.53, Positive 0.52, Negative 0.50.

## What I found
- **Positive vs negative is easy; "how strongly" is hard.** Merging the "Extremely" labels takes accuracy from 54% to 78%. In the 5-class model, 44% of all mistakes are between the extreme and ordinary version of the *same* sentiment (for example Positive vs Extremely Positive), which even human labellers would disagree on.
- **Neutral is the weakest class** in the 3-class model (F1 0.64): short factual tweets and news links sit near the boundary.
- A simple linear model on word and word-pair (TF-IDF) features is already a strong baseline; I did not try deep models or pre-trained transformers, which would probably do better.

## Method
- Cleaning: lower-case; URLs and @mentions replaced with placeholders.
- Features: TF-IDF on word 1-2 grams (3 or more occurrences, top 60,000), fitted inside each cross-validation fold so no test information leaks into the vocabulary.
- Tuning: 3-fold stratified cross-validation on the training set only; the test set was scored once at the end.
- Models: multinomial logistic regression and a linear SVM, compared to a majority-class baseline.

## Limitations
- Labels are the dataset's own (they appear to be human-assigned, but I did not verify how). Tweets are from a short window early in the pandemic, so the model may not transfer to later periods.
- Only a linear bag-of-words model was tried; the hyperparameter search was small (C = 3 for logistic regression, 0.1 or 0.3 for the SVM).
- Tweets may contain personal information; the data is not included here.

## Run it

A notebook version, [`covid_tweet_sentiment.ipynb`](covid_tweet_sentiment.ipynb), walks through the same analysis step by step. Its outputs are built from the committed `results.json`; the full-pipeline section runs only if you add the raw data.
```
pip install pandas numpy scikit-learn
python sentiment.py             # reads ./data/Corona_NLP_train.csv and Corona_NLP_test.csv
python sentiment.py other/dir   # or pass the folder
```
Put the dataset's two CSVs (`Corona_NLP_train.csv`, `Corona_NLP_test.csv`, "Coronavirus tweets NLP - Text Classification" as supplied in my course materials) in a `data/` folder. I have not verified the licence, so the files are not committed. Results are written to `results.json`.
