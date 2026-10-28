Put your trained text models here (both are optional):

  models/tfidf_logreg.joblib        <- scikit-learn Pipeline (TF-IDF + LogisticRegression)
  models/distilbert_fake_news/      <- folder from model.save_pretrained() + tokenizer.save_pretrained()

If a file or folder is missing, the app shows a warning and skips that model.
