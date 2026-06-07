# Fine Tuning Bert For Sentiment Anaysis

This project fine-tunes BERT (bert-base-uncased) for a **3-class sentiment classification** task using the syedkhalid0/Sentiment-Analysis dataset from HuggingFace, which contains ~105K English text samples labeled as Negative (0), Neutral (1), and Positive (2), split across train (84K), validation (10.5K), and test (10.5K) sets. 

The dataset covers diverse text sources including tweets, product reviews, and movie reviews, pre-processed to remove duplicates and null values for consistency. 

Built using **HuggingFace transformers** and datasets libraries, the pipeline handles tokenization via **AutoTokenizer**, dynamic padding via **DataCollatorWithPadding**.



``from transformers import pipeline

model_id = "edaUsha/Fine_Tuning_Bert_For_Sentiment_Anaysis"
clf = pipeline("text-classification", model=model_id)

print(clf("I really loved this movie!"))
print(clf("This was the worst experience ever."))``
