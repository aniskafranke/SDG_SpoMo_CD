import os

import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)


def main():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # get model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_classifier = AutoModelForSequenceClassification.from_pretrained(
         'Nisk99/TtS_AI_Classifier',
         id2label={0: '1', 1: '10', 2: '11', 3: '12', 4: '13', 5: '14', 6: '15', 7: '16', 8: '2', 9: '3', 10: '4', 11: '5',
                  12: '6', 13: '7', 14: '8', 15: '9'})
    pipe_classifier = pipeline('text-classification', model=model_classifier, tokenizer=tokenizer)


    model_recognizer = AutoModelForSequenceClassification.from_pretrained("Nisk99/TtS_AI_Recognizer", id2label={0: 'False', 1: 'True'})
    pipe_recognizer = pipeline('text-classification', model=model_recognizer, tokenizer=tokenizer)

    # ---------------------------------------------- get the all text snippets ----------------------------------------------
    path = "data/storage/snippets.csv"
    df = pd.read_csv(path, sep=",",
                     engine='python', on_bad_lines='skip')

    df["text"] = df["text"].astype(str)
    df["filename"] = df["filename"].astype(str)


    # ---------------------------------------------- run through models --------------------------------------------------
    # erst sdg recognizer, dann auch mit classifier
    res_df = pd.DataFrame(columns=['label', 'score'])
    res_label = []
    res_score = []

    for text in df['text']:
        try:
            if (pipe_recognizer(text)[0]['label'] == "True"):
                res = pipe_classifier(text)
                # res_df = pd.concat([res_df, pd.DataFrame(res)], ignore_index=True)
                res_label.append(res[0]['label'])
                res_score.append(res[0]['score'])
            else:
                res_label.append('0')
                res_score.append(0)
        except:
            print("error occured")
            res_label.append('33')
            res_score.append(0)

    df["sdg"] = res_label
    df["score"] = res_score

    output_path = 'data/storage/classified_snippets.csv'
    df.to_csv(output_path, header=True)
    print("Finished processing text snippets and saved them to classified_snippets.csv")


