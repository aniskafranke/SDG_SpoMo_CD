import re
import fitz
import pandas as pd
import spacy
from haystack.nodes import TextConverter, PreProcessor
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

"""
1)  get PDF file and process it page by page
        - delete pages with very little text
        - maybe after u_cleaning the page from numbers and special characters

        Problems: 
            - some statements concerning sustainability have just 180 characters

2)  clean noisy data from every page
        - titles (if possible)
        - header and footer
        - page numbers
        - leftovers of tables and graphs

3)  get sentences from page
        - concat the sentences that are split by hyphens

4) since some pdf documents cannot be mined - check somehow if it is "senseful" english text
        - check for correctness score
            --> did that with model "salesken/query_wellformedness_score" --> score higher than 0.3 = ok
        - also check for english
            --> "papluca/xlm-roberta-base-language-detection"


5)  put sentences together to match sdg-cd
        - 90 avg. word count
        - 3-6 sentences
"""

# init model for correctness score
tokenizer = AutoTokenizer.from_pretrained("salesken/query_wellformedness_score")
model = AutoModelForSequenceClassification.from_pretrained("salesken/query_wellformedness_score")

# init model for language
tokenizer_lang = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
model_lang= AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")


# load spacy tokenizer
nlp = spacy.load("en_core_web_md")
converter = TextConverter(remove_numeric_tables=True)


preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=120,
    split_respect_sentence_boundary=True,
)


def set_custom_boundaries(doc):
    for i, token in enumerate(doc):
        if token.text in ("’s", "'s"):
            doc[i].is_sent_start = False
        elif token.text in ("“", "‘") and i < len(doc) - 1:
            # opening quote
            doc[i+1].is_sent_start = False
        elif token.text in ("”", "’"):
            # closing quote
            doc[i].is_sent_start = False
    return doc



def get_text_bocks_from_pdf(filepath):


    docu = fitz.open(filepath)
    pages = []

    for page in docu:

        string = page.get_text()
        nor_text_blocks = []
        # page = docu.load_page(1)
        blocks = page.get_text("blocks")
        for x0, y0, x1, y1, lines, block_no, block_type in blocks:
            # ignore image blocks, ignore header and footer
            if (block_type == 0) and y1 < 800 and y1 > 80:

                normalized = per_line(lines)
                temp = re.sub(r'[^a-z ]+', '', normalized)
                if len(temp.split()) > 3:
                    nor_text_blocks.append(normalized)
        pages.append(nor_text_blocks)

    return pages


def per_line(text):
    nor_text = ""
    unfinished_line = ""
    # remove url
    text = re.sub(r"http\S+", "", text)
    text = text.replace("%", "")
    text = text.replace("*", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("|", "")
    text = text.replace(">", "")
    text = text.replace("<", "")
    text = text.replace("&", ",")
    text = text.replace(";", ",")

    for line in text.splitlines():
        # if we have an unfinished line saved, we should add it to the line
        if len(unfinished_line) > 0:
            line = "".join([unfinished_line, line])
            unfinished_line = ""

        line = line.strip()
        if line.endswith("-"):
            line = line[:-1]
            line = line.strip()
            unfinished_line = line
        else:
            nor_text = "".join([nor_text, (line + " ")])
    return nor_text


def sentencizer(blocktext):
    tok_text = nlp(blocktext)
    sents = []

    for sentence in tok_text.sents:
        sents.append(sentence.text)

    scores = calc_sent_correctness_score(sents)

    s_data = {
        'sent': sents,
        'score': scores
    }
    df = pd.DataFrame(s_data)
    return df, len(sents)


def calc_sent_correctness_score(sentences):
    res = []
    try:
        features = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        model.eval()
        scores = model(**features).logits

        scores = scores.detach().numpy()
        scores = scores.tolist()

        for s in scores:
            res.append(s[0])
    except:
        print("An exception occurred in the calculation of correctness. Continue with next.")

    return res


def get_language():
    # use ml model to check if language is english using an example sentence
    pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    text = open('data/storage/helper.txt', 'r').read()
    res = pipe(text)
    return (res[0]['label'] == 'en')

def get_snippets(blocks_in_doc):
    df = pd.DataFrame(columns=['pg_num', 'text', 's_count', 'word_ct'])
    sents_df = pd.DataFrame(columns=['sent', 'score'])
    pagenum = 1

    good_sents = 0
    bad_sents = 0

    for page_blocks in blocks_in_doc:
        for block in page_blocks:
            df_sents_block, amt_sents = sentencizer(block)
            # remove all rows with sent correctness < 0.3
            df_clean = df_sents_block[(df_sents_block['score'].astype(float) >= 0.30)]
            df_dirty = df_sents_block[(df_sents_block['score'].astype(float) < 0.30)]

            good_sents = good_sents + len(df_clean)
            bad_sents = bad_sents + len(df_dirty)


            block = ""

            for i in range(len(df_clean)):
                block = block + " " + df_clean.iloc[i, 0]

            if len(block.split()) > 5:
                data = {
                    'pg_num': pagenum,
                    's_count': amt_sents,
                    'text': block,
                }
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
                sents_df = pd.concat([sents_df, df_sents_block], ignore_index=True)
        pagenum = pagenum + 1

    return df, good_sents/(good_sents+bad_sents)


def sentences_splitter():

    oc_txt = converter.convert(file_path="data/storage/helper.txt", meta=None)[0]
    snippets = []
    docs_default = preprocessor.process(oc_txt)
    for doc in docs_default:
        snippets.append(doc.content)

    return snippets
def sentences_to_text_snippets(df):

    df_pages = df.groupby(["pg_num"]).word_ct.sum().reset_index()

    df_temp = df.groupby(["pg_num"])['text'].apply(lambda x: ' '.join(x)).reset_index()
    df_temp = df_temp.drop("pg_num", axis=1)

    df_pages['text'] = df_temp
    df = pd.DataFrame(columns=['pg_num', 'text', 'word_ct'])

    #iterate over pages
    for i in range(len(df_pages)):

        page_num = df_pages.iloc[i]['pg_num']
        page_wc = df_pages.iloc[i]['word_ct']
        all_text = df_pages.iloc[i]['text']

        f = open("data/storage/helper.txt", "w")
        f.write(all_text)
        f.close()

        #split the page text into snipptes of 90 words
        splits = sentences_splitter()

        for split in splits:
            data = {
                'pg_num': page_num,
                'text': split,
                'word_ct': len(split.split())
            }
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    isEng = get_language()
    return df, isEng


