import os
import re
import pandas as pd
from pathlib import Path
from a_preprocessing_pdfs import pdf_to_text_snippets as p

# constants
from config import LOCAL_FOLDER_PATH


def get_all_filepaths(path):
    directory = os.fsencode(path)
    filepaths_array = []
    for file in os.listdir(directory):
        if not os.fsdecode(file).startswith('.'):
            filepath = os.path.join(path, os.fsdecode(file))
            filepaths_array.append(filepath)
    return filepaths_array

def clean_file_path_with_companies_csv(filepaths):
    df = pd.read_csv('/Users/aniskafranke/Desktop/MA_data/full run_18_06_output/companies.csv',
                     engine='python')
    df = df.loc[df.lang_isEn == "True"]
    filepaths_cleaned = []
    for file in filepaths:
        filename = Path(file).stem
        if df['name'].eq(filename).any():
            filepaths_cleaned.append(file)
    return filepaths_cleaned

def get_all_filepaths_subdirectories(path):
    res = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        for file in file_names:
            if not os.fsdecode(file).startswith('.'):
                res.append(os.path.join(dir_path, file))
    return res


def main():
    filepaths = get_all_filepaths_subdirectories(LOCAL_FOLDER_PATH)
    cleaned_filepaths = filepaths

    result = pd.DataFrame()
    res_ex_lang = pd.DataFrame(columns= ['filename', 'companyname', 'isEng', 'extractability'])

    for filepath in cleaned_filepaths:

        filename = Path(filepath).stem
        subdirname = os.path.basename(os.path.dirname(filepath))

        print("Parsing blocks from the following document: "+ str(filepath))
        blocks_in_doc = p.get_text_bocks_from_pdf(filepath)
        dataframe, wellformedness = p.get_snippets(blocks_in_doc)
        comp_res, isEng = p.sentences_to_text_snippets(dataframe)


        #add rep type, filename and company name
        comp_res["filename"] = filename
        comp_res["companyname"] = subdirname


        pattern = re.compile("(_[A-Z]{2,3}_)")
        rep_type = ""
        if (pattern.search(filename) != None):
            rep_type = pattern.search(filename).group(1)
        comp_res["rep_type"] = rep_type

        result = pd.concat([result, comp_res])


        new_row = {
            'filename': filename,
            'companyname': subdirname,
            'isEng': isEng,
            'extractability': wellformedness
        }

        res_ex_lang.loc[len(res_ex_lang)] = new_row

    os.remove("data/storage/helper.txt")
    output_path = "data/storage/snippets.csv"
    result.to_csv(output_path, header=True, index=False)

    output_path = "data/storage/ex_lang.csv"
    res_ex_lang.to_csv(output_path, header=True, index=False)

