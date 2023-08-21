from a_preprocessing_pdfs import process_all_pdfs
from b_SDG_extraction_from_snippets import add_TtS_AI_info
from c_appy_thresholds import apply_thresholds

if __name__ == '__main__':
    process_all_pdfs.main()
    add_TtS_AI_info.main()
    apply_thresholds.main()