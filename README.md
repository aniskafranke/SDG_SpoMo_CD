# SDG Spotter Model 

The SDG SpoMo can be used to assess SDG relevance in the corporate domain. 

It includes: 
- text extraction from a PDF based databases (of company reports)
- Document Extractability and Language Check
- Extraction of textual data, cleaning and tokenization, merging to text snippets
  
- NLP-based Recognition of SDG relevant text (SDG Recognizer model)
- NLP-based classification of text to one SDG (SDG Classifier model trained on OSDG-CD available here: https://zenodo.org/record/7816403 - Thanks OSDG!)

- application of predefined thresholds for:
    - confidence scores (suggestion: 0.75)
    - SDG relevance per company based on the distribution (suggestion: 0.737)
 
    
# usage

To use SDG SpoMo clone the repository and change the config.

LOCAL_FOLDER_PATH needs to point towards your PDF data. 
The folder with PDF company report data needs to have the following structure:

  - directoryname
    - companyname 1
        - report1.pdf
        - report2.pdf
    - companyname 2
        - report1.pdf
              ....    

Thresholds can be adjusted in the config.

Run 
1. process all PDFs
2. add TtS
3. apply Thresholds
