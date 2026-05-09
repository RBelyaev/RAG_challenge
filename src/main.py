import json
import warnings
warnings.filterwarnings('ignore')
import gc
import torch

CONFIG = './config.json'
with open(CONFIG, 'r') as f:
    config = json.load(f)
PDF_DIRECTORY = config['pdfs_dir']
PAGES_DIRECTORY = config['pages_dir']

COMPANY_INFO = config['company_info']
QUESTIONS = config['questions']

PERSIST_DIRECTORY = config['pars_dir']
SUBMISSION_FILE = config['submission_file']

TEAM_EMAIL = config['team_email']
SUBMISSION_NAME = config['submission_name']

API_BASE = config['api_base']
API_KEY = config['api_key']
MODEL_NAME = config['model']

from parsing import PDFParser
from rag_vector_db import DataBase
from rag_process import RAGSystem, load_questions, load_company_info, AnswerSubmission


def main():
    try:
        questions = load_questions(QUESTIONS)
    except FileNotFoundError:
        print('questions file not found')
        return
    
    try:
        company_info = load_company_info(COMPANY_INFO)
    except FileNotFoundError:
        print('company info file not found')
        return

    pdf_parser = PDFParser(PAGES_DIRECTORY)
    chunks = pdf_parser.pdfs_to_chunks_process(PDF_DIRECTORY)

    del pdf_parser.converter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vector_store = DataBase(PAGES_DIRECTORY, PERSIST_DIRECTORY)
    vector_store.add(chunks)


    rag = RAGSystem(API_BASE, API_KEY, MODEL_NAME)
    answers = rag.process_questions(vector_store, company_info, questions)

    submission = AnswerSubmission(
            team_email=TEAM_EMAIL,
            submission_name=SUBMISSION_NAME,
            answers=answers
        )
        
    with open(SUBMISSION_FILE, 'w', encoding='utf-8') as f:
        f.write(submission.model_dump_json(indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
