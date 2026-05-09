import os
import json
from typing import List
import warnings
warnings.filterwarnings('ignore')
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore, EncoderBackedStore


from langchain_text_splitters import RecursiveCharacterTextSplitter

# основной класс-контейнер, с которым работают все методы. используется для представления документов в формате маркдаун
# имеет полями page_content - текст файла и metadata - информация о файле
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import time


import torch

from pathlib import Path


from docling.datamodel.document import InputDocument, ConversionResult

from docling_core.types.doc import DoclingDocument, TableItem, PictureItem, TextItem


        

    

class DataBase:
    def __init__(self, doc_directory: str, vector_directory: str):
        self.doc_directory = doc_directory
        self.vector_directory = vector_directory
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'на карте: {device}')
        try:
            # загрузка модели sentance-transformers для получения векторного представления документов
            self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-mpnet-base-v2',      #'sentence-transformers/all-mpnet-base-v2',  'intfloat/multilingual-e5-small'     # название модели
            model_kwargs={'device': device},                    # параметры модели (можно добавить девайс)
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 64},      # нормализация векторов
            show_progress=True                               # можно убрать (отображает процесс загрузки модели)
            )
        except Exception as e:
            print(f'error loading embeddings: {e}')
            return
        self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=vector_directory
            )



    def add(self, chunks: List[Document]):
        batch_size = 2048  # безопасный размер батча
        
        # Разбиваем chunks на батчи вручную
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.vectorstore.add_documents(batch)

        


    def search(self, sha1: str, question: str) -> List[Document]:
        # Ищем только в этом документе
        filtered_chunks = self.vectorstore.similarity_search_with_score(
            query=question,
            k=10,
            filter={"pdf_sha1": sha1}
        )

        pages_index = list(set([
            page for chunk, score in filtered_chunks 
            if chunk.metadata.get('page_ind') is not None and score <= 0.8
            for page in chunk.metadata.get('page_ind')
        ]))

        pages = []
        for page_index in pages_index:
            path = self.doc_directory + '/' + sha1 + '/' + str(page_index) + '.md'
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    page_text = file.read()
                    page = Document(
                            page_content=page_text,
                            metadata={
                                "pdf_sha1": sha1,
                                "page_ind": page_index
                            }
                        )
            except Exception:
                pages.extend([Document(
                            page_content=chunk.page_content,
                            metadata={
                                "pdf_sha1": sha1,
                                "page_ind": page_index
                            }
                        ) 
                            for chunk, score in filtered_chunks 
                              if page_index in chunk.metadata.get('page_ind') and score <= 0.8])
            pages.append(page)
                
                
        pages.sort(key=lambda d: d.metadata.get('page_ind', 0))
        return pages
    
