import hashlib
from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings('ignore')
import gc
import torch


from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument, TableItem
from docling.datamodel.base_models import InputFormat

from langchain_core.documents import Document


def calculate_pdf_sha1(pdf_path: str) -> str:
    sha1 = hashlib.sha1()
    with open(pdf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha1.update(chunk)
    return sha1.hexdigest() 


class PDFParser:
    def __init__(self, pages_dir: str, layout_batch_size: int = 64, 
                 table_batch_size: int = 4, EMBED_MODEL: str = 'sentence-transformers/all-mpnet-base-v2', MAX_TOKENS: int = 300):
        
        accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.AUTO,
            num_threads=8
        )
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.layout_batch_size = layout_batch_size
        pipeline_options.table_batch_size = table_batch_size
        pipeline_options.accelerator_options = accelerator_options
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        self.chunker = HybridChunker(tokenizer=EMBED_MODEL, max_tokens = MAX_TOKENS)
        self.pages_dir = Path(pages_dir)



    def _extract_page_text(self, doc: DoclingDocument, target_page: int, pdf_sha1: str) -> str:
        page_parts = []
  
        for item, _level in doc.iterate_items():
            provs = getattr(item, "prov", None)
            if not provs:
                continue
            
            item_pages = {p.page_no for p in provs}
            if target_page not in item_pages:
                continue
            
            if isinstance(item, TableItem):
                try:
                    df = item.export_to_dataframe(doc=doc)
                    page_parts.append(df.to_markdown(index=False))
                except Exception:
                    text = getattr(item, "text", "").strip()
                    if text:
                        page_parts.append(text)
            else:
                text = getattr(item, "text", "").strip()
                if text:
                    page_parts.append(text)

        result = "\n\n".join(page_parts)

        doc_dir = self.pages_dir / pdf_sha1
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / f"{target_page-1}.md").write_text(result, encoding="utf-8")        
        


    def load_pdf(self, pdf_path: Path, pdf_sha1: str) -> DoclingDocument:
        result = self.converter.convert(str(pdf_path))
        doc = result.document

        for page_no in range(1, len(result.pages) + 1):
            self._extract_page_text(doc, page_no, pdf_sha1)

        return doc
    

    def doc_to_chunks(self, document: DoclingDocument, pdf_sha1: str) -> List[Document]:
        langchain_chunks = []
        
        for chunk in self.chunker.chunk(document):
            page_numbers = sorted(set(
                prov.page_no - 1 
                for item in chunk.meta.doc_items 
                for prov in item.prov
            ))

            lc_doc = Document(
                page_content=chunk.text,
                metadata={
                    "pdf_sha1": pdf_sha1,
                    "page_ind": page_numbers
                }
            )
            langchain_chunks.append(lc_doc)
        
        return langchain_chunks
    

    def pdfs_to_chunks_process(self, pdf_dir: str):
        chunks = []

        pdf_files = list(Path(pdf_dir).glob('*.pdf'))

        for pdf_path in pdf_files:
            pdf_sha1 = calculate_pdf_sha1(pdf_path)
            documents = self.load_pdf(pdf_path, pdf_sha1)
            chunk = self.doc_to_chunks(documents, pdf_sha1)
            chunks.extend(chunk)
            del documents
            del chunk

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return chunks