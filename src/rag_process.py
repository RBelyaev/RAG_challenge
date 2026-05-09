import json
import re
from typing import List, Dict, Optional, Union, Literal
import warnings
warnings.filterwarnings('ignore')
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from rag_vector_db import DataBase

from typing import Optional

from langchain_openai import ChatOpenAI




class Question(BaseModel):
    text: str
    kind: Literal['number', 'name', 'boolean', 'names']

class SourceReference(BaseModel):
    pdf_sha1: str = Field(..., description='SHA1 hash of the PDF file')
    page_index: int = Field(..., description='Zero-based physical page number in the PDF file')

class Answer(BaseModel):
    question_text: Optional[str] = Field(None, description='Text of the question')
    kind: Optional[Literal['number', 'name', 'boolean', 'names']] = Field(None, description='Kind of the question')
    value: Union[float, str, bool, List[str], Literal['N/A']] = Field(..., description='Answer to the question')
    references: List[SourceReference] = Field([], description='References to the source material')

class AnswerSubmission(BaseModel):
    team_email: str = Field(..., description='Email that your team used to register')
    submission_name: str = Field(..., description='Unique name of the submission')
    answers: List[Answer] = Field(..., description='List of answers to the questions')


class RAGSystem:
    def __init__(self, api_base: str, api_key: str, model_id):
        self.llm = self._setup_llm(api_base, api_key, model_id)

    def _setup_llm(self, api_base: str, api_key: str, model_id: str):
        llm_endpoint = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=api_base,
            model=model_id, 
            temperature=0.05,
            timeout=60,
        )
        
        return llm_endpoint
    
    
    def _decomposition_prompt(self) -> str:
        return '''
        You are a question rephrasing system.
        Your task is to break down a comparative question into individual questions for each company mentioned.
        Each output question must be self-contained, maintain the same intent and metric as the original question, be specific to the respective company, and use consistent phrasing.
        Answer in the same language in which the question was asked.

        CRITICAL RULES:
        1. Output ONLY a valid raw JSON dictionary. 
        2. Do NOT wrap the JSON in markdown code blocks like ```json ... ```.
        3. Do NOT include any introductory or concluding text. No explanations.
        4. The keys must be the company names, and the values must be the specific questions.

        Example of expected output:
        {{"Apple": "What was the revenue of Apple in 2023?", "Microsoft": "What was the revenue of Microsoft in 2023?"}}
        
        Original comparative question: {question}
        Companies mentioned: {company}

        The answer is (JSON dictionary only): 
        '''
    


    def _search_for_main_thing_prompt(self) -> str:
        return '''
        Your task is to identify the thing being asked about in the question. 
        Your answer should be such that I can find an exact match in the text.
        Your answer should be as short as possible, but keep the main part of the question.
        Example: What was Apple's total revenue for the year presented in the report?
        Your answer should be: total revenue?

        Question: {question}
        '''

    

    
    def _summarizing_context_for_question(self) -> str:
        return '''
        You are an assistant who answers questions ONLY based on the documents provided.
        Your task is to answer the user's question using ONLY information from the context provided.

        Rules:
        1. The answer should contain all the information from the context, but without unnecessary words and repetitions, briefly, but without loss of meaning.
        2. Don't add anything from yourself — no guesses, assumptions, or outside knowledge.
        3. If there are several facts or details about the issue in the context, list them all.
        4. If the context does not contain an answer to the question, say "N/A".
        5. Do NOT use phrases like "According to the context" or "Based on the provided documents". State the facts directly.
        6. Answer in the same language in which the question was asked.

        Context example: "The company's shares increased by 20 percent during 2020 and amounted to $ 40 million, but in 2021 they fell by 10 percent and amounted to $ 36 million."
        Question example: "How much are shares worth in 2021?"
        An example of an answer: "The company's shares are worth 36 million dollars."
        
        Context:
        {context}

        Question: {question}

        The answer is:
        '''

    def _create_prompt_for_question_type(self, question: Question) -> str:
        base_instructions = '''
        You are an assistant who answers questions ONLY based on the documents provided.
        Your task is to answer the user's question using ONLY information from the context provided.
        
        Context:
        {context}
        
        Question: {question}
        
        '''
        
        if question.kind == 'number':
            return base_instructions + '''
            1. Find the exact number in the context (year, quantity, amount, etc.) that answers the question.
            2. The answer should be ONLY a number (integer or floating point).
            3. If the number is not found, answer "N/A".
            4. Do not round the number.
            5. CRITICAL: Check if the table or section has a scale note (e.g., 'in thousands', '000s', 'in millions'). If it does, you MUST multiply the raw number accordingly. For example, '21,970' in a 'thousands' table must be returned as 21970000.
            6. DO NOT add any words, currency symbols, or units of measurement. Output ONLY the digits.
            7. BE CAREFUL, a comma is used as a thousands separator (e.g., 1,200  means 1200), while a period indicates a decimal point (e.g., 1.5). The answer SHOULD be in the units specified in the question.
            
            Answer (only a number or N/A):
            '''
        
        elif question.kind == 'boolean':
            return base_instructions + '''
            1. Find an unambiguous answer in the context (yes/no, true/false).
            2. The answer should ONLY be "true" or "false" (lowercase).
            3. If the answer is not found, answer "N/A".
            4. Do not add explanations or any other words.
                        
            Response (only true/false or N/A):
            '''
        
        elif question.kind == 'names':
            return base_instructions + '''
            1. Find ALL the names, titles, headings, and enumerations in the context that ANSWER the QUESTION.
            2. The response must be a valid JSON array of strings.
            3. If you found only one name, return a single-element JSON array, e.g., ["Apple"].
            4. If the list is empty, return an empty array [].
            5. Do NOT wrap the JSON in markdown code blocks like ```json ... ```.
            6. Do not add explanations, just a JSON array.
            
            Response (JSON array only):
            '''
        elif question.kind == 'name':
            return base_instructions + '''
            1. Find ONLY ONE name, title, heading in the context that ANSWERS the QUESTION.
            2. The response must COMPLETELY REPEAT the form from the context.
            3. If you couldn't find it, return N/A.
            4. DON'T ADD ANY EXPLANATIONS, just a response from the context.

            Response (only ONE name FROM THE CONTEXT, a name and the like that ANSWERS THE QUESTION ASKED):
            '''



    def _main_thing_question(self, question: str):
        prompt_template = self._search_for_main_thing_prompt()
        PROMPT = ChatPromptTemplate.from_messages([
            ("user", prompt_template) 
        ])
        chain = PROMPT | self.llm
        result = chain.invoke({'question': question})
        return result.content

    
    def _decomposition_question(self, question: Question, company_to_hash: Dict[str, str]):
        company = ' '.join(list(company_to_hash.keys()))
        prompt_template = self._decomposition_prompt()
        PROMPT = ChatPromptTemplate.from_messages([
            ("user", prompt_template) 
        ])

        chain = PROMPT | self.llm
        result = chain.invoke({'question': question.text, 'company': company})
        return result
    


    def _create_llm_request_for_question(self, vector_store: DataBase, sha1: str, question: str, PROMPT: ChatPromptTemplate):
        chain = PROMPT | self.llm

        new_question = self._main_thing_question(question)
        doc_context = vector_store.search(sha1, new_question)

        context = "\n-------\n".join([doc.page_content for doc in doc_context])
        result = chain.invoke({'context': context, 'question': question})

        return result, doc_context
    

    def _create_llm_request_for_complex_question(self, vector_store: DataBase, company_to_hash: Dict[str, str], question: Question):
        decomposition = self._decomposition_question(question, company_to_hash)

        clean_json_text = decomposition.content.strip()
        clean_json_text = re.sub(r"<think>.*?</think>", "", clean_json_text, flags=re.DOTALL).strip()

        if "```json" in clean_json_text:
            clean_json_text = re.search(r"```json\s*(.*?)\s*```", clean_json_text, re.DOTALL).group(1)
        elif "```" in clean_json_text:
            clean_json_text = re.search(r"```\s*(.*?)\s*```", clean_json_text, re.DOTALL).group(1)
            
        try:
            simple_quetions = json.loads(clean_json_text.strip())
        except json.JSONDecodeError:
            print(f"Ошибка парсинга JSON! Сырой ответ модели был:\n{decomposition.content}")
        prompt_template = self._summarizing_context_for_question()
        PROMPT = ChatPromptTemplate.from_messages([
            ("user", prompt_template)
        ])
        
        context  = ''
        source_documents = []
        for company, simple_question in simple_quetions.items():
            new_context, docs = self._create_llm_request_for_question(vector_store, 
                                                                      company_to_hash[company], 
                                                                      simple_question, 
                                                                      PROMPT)
            
            context = context + company + ': ' + new_context.content + '\n'
            source_documents.extend(docs)

        complex_prompt_template = self._create_prompt_for_question_type(question)
        COMPLEX_PROMPT = ChatPromptTemplate.from_messages([
            ("user", complex_prompt_template)
        ])

        complex_chain = COMPLEX_PROMPT | self.llm
        result = complex_chain.invoke({'context': context, 'question': question.text})
        return result, source_documents
    

    def _take_company(self, question: Question, company_info: List[Dict[str, str]]) -> Dict[str, str]:
        question_text = question.text
        company_to_hash = {}

        for company in company_info:
            company_name = company['company_name']
            if company_name in question_text:
                company_to_hash[company_name] = company['sha1']
        
        return company_to_hash
    
    

    def _parse_answer_by_kind(self, answer_text: str, kind: str) -> Union[float, str, bool, List[str], Literal["N/A"]]:
        answer_text = answer_text.strip()

        if answer_text.upper() in ["N/A", "ОТВЕТ НЕ НАЙДЕН", "NOT FOUND", ""]:
            return "N/A"
        
        try:
            if kind == "number":
                numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                if numbers:
                    num_str = numbers[0]
                    if '.' in num_str:
                        return float(num_str)
                    else:
                        return int(num_str)
                return "N/A"
            elif kind == "boolean":
                true_variants = ["true", "да", "yes", "верно", "истина"]
                false_variants = ["false", "нет", "no", "неверно", "ложь"]
                
                answer_lower = answer_text.lower()
                if any(variant in answer_lower for variant in true_variants):
                    return True
                elif any(variant in answer_lower for variant in false_variants):
                    return False
                return "N/A"
            elif kind == "names":
                try:
                    json_match = re.search(r'\[.*\]', answer_text, re.DOTALL)
                    if json_match:
                        names = json.loads(json_match.group())
                        if isinstance(names, list):
                            if names:
                                return names
                            return 'N/A'   
                except:
                    return 'N/A'
                
                else:
                    return answer_text.strip().strip('"\'')
            elif kind == "name":
                return answer_text

        except Exception as e:
                print(f'error getting answer of type {kind}: {e}')
                return 'N/A'
        

    def process_questions(self, vector_store, company_info: List[Dict[str, str]], questions: List[Question]) -> List[Answer]:
        answers = []
        for _, question in enumerate(questions, 1):
            company_to_hash = self._take_company(question, company_info)
            if not company_to_hash:
                parsed_value = 'N/A'
                references = []
                answer = Answer(
                    question_text=question.text,
                    kind=question.kind,
                    value=parsed_value,
                    references=references
                )
                answers.append(answer)
                continue

            elif len(company_to_hash) == 1:
                sha1 = list(company_to_hash.values())[0]
                prompt_template = self._create_prompt_for_question_type(question)
                PROMPT = ChatPromptTemplate.from_messages([
                    ("user", prompt_template)
                ])
                result, source_documents = self._create_llm_request_for_question(vector_store, sha1, question.text, PROMPT)
            else:
                result, source_documents = self._create_llm_request_for_complex_question(vector_store, company_to_hash, question)

            parsed_value = self._parse_answer_by_kind(result.content, question.kind)
            references = []

            if parsed_value != 'N/A':
                for doc in source_documents:
                    references.append(SourceReference(
                        pdf_sha1=doc.metadata.get('pdf_sha1', 0),
                        page_index=doc.metadata.get('page_ind', 0)
                        )
                    )
            
            answer = Answer(
                question_text=question.text,
                kind=question.kind,
                value=parsed_value,
                references=references
            )
            
            answers.append(answer)
        
        return answers
    



def load_questions(questions_file: str) -> List[Question]:
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    questions = []
    for q_data in questions_data:
        if isinstance(q_data, dict) and 'text' in q_data and 'kind' in q_data:
            questions.append(Question(**q_data))
    return questions


def load_company_info(company_file: str) -> List[Dict[str, str]]:
    with open(company_file, 'r', encoding='utf-8') as f:
        company_data = json.load(f)
    
    if not isinstance(company_data, list):
        raise ValueError(
            f"Ошибка в файле {company_file}: ожидается массив (список), "
            f"получен {type(company_data).__name__}"
        )
    
    if len(company_data) == 0:
        raise ValueError(
            f"Ошибка в файле {company_file}: массив с информацией о компаниях пуст"
        )
    
    for idx, company in enumerate(company_data):
        if not isinstance(company, dict):
            raise ValueError(
                f"Ошибка в файле {company_file}: элемент {idx} должен быть словарем, "
                f"получен {type(company).__name__}"
            )
        
        if 'sha1' not in company:
            raise ValueError(
                f"Ошибка в файле {company_file}: элемент {idx} не содержит ключ 'sha1'"
            )
        
        if 'company_name' not in company:
            raise ValueError(
                f"Ошибка в файле {company_file}: элемент {idx} не содержит ключ 'company_name'"
            )
        
        if not company['sha1'] or not isinstance(company['sha1'], str):
            raise ValueError(
                f"Ошибка в файле {company_file}: элемент {idx}, поле 'sha1' "
                f"должно быть непустой строкой"
            )
        
        if not company['company_name'] or not isinstance(company['company_name'], str):
            raise ValueError(
                f"Ошибка в файле {company_file}: элемент {idx}, поле 'company_name' "
                f"должно быть непустой строкой"
            )
    
    return company_data
