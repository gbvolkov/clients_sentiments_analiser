from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

import json
import re
import config

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=1)

rephrase_system_prompt_text = """Ты умеешь перефразировать тексты.
Перефразируй полученную фразу на русский язык.
Сохраняй внутреннюю струтуру фразы. Если внутри фраза содержит ответы на несколько ворросов, созхраняй все ключевые слова типа 'Ответ на вопрос 1', 'Ответ на вопрос 2'.
Когда перефразируешь, сохраняй смысл сказанного.
"""

prompt_rephrase = ChatPromptTemplate.from_messages(
        [
            ("system", rephrase_system_prompt_text),
            ("user", "{text}"),
        ]
    )

chain_rephrase = prompt_rephrase | llm

split_system_prompt_text = """Ты умеешь анализировать высказывания клиента относительно работы Компани и выделять аспекты, которые были отмечены, как положительные и как отрицательные.
(1) Внимательно проанализируй текст ответа клиента. 
(2) выдели аспекты деятельности Компании (персонал, процессы, коммуникация, скорость и так далее), которые были отмечены, как положительные (всё хорошо и не требует улучшений) и как отрицательные (плохо, или требует улучшений).
(3) Для каждого из аспектов объясни, почему этот аспект был веделен, по возможности используя части исходного ответа, которая позволила тебе определить этот аспект.
"""

prompt_split = ChatPromptTemplate.from_messages(
        [
            ("system", split_system_prompt_text),
            ("user", "{text}"),
        ]
    )

class AspectType(str, Enum):
    POS = "POS"
    NEG = "NEG"

class Aspect(BaseModel):
    aspect: Optional[str] = Field(default=None, description="Категория ответа (персонал, процессы, коммуникация, скорость и так далее), по которой ответ можно отнести к положительным или отрицательным. ")
    phrase: Optional[str] = Field(default=None, description="Объясни, что во фразе клиента позволило выделить этот аспект")

class Sentences(BaseModel):

    aspects: Optional[List[Aspect]] = Field(default=None, description="Список аспектов (POS/NEG) и фраз из ответа клиента")

structured_llm = llm.with_structured_output(Sentences)
chain_splitter = prompt_split | structured_llm

def split_sentences(text):
    result = chain_splitter.invoke({"text": answer})
    return result.json()

def rephrase(text: str):
    #match = re.match(r"Q:\s*(.*?)\s*A:\s*(.*)", text)
    #if match:
    #    question = match.group(1)
    #    answer = match.group(2)    
    #else:
    answer = text
    result = chain_rephrase.invoke({"text": answer})
    #return f"Q: {question} A: {result.content}"
    return result.content



if __name__ == "__main__":
    text= "Q: Расскажите, пожалуйста, что вам понравилось в работе с нами, а что необходимо улучшить? A: Ответ на 1 вопрос - 10; Ответ на 2 вопрос - Необходимо улучшить оперативность, понравился человечный подход к работе, ну грамотность, оперативность в части как проходит сделка по всем этапам, руководителя и менеджера."
    match = re.match(r"Q:\s*(.*?)\s*A:\s*(.*)", text)
    if match:
        question = match.group(1)
        answer = match.group(2)    
    #rephrased = rephrase(answer)
    #print(rephrased)

    split = split_sentences(answer)
    data = json.loads(split)              # now data is a dict, with keys like "sentences"
    sentences = data["aspects"]       # this is a Python list of str, with proper Cyrillic

    for idx, sent in enumerate(sentences, start=1):
        print(f"{idx}. {sent}")    
