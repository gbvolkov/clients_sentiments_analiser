import re
import config

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=1)

system_prompt_text = """Ты умеешь перефразировать тексты.
Перефразируй полученную фразу на русский язык.
Сохраняй внутреннюю струтуру фразы. Если внутри фраза содержит ответы на несколько ворросов, созхраняй все ключевые слова типа 'Ответ на вопрос 1', 'Ответ на вопрос 2'.
Когда перефразируешь, сохраняй смысл сказанного.
"""

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            ("user", "{text}"),
        ]
    )

chain = prompt | llm

def rephrase(text: str):
    #match = re.match(r"Q:\s*(.*?)\s*A:\s*(.*)", text)
    #if match:
    #    question = match.group(1)
    #    answer = match.group(2)    
    #else:
    answer = text
    result = chain.invoke({"text": answer})
    #return f"Q: {question} A: {result.content}"
    return result.content


if __name__ == "__main__":
    text= "Q: Расскажите, пожалуйста, что вам понравилось в работе с нами, а что необходимо улучшить? A: Ответ на 1 вопрос - У меня нет партнеров кому я могу рекомендовать брать лизинг; Ответ на 2 вопрос - Мне не понравилось что досрочное погашение нельзя делать, точнее его можно делать частично нельзя делать и что счет фактуры на аван стоит денег, если делаешь частично досрочное погашение что бы получить счет фактуры надо на аванс заплатить комиссию, это мне не понятно, потому что по итогу НДС в любом случае выходит одно и тоже, что вы дали на счет фактуры на аванс, что вы потом отдадите мне, одна и та же сумма выйдет, за что я должна была платить деньги было не понятно мне."
    result = rephrase(text)

    print(result)