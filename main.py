import openai
from langchain.embeddings import OpenAIEmbeddings
from api_key import API_KEY
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.prompts import PromptTemplate
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
import textwrap
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

openai.api_key = API_KEY
os.environ['OPENAI_API_KEY'] = API_KEY


# Функція для взаємодії з моделлю GPT-3.5-turbo
def generate_response(user_input):
    # Запускаємо модель GPT-3.5-turbo для отримання відповіді
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=user_input,
        max_tokens=4000,
        temperature=0.7,
        n=1,
        stop=None
    )

    # Отримуємо відповідь моделі
    reply = response.choices[0].text.strip().split('\n')[0]

    return reply

# Функція для покращення виводу відповіді
def print_response(response: str):
    print("\n".join(textwrap.wrap(response, width=100)))


try:
    os.environ["OPENAI_API_KEY"] = API_KEY

    model = OpenAI(temperature=0, max_tokens=4000)

    loader = TextLoader('conversation.txt', encoding='utf-8')

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    document = documents[0]

    document.__dict__.keys()

    index = VectorstoreIndexCreator().from_loaders([loader])

    query = """
    Ви Ілон Маск.
    Поясніть опоясніть що було написано в тексті.
    """
    template = """Ви Ілон Маск.
    
    {context}
    
    Відповідайте, використовуючи манери Ілона Маска.
    
    Питання: {question}
    Відповідь:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embeddings)

    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs=chain_type_kwargs,
    )

    query = "Поясніть про що була розмова"

    response = chain.run(query)
    print_response(response)
except:
    print_response('Це Ваш перший чат')


with open('conversation.txt', 'a', encoding='utf-8') as file:
# Головний цикл програми
    while True:

        print("\n\nЩоб завершити розмову напишіть 'Завершити'\n\n")
        # Отримуємо введення користувача
        user_input = input('User: ')

        if user_input == 'Завершити' or user_input == 'завершити':
            print_response('Дякую за розмову!!')
            break

        file.write(f"User: {user_input}"+'\n')

        response = generate_response(user_input)
        file.write(f"AI: {response}"+'\n')

        # Виводимо відповідь моделі
        print_response(f'AI: {response}')

