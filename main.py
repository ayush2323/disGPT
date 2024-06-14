import discord
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

my_secret = os.environ['DISCORD_KEY']
api_key = os.environ['GOOGLE_API_KEY']

#Load the models
llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=api_key, temperature=0.9)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)

#Load the PDF and create chunks
loader = PyPDFLoader("handbook_removed.pdf")
text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
pages = loader.load_and_split(text_splitter)

#Turn the chunks into embeddings and store them in FAISS
vector_db = FAISS.from_documents(pages, embeddings)

#Configure FAISS as a retriever with top_k=5
retriever = vector_db.as_retriever(search_kwargs={"k": 5})


prompt_template = """
You are a helpful AI assistant that will help user to give answer in simple words like a friend
context: {context}
input: {input}
answer:
"""

prompt = PromptTemplate.from_template(prompt_template)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


class MyClient(discord.Client):

    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        if self.user != message.author and self.user in message.mentions:
            response=retrieval_chain.invoke({"input":message.content})
            channel = message.channel
            print(response["answer"])
            messageToSend = response["answer"]
            await channel.send(messageToSend)


intents = discord.Intents.default()
intents.message_content = True

my_client = MyClient(intents=intents)
my_client.run(my_secret)
