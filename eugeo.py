import os
from dotenv import load_dotenv
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType

from langchain_google_calendar_tools.utils import build_resource_service, get_oauth_credentials
from langchain_google_calendar_tools.tools.create_new_event.tool import CreateNewEvent
from langchain_google_calendar_tools.tools.list_events.tool import ListEvents
from langchain_google_calendar_tools.tools.update_exist_event.tool import UpdateExistEvent

from langchain_google_calendar_tools.helper_tools.get_current_datetime import GetCurrentDatetime
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from pprint import pprint
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import warnings
warnings.filterwarnings("ignore")
from langchain.chains import LLMChain
import sys


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ------------------------ NECESSÁRIO POIS DESENVOLVI UTILIZANDO WSL --------------------------
import webbrowser

browser_path = "/usr/bin/firefox"
webbrowser.register('firefox', None, webbrowser.BackgroundBrowser(browser_path))
# ----------------------------------------------------------------------------------------------

credentials = get_oauth_credentials(
    client_secrets_file="credentials.json",
)

api_resource = build_resource_service(credentials=credentials)
agent = initialize_agent(
    tools=[
        ListEvents(api_resource=api_resource),
        CreateNewEvent(api_resource=api_resource),
        UpdateExistEvent(api_resource=api_resource),
        GetCurrentDatetime(),
    ],
    llm = ChatGroq(temperature=0, model="llama3-70b-8192"),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

# Data model
class RouteQuery(BaseModel):
    """Encaminhe uma consulta de usuário para o datasource mais relevante."""

    datasource: Literal["vectorstore", "websearch", "agendamento"] = Field(
        ...,
        description="Dada uma pergunta do usuário, escolha encaminhá-la para websearch, vectorstore ou agendamento.",
    )

# LLM with function call
llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt 
system = """Você trabalha na Tech4.ai e é um especialista em encaminhar uma pergunta do usuário para um vectorstore, pesquisa na web ou agendamento de reunião. \n
O vectorstore contém documentos relacionados a perguntas comuns apenas sobre a empresa (Tech4.ai), tais como missão, visão, valores, cultura, programas internos, políticas \n
de trabalho remoto, horários, etc. Use o vectorstore para perguntas sobre esses tópicos. \n
Para perguntas sobre as ferramentas Github, Vscode, Jira e discord use a pesquisa na web. \n
Para perguntas sobre agendamento de reuniões use o agendamento."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

# Load PDF document
pdf_path = "Data/Base.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split the combined documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs)

""" 
(VOLTAR SE SOBRAR TEMPO!)
O text_splitter não pega as imagens no final do pdf (Valores da Empresa). Tentativa de extrair texto das imagens:
# Extract text from images
image_paths = ["image1.PNG", "image2.PNG", "image3.PNG", "image4.PNG", "image5.PNG", "image6.PNG"]
image_texts = [extract_text_from_image(image_path) for image_path in image_paths]

doc_splitsIMAGES = text_splitter.split_documents(text_splitter.create_documents(image_texts)) 
"""

# Load an open-source embedding model from Hugging Face
embedding_function = SentenceTransformerEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Add to vector store
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="Empresa",
    embedding=embedding_function,
)

retriever = vectorstore.as_retriever()

# COMANDO PRA DELETAR O VECTORSTORE
# vectorstore = Chroma(collection_name="Empresa")
# vectorstore.delete_collection()

# Data model
class GradeDocuments(BaseModel):
    """Pontuação binária para verificar a relevância nos documentos utilizados."""

    binary_score: str = Field(description="Documentos são relevantes para a pergunta, 'sim' or 'não'")

# LLM with function call 
structured_llm_grader_docs = llm.with_structured_output(GradeDocuments)

# Prompt 
system = """Você é um avaliador que avalia a relevância de um documento recuperado para uma pergunta do usuário. \n
 Se o documento contiver palavra(s)-chave ou significado semântico relacionado à pergunta, classifique-o como relevante. \n
 Dê uma pontuação binária "sim" ou "não" para indicar se o documento é relevante para a pergunta."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader_relevance = grade_prompt | structured_llm_grader_docs

# Prompt
prompt = ChatPromptTemplate.from_template(
    """Você é um assistente para tarefas de resposta a perguntas. Use as seguintes partes do contexto recuperado para responder à pergunta. \n
      Se você não souber a resposta, apenas diga que não sabe. Sempre responda em português do Brasil.
Question: {question}
Context: {context}
Answer:"""
)
 
# Chain
rag_chain = prompt | llm | StrOutputParser()

# Data model
class GradeHallucinations(BaseModel):
    """Pontuação binária para alucinação presente na resposta obtida."""

    binary_score: str = Field(description="Não considere a possibilidade de chamar APIs externas para obter informações adicionais. A resposta é apoiada pelos fatos, 'sim' ou 'não'.")
 
# LLM with function call 
structured_llm_grader_hallucination = llm.with_structured_output(GradeHallucinations)
 
# Prompt 
system = """Você é um avaliador que avalia se uma resposta gerada por LLM é apoiada por um conjunto de fatos recuperados. \n 
     Restrinja-se a dar uma pontuação binária, seja "sim" ou "não". Se a resposta for apoiada ou parcialmente apoiada pelo conjunto de fatos, considere-a um sim. \n
    Não considere a chamada de APIs externas para obter informações adicionais."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
]
)
  
hallucination_grader = hallucination_prompt | structured_llm_grader_hallucination

# Data model
class GradeAnswer(BaseModel):
    """Pontuação binária para avaliar se a resposta responde a pergunta."""

    binary_score: str = Field(description="Responde responde a pergunta, 'sim' ou 'não'")

# LLM with function call 
structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)

# Prompt 
system = """Você é um avaliador que avalia se uma resposta aborda/resolve uma pergunta \n 
     Dê uma pontuação binária 'sim' ou 'não'. 'Sim' significa que a resposta resolve a pergunta."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader_answer

# Prompt
prompt = ChatPromptTemplate.from_template(
    """Você é um assistente que trabalha na empresa tech4.ai e deve fazer o atendimento de um novo funcionário. Pergunte como pode ajudá-lo hoje.
Answer:"""
)
 
# Chain
init_chain = prompt | llm | StrOutputParser()

web_search_tool = TavilySearchResults(k=3)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question : str
    generation : str
    web_search : str
    documents : List[str]

### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    #print("---RETRIEVE from Vector Store DB---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    #print("---GENERATE Answer---")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    #print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader_relevance.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        # Document relevant
        if grade.lower() == "sim":
            #print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            #print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
    
def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    #print("---WEB SEARCH. Append to vector store db---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

def meeting(state):
    """
    Call agent to schedule a meeting

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    #print("---SCHEDULE meeting---")
    question = state["question"]

    atual = agent.run("Qual o dia e a hora atual?")
    generation = agent.run(question+" "+atual)
    return {"question": question, "generation": generation}

### Edges

def route_question(state):
    """
    Route question to web search or RAG 

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    #print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})   
    if source.datasource == 'websearch':
        #print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source.datasource == 'vectorstore':
        #print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    elif source.datasource == 'agendamento':
        #print("---ROUTE QUESTION TO CALENDAR AGENT---")
        return "agendamento"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    #print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        #print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        #print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    #print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "sim":
        #print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        #print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score.binary_score
        if grade == "sim":
            #print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            #print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        #pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

    
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search) # web search # key: action to do
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("grade_documents", grade_documents) # grade documents
workflow.add_node("generate", generate) # generate
workflow.add_node("meeting", meeting) # meeting

workflow.add_edge("websearch", "generate") #start -> end of node
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_edge("meeting", END)
workflow.add_edge("generate", END)



# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
        "agendamento": "meeting",
    },
)
 
#RETIREI GRADE GENERATION PQ TEMPO ERA MUITO LONGO
""" workflow.add_conditional_edges(
    "generate", # start: node
    grade_generation_v_documents_and_question, # defined function
    {
        "not supported": "generate", #returns of the function
        "useful": END,               #returns of the function
        "not useful": "websearch",   #returns of the function
    },
) """

app = workflow.compile()

def analyzer(user_input):
    global var
    resp = verify_context(user_input)
    if(resp == "sim"):
        return
    else:
        var = 1984

def verify_context(user_input):
    chat = ChatGroq(
            temperature=0,
            model="llama3-70b-8192",
    )

    # Template to use for the system message prompt
    template =  """
    Você é um assistente de um funcionário, que tem a função de identificar se a solicitação {input}
    do usuário está condizente com um contexto de uma dúvida sobre a empresa, sobre algum software ou o desejo de marcar uma reunião.
    
    Se a solicitação estiver presente no contexto, responda "sim", caso contrário, responda "não".
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Verifique a frase: {input}"

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = chat_prompt | chat
    
    response = chain.invoke(input=user_input)
    #print(response.content.lower())
    return response.content.lower()



#classe principal do atendimento
class eugeo_service:
    def __init__(self):
        self._orientacao = "" # Atributo protegido
        self._history: List[str] = [] # Atributo protegido
        self._user_input = ""
        self._eugeo_output = ""
        self.contaUsuario = None  # Initialize contaUsuario as an attribute to be used 
        self.codRast = ""

    #get e set dos atributos
    @property
    def eugeo_output(self):
        return self._eugeo_output
    
    @eugeo_output.setter
    def eugeo_output(self,eugeo_output):
        self._eugeo_output = eugeo_output
        
    @property
    def orientacao(self):
        return self._orientacao
    
    @orientacao.setter
    def orientacao(self,nova_orientacao):
        self._orientacao = nova_orientacao
        
    @property
    def history(self):
        return self._history
    
    @history.setter
    def history(self,novo_history):
        self._history = novo_history
        
    # Função para buscar frases semelhantes no banco de dados
    def atendimento(self,user_input,orientacao):
        
        self._user_input = user_input
        self._orientacao = orientacao
        #pré análise do input do usuário
        chat = ChatGroq(
            temperature=0,
            model="llama3-70b-8192",
    )

        # Template to use for the system message prompt
        template =  """
        Você é um assistente que trabalha na empresa tech4.ai, seu nome é Eugeo. 
        Sua função é realizar o atendimento de um novo funcionário da empresa seguindo esta orientação:{orientacao}
        para tomar a ação.Não é necessáriamente o primeiro dia de trabalho do funcionário.
        Leve também em consideração o histórico da conversa abaixo após o símbolo
        '===' até o próximo símbolo'==='.
        ===
        {historico}.
        ===
        não diga olá nas frases depois da apresentação. 
        Não é necessário que se apresente de novo e nem mesmo cumprimente o usuário a cada ação.
        Nunca escreva '<FIM_TURNO_CLIENTE>' ou '<FIM_TURNO_ATENDENTE>' nas frases.
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        human_message_prompt = HumanMessagePromptTemplate.from_template(self.orientacao)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        
        chain = chat_prompt | chat

        response = chain.invoke({"input":user_input, "historico": self.history, "orientacao": self.orientacao})
        return response.content
    
    #guarda input do usuário no histórico
    def human_step(self,human_input):
        human_input = human_input + '<FIM_TURNO_CLIENTE>'
        self.history.append(human_input)

    #verifica se o usuário quer sair do atendimento      
    def exit(self,user_input):
        orientacao = " verifique através da fala {input} se o usuário quer sair do atendimento, se sim retorne apenas 'sim' se não retone apenas 'não'"
        eugeo_output = self.atendimento(user_input,orientacao)
        if eugeo_output =="sim":
            orientacao = "se dispeça do usuário. não é necessário dizer nada além disso"
            eugeo_output = self.atendimento(user_input,orientacao)
            print("\n Eugeo turn =>",eugeo_output)
            sys.exit()

    #guarda output da eugeo no histórico       
    def eugeo_step(self,eugeo_output):
            # process human input
            eugeo_output = eugeo_output + '<FIM_TURNO_ATENDENTE>'
            self.history.append(eugeo_output)
            
def flow(user_input):
    inputs = {"question": user_input}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint("")
    return value["generation"]

def main():
    global var
var = 0

#inicializa a classe eugeo_service
eugeo = eugeo_service()
flag = False
while True:
    user_input = " "
    #intrdução do atendimento     
    if var == 0:
        orientacao = "Apresente-se e pergunte como pode ajudar o novo funcionário"
    elif var == 1984:
        orientacao = "Diga que não entendeu, e peça mais detalhes"
        var = 0
    #retorno
    else:
        orientacao = "pergunte se pode ajudar o usuário em mais alguma coisa?"
        flag = True

    #turno da eugeo receber o input e orientacao para seguir
    eugeo_output = eugeo.atendimento(user_input,orientacao)
    print("\n Eugeo turn =>",eugeo_output) 
    eugeo.eugeo_step(eugeo_output)

    user_input = input("\nUser Input  => ")
    
    if var!=0:
        orientacao = "Voce perguntou se o usuario quer mais ajuda.verifique se na resposta dele:{input} ele quer mais ajuda ou não.Se sim diga apenas 'sim'. Se não diga apenas 'não'"
        eugeo_output = eugeo.atendimento(user_input,orientacao)
        eugeo.eugeo_step(eugeo_output)
        if eugeo_output == "não":
            orientacao = "se dispeça do usuário. não é necessário dizer nada além disso"
            eugeo_output = eugeo.atendimento(user_input,orientacao)
            print("\n Eugeo turn =>",eugeo_output) 
            break
    if flag == False:
        eugeo.exit(user_input)
    flag = False
    analyzer(user_input)

    if(var!=1984):
        #ATENDIMENTO
        orientacao = "responda ao {input} pedindo para o cliente aguardar enquanto você faz o que ele pediu "
        eugeo.human_step(user_input)
        eugeo_output = eugeo.atendimento(user_input ,orientacao)
        print("\n Eugeo turn =>",eugeo_output) 

        retorno = flow(user_input)   
        orientacao = "Sua busca por informações resultou no seguinte: {input}. Tudo o que você precisa fazer é passar essas informaçoes ao usuário. Apenas isso. Não responda o agente que te enviou as informações."
        eugeo_output = eugeo.atendimento(retorno , orientacao)
        print("\n Eugeo turn =>",eugeo_output) 

        eugeo.eugeo_step(eugeo_output)
        var += 1           
        
main()