# TECHLAB_Eugeo

Repositório destinado ao desafio de IAG do TechLab

## Objetivo
Desenvolver um agente conversacional que facilite a integração de novos funcionários na Tech4ai, ajudando-os a se familiarizar rapidamente com a cultura, políticas, programas e ferramentas de trabalho da empresa.

## Tecnologias utilizaddas
- Groq API
- Python
- Hugging Face
- Langchain
- LangGraph
- Tavily API
  
## Installação

#### 1. Clone the repository

```bash
git clone git@github.com:matgaldino/TECHLAB_Eugeo.git
```

#### 2. Criando ambiente Python

Python 3.10 ou acima usando `venv` :

``` bash
cd TECHLAB
python3 -m venv env
source env/bin/activate
```
#### 3. Intallando as dependências
``` bash
pip install -r requirements.txt
```

#### 4. Set up the keys in a .env file

Crie `.env` no diretório principal do projeto. Dentro do arquivo, adcione sua OpenAI API key:

```makefile
GROQ_API_KEY="your_api_key_here"
TAVILY_API_KEY="your_api_key_here"
```
### 5. configurando arquivo JSON google cloud
Vá para google cloud console :https://console.cloud.google.com/
selecione seu projeto, vá para:
"APIs & Services" > "Credentials".
clique em  "Create credentials" e selecione "ID do cliente OAuth".
preencha as informações necessárias e crie a chave no formato JSON.

tenha certeza que ela contém os campos necessários:
client_id, project_id, auth_uri, token_uri, auth_provider_x509_cert_url, client_secret, redirect_uris.

Em seu código, verifique se você está fornecendo o caminho correto para o arquivo de chave JSON.  O caminho deve ser absoluto ou relativo ao local do script.  Verifique novamente o caminho para se certificar de que é preciso.
por exemplo:
```python
credentials = get_oauth_credentials(
    client_secrets_file="credentials.json",
)
```
Depois de confirmar que seu arquivo de chave JSON tem os campos obrigatórios e o caminho está correto, tente executar o código novamente usando a chave JSON válida.

## Rodando
No seu terminal rode o código principal:
```python
python3 eugeo.py
```

### Observações
- Se atente ao seguinte trecho de código pois o desenvolvimento do projeto foi em wsl, caso necessário comente o trecho a seguir:
```python
# ------------------------ NECESSÁRIO POIS DESENVOLVI UTILIZANDO WSL --------------------------
import webbrowser

browser_path = "/usr/bin/firefox"
webbrowser.register('firefox', None, webbrowser.BackgroundBrowser(browser_path))
# ----------------------------------------------------------------------------------------------
```
- É necessário realizar o login com uma conta do google para realizar os agendamentos de reunião
- Erros de data e horário ocorrem pois o agente não está com o fuso horário brasileiro. Está com o fuso horário GMT+7, ou seja, tem uma diferença de 10h com o fuso brasileiro.
- As inferências que utilizam RAG podem demorar alguns minutos para retornar.
- Foi verificado que às vezes a função de agendar a reunião da o seguinte erro 'datetime.timezone' object has no attribute 'zone'. Rodar novamente geralmente resolve o erro.