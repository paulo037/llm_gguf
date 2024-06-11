from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
import gradio as gr


template= """# <|system|>
You are a chatbot tasked with responding to questions about an SQLite database.
Your responses must always consist of valid SQL code, and only that.
If you are unable to generate SQL for a question, respond with 'I do not know'.

# <|ddl|>
The query will run on a database with the following schema:
{schema}

# <|user|>
{question}

# <|assistant|>
[SQL]"""


llm = LlamaCpp(
    model_path="FP16.gguf",
    temperature=0.3,
    max_tokens=2048,
    top_p=1,
    n_ctx=2048,
    stop=["[/SQL]"],
 # Verbose is required to pass to the callback manager
)

prompt = PromptTemplate.from_template(template)
chain = (prompt | llm)

def fn_chain(schema, question):
    return chain.invoke({"schema":schema, "question":question}).split(";")[0] + ";"

# Gradio interface
iface = gr.Interface(
    fn=fn_chain,
    inputs=["text", "text"],
    outputs="text",
    title="TEXT2SQL",
    description="Digite o esquema do banco de dados juntamente com uma pergunta e obtenha a consulta para sua resposta.",
)

# Launch the app
iface.launch(server_name = "0.0.0.0", 
    server_port= 5000)
