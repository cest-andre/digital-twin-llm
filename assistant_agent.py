from typing import List
import os
import torch
from pathlib import Path

from langchain.agents import tool
from langchain.schema import SystemMessage
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.utilities.jira import JiraAPIWrapper
from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders.sharepoint import SharePointLoader

from custom_chain import CustomChain
from custom_embeddings import CustomEmbeddings
from vectordb_assistant import VectorDBAssistant
from syndeia_utils import syndeia_prompt_template, get_syndeia_relations
from data_catalog_utils import datacat_prompt_template, get_datacat_assets
from sensors_utils import get_sensor_data

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig


def init_embeddings():
    #   Current best (without loadin4bit).
    # rag_tokenizer = AutoTokenizer.from_pretrained("WhereIsAI/UAE-Large-V1", device_map="auto")
    # rag_model = AutoModel.from_pretrained("WhereIsAI/UAE-Large-V1", torch_dtype=torch.float32, load_in_4bit=True)#.cuda()

    # preferred.
    # rag_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", device_map="auto")
    # rag_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()

    #   Mixtral test.
    rag_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct", device_map="auto")
    rag_model = AutoModel.from_pretrained("intfloat/e5-mistral-7b-instruct").to("cuda:0")

    embeddings = CustomEmbeddings(rag_model, tokenizer=rag_tokenizer)

    return embeddings


def init_chain_llm(tokenizer=None, model=None):
    #   Testing
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", device_map="auto", trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="auto", trust_remote_code=True)
    # embeddings = CustomEmbeddings(model.transformer.embd, tokenizer=tokenizer)

    #   Current best llm and embedder.
    # tokenizer = AutoTokenizer.from_pretrained("Undi95/Mistral-11B-OmniMix", device_map="auto")
    # model = AutoModelForCausalLM.from_pretrained("Undi95/Mistral-11B-OmniMix", device_map="auto")

    #   Current Sev1Tech preferred.
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True # This causes an inference error.
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained("yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B", device_map="auto", use_default_system_prompt=False)#, cache_dir='/home/ec2-user/SageMaker')
    model = AutoModelForCausalLM.from_pretrained("macadeliccc/WestLake-7B-v2-laser-truthy-dpo", quantization_config=nf4_config, device_map="auto")#, cache_dir='/home/ec2-user/SageMaker')

    llm = CustomChain(llm=None, tokenizer=tokenizer, model=model)

    return llm


def init_jira_agent(llm):
    os.environ["JIRA_API_TOKEN"] = ""
    os.environ["JIRA_USERNAME"] = ""
    os.environ["JIRA_INSTANCE_URL"] = ""

    jira = JiraAPIWrapper()
    toolkit = JiraToolkit.from_jira_api_wrapper(jira)
    agent = initialize_agent(
        toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    #   Perhaps obtain intermediate steps, organize and include as chat_history.
    # agent.return_intermediate_steps = True
    agent.max_iterations = 4

    return agent


def init_syndeia_assistant(embeddings, llm, refresh_db=True, write_path=None):
    vector_db_path = os.path.join(os.getcwd(), "vectordb", "syndeia")

    if not os.path.isdir(vector_db_path):
        os.mkdir(vector_db_path)

    dataload = None
    if refresh_db:
        dataload = get_syndeia_relations(write_path)

    assistant = VectorDBAssistant(llm, embeddings, vector_db_path, syndeia_prompt_template, dataload=dataload)

    return assistant.chain


def init_datacat_assistant(embeddings, llm, refresh_db=True, write_path=None):
    vector_db_path = os.path.join(os.getcwd(), "vectordb", "datacat")

    if not os.path.isdir(vector_db_path):
        os.mkdir(vector_db_path)

    dataload = None
    if refresh_db:
        dataload = get_datacat_assets(write_path)

    assistant = VectorDBAssistant(llm, embeddings, vector_db_path, datacat_prompt_template, retrieve_count=12, dataload=dataload)

    return assistant.chain


def init_sensor_agent(llm):
    agent = initialize_agent(
        [get_sensor_data], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    #   Perhaps obtain intermediate steps, organize and include as chat_history.
    # agent.return_intermediate_steps = True
    agent.max_iterations = 4

    return agent


def get_assistant_selection(question, llm):
    prompt = f"You will receive a question and your job is to decide which area this question pertains to.  A question will only pertain to ONE area so your decision must select only a SINGLE area.  \
Here is the list of areas:  1) syndeia  2) jira  3) datacat  4) sensors.  To aid in your decision making, here are some relevant keywords for each area.  These keywords or similar will typically be found in the question for the correct area.  \
Keywords:  syndeia - relation, affected, artifact.  jira - issue, project.  datacat - asset, catalog, model, documentation.  sensors - sensors, sensorid, temperature, humidity.  Your decision response must only specify the selected area.  Do not respond with any additional information.  \
Begin!  Question:  {question}  Decision:  "

    model_inputs = llm.tokenizer([prompt], return_tensors="pt").to("cuda")
    generated_ids = llm.model.generate(**model_inputs, max_new_tokens=32, temperature=0, top_p=1)
    results = llm.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    selection = results.replace(prompt, "")

    print(selection)
    return selection


def ask_assistant(assistant, question, clear_memory=False):
    response = None
    try:
        if clear_memory:
            assistant.memory.clear()

        response = assistant.run(question)
    except Exception as e:
        print("EXCEPTION")
        print(e)
        response = "The assistant triggered an exception.  The question could not be answered.  Please rephrase the question or try a different one."

    return {"llm_response": response}