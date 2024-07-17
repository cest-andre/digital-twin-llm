from http.server import BaseHTTPRequestHandler, HTTPServer
import random
import json
import os
from datetime import datetime
from pytz import timezone

from assistant_agent import init_embeddings, init_chain_llm, init_syndeia_assistant, init_jira_agent, init_datacat_assistant, init_sensor_agent, ask_assistant, get_assistant_selection
from sensors_utils import get_sensor_data


#   TODO:  make dataload refresh a script arg.  If true, then delay the initialization of llm until after load to save memory.  Then init and pass
#          to functions to create the vectordb assistants.

print("Initializing llm")
embs = init_embeddings()
llm = init_chain_llm()

#   For testing convenience
# embs=None
# llm=None

# llm = init_chain_llm(embs.tokenizer, embs.embedder)

# init_sharepoint_assistant()
# exit()
# jira_agent = init_jira_agent(llm)

# syndeia_write_path = '/home/andrelongon/Documents/llm_experiments/syndeia_data/syndeia_relations.json'
# syndeia_assistant = init_syndeia_assistant(embs, llm, refresh_db=False, write_path=syndeia_write_path)

datacat_write_path = '/home/andrelongon/Documents/llm_experiments/datacat/assets.json'
datacat_assistant = init_datacat_assistant(embs, llm, refresh_db=True, write_path=datacat_write_path)

# sensor_agent = init_sensor_agent(llm)
# stored_template = sensor_agent.agent.llm_chain.prompt.template

#   TODO:  possibly split prompt at certain point, add current datetime, and piece back together and set.  Do at each call to sensor agent.

#base_tool_desc = get_sensor_data.description


""" The HTTP request handler """
class RequestHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        """ Sets headers required for CORS """
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "x-api-key,Content-Type")


    def send_dict_response(self, d):
        """ Sends a dictionary (JSON) back to the client """
        self.wfile.write(bytes(json.dumps(d), "utf8"))


    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()


    def do_GET(self):
        pass
        # self.send_response(200)
        # self._send_cors_headers()
        # self.end_headers()

        # response = {}
        # response["status"] = "GET SUCCESS"
        # self.send_dict_response(response)


    def do_POST(self):
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        dataLength = int(self.headers["Content-Length"])
        data = json.loads(self.rfile.read(dataLength).decode("UTF-8"))

        selection = get_assistant_selection(data['question'], llm)

        response = None
        print(f'Ask Agent Request.  Question: "{data["question"]}"')
        if "jira" in selection:
            response = ask_assistant(jira_agent, data["question"])
        elif "syndeia" in selection:
            response = ask_assistant(syndeia_assistant, data["question"], clear_memory=True)
        elif "datacat" in selection:
            response = ask_assistant(datacat_assistant, data["question"], clear_memory=True)
        elif "sensors" in selection:
            #   Update tool desc to contain current datetime so that it appears in the agent's prompt for context.
            # temp_split = stored_template.split("hour:minutes:seconds")
            # sensor_agent.agent.llm_chain.prompt.template = temp_split[0] + "hour:minutes:seconds"
            # sensor_agent.agent.llm_chain.prompt.template += f"\nThe user may provide times in a relative way (e.g., 'What was the sensor's reading yesterday?'). The current local time is: {datetime.now(timezone('US/Central')).strftime('%m/%d/%Y %H:%M:%S')}\nThe value of startTime and endTime must be directly inferred from the current local time.  For example, if current local time is ""01/01/2024 00:00:00"" and the user wants to know data about yesterday, the datetime of yesterday is ""12/31/2023 00:00:00"""
            # sensor_agent.agent.llm_chain.prompt.template += temp_split[1]

            response = ask_assistant(sensor_agent, data["question"])

            # sensor_agent.agent.llm_chain.prompt.template = stored_template

        self.send_dict_response(response)


#   Catch address currently used exception for torchrun mp=2.
print("Starting server")
httpd = HTTPServer(("127.0.0.1", 8000), RequestHandler)
print("Hosting server on port 8000")
print("Server setup complete.  Now serving forever...")
httpd.serve_forever()