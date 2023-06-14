#!/usr/bin/env python
# coding: utf-8
import streamlit as st

st.set_page_config(page_title="Genesis Demo", page_icon=":robot:")


import os
import getpass
import pandas as pd
import re

# Use environment variable to set OpenAI API key if already set (just leave this code commented out if already set)
# secret_key = getpass.getpass("Enter OpenAI secret key: ")
# os.environ["OPENAI_API_KEY"] = secret_key


# Or use Vicuna via Shale Protocol (free tier)

# os.environ['OPENAI_API_BASE'] = "https://shale.live/v1"
# shale_api = getpass.getpass('Enter Shale API key: ')
# os.environ['OPENAI_API_KEY'] = shale_api


# <a href="https://colab.research.google.com/github/ryderwishart/biblical-machine-learning/blob/main/gpt-inferences/greek-hebrew-tsv-qa-agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Question answer over MACULA Greek and Hebrew

# In[1]:

# !pip install tabulate pandas langchain

# ## Set up MACULA dataframes
verse_df = pd.read_csv("preprocessed-macula-dataframes/verse_df.csv")
mg = pd.read_csv("preprocessed-macula-dataframes/mg.csv")
# mh = pd.read_csv("preprocessed-macula-dataframes/mh.csv")

# ## Set up QA agent

# Use OpenAI

import pandas as pd
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent

macula_greek_agent = create_pandas_dataframe_agent(
    OpenAI(temperature=0),
    # mg, # verse_df (?)
    verse_df,
    # verbose=True,
    max_tokens_limit=3000,
    max_tokens_per_prompt=1000,
)

# # Expand functionality for more tools using DB lookups

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

# bible_persist_directory = '/Users/ryderwishart/genesis/databases/berean-bible-database'
bible_persist_directory = "/Users/ryderwishart/genesis/databases/berean-bible-database"
bible_chroma = Chroma(
    "berean-bible", embeddings, persist_directory=bible_persist_directory
)
# print(bible_chroma.search("jesus speaks to peter", search_type="similarity", k=1))

encyclopedic_persist_directory = "/Users/ryderwishart/biblical-machine-learning/gpt-inferences/databases/encyclopedic"
encyclopedic_chroma = Chroma(
    persist_directory=encyclopedic_persist_directory,
    embedding_function=embeddings,
    collection_name="encyclopedic",
)
# print(
#     encyclopedic_chroma.similarity_search_with_score(
#         "What is a sarcophagus?", search_type="similarity", k=1
#     )
# )

theology_persist_directory = (
    "/Users/ryderwishart/biblical-machine-learning/gpt-inferences/databases/theology"
)
theology_chroma = Chroma(
    "theology", embeddings, persist_directory=theology_persist_directory
)
# print(theology_chroma.search("jesus speaks to peter", search_type="similarity", k=1))

# # persist_directory = '/Users/ryderwishart/genesis/databases/itemized-prose-contexts copy' # NOTE: Itemized prose contexts are in this db
# persist_directory = '/Users/ryderwishart/genesis/databases/prose-contexts' # NOTE: Full prose contexts are in this db
# context_chroma = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name="prosaic_contexts_itemized")
# print(context_chroma.similarity_search_with_score('jesus (s) speaks (v) to peter (o)', search_type='similarity', k=1))

persist_directory = (
    "/Users/ryderwishart/genesis/databases/prose-contexts-shorter-itemized"
)
context_chroma = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="prosaic_contexts_shorter_itemized",
)
# print(
#     context_chroma.similarity_search_with_score(
#         "jesus (s) speaks (v) to peter (o)", search_type="similarity", k=1
#     )
# )


# ## Get Syntax brackets

# os.system("pip install lxml")

# Get the plain treedown representation for a token's sentence

from lxml import etree
import requests

# Get the plain treedown representation for a token's sentence

# example endpoint: "https://labs.clear.bible/symphony-dev/api/GNT/Nestle1904/lowfat?usfm-ref=JHN%2014:1" - JHN 14:1


def process_element(element, usfm_ref, indent=0, brackets=False):
    if brackets:
        indent = 0
    treedown_str = ""
    open_bracket = "[" if brackets else ""
    close_bracket = "] " if brackets else ""

    if element.get("class") == "cl":
        treedown_str += "\n" + open_bracket + ("  " * indent)

    if element.get("role"):
        role = element.attrib["role"]
        if role == "adv":
            role = "+"
        if not brackets:
            treedown_str += "\n"
        treedown_str += open_bracket + ("  " * indent) + role + ": "

    # # bold the matching token using usfm ref # NOTE: not applicable, since I think you have to use a USFM ref without the word on the endpoint
    # if element.tag == "w" and element.get("ref") == usfm_ref:
    #     treedown_str += "**" + element.text + "**"
    #     treedown_str += element.attrib.get("after", "") + close_bracket

    if element.tag == "w" and element.text:
        treedown_str += (
            element.attrib.get("gloss", "")
            + element.attrib.get("after", "")
            + f"({element.text})"
        )
        treedown_str += close_bracket

    for child in element:
        treedown_str += process_element(child, usfm_ref, indent + 1, brackets)

    return treedown_str


def get_treedown_by_ref(usfm_ref, brackets=True):
    usfm_passage = usfm_ref.split("!")[0]
    endpoint = (
        "https://labs.clear.bible/symphony-dev/api/GNT/Nestle1904/lowfat?usfm-ref="
        # "http://localhost:8984/symphony-dev/api/GNT/Nestle1904/lowfat?usfm-ref="
        + usfm_passage
    )

    # uri encode endpoint
    endpoint = requests.utils.requote_uri(endpoint)

    # print(endpoint)

    text_response = requests.get(endpoint).text
    xml = etree.fromstring(text_response.encode("utf-8"))

    treedown = process_element(xml, usfm_passage, brackets=brackets)
    return treedown


def get_syntax_for_query(query):
    # Get passage using bible passage lookup and grabbing first result metadata['usfm']
    most_relevant_passage_usfm_ref = bible_chroma.search(
        query, search_type="similarity", k=1
    )[0].metadata["usfm"]
    return get_treedown_by_ref(most_relevant_passage_usfm_ref)


# Define callback handlers
# In[33]:


from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# # from langchain.callbacks.streamlit import StreamlitCallbackHandler
# from langchain.llms.fake import FakeListLLM

# from flask_socketio import emit
# from langchain.callbacks.base import BaseCallbackHandler
# from typing import Any


# class StreamingSocketIOCallbackHandler(BaseCallbackHandler):
#     """Callback handler for streaming. Only works with LLMs that support streaming."""

#     def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
#         """Run on new LLM token. Only available when streaming is enabled."""
#         emit("agent_stdout", {"stdout": token})


# from flask_socketio import emit
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from typing import Any, Dict, List, Union

# """Callback Handler that logs to streamlit."""
from typing import Any, Dict, List, Optional, Union

import streamlit as st

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class StreamlitSidebarCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to streamlit."""

    def __init__(self) -> None:
        """Initialize the callback handler."""
        # self.tokens_area = st.sidebar.markdown("## Database Resources Consulted")
        tokens_area = st.expander("See Agent Reasoning")
        self.tokens_area = tokens_area
        self.tokens_stream = ""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        # st.write("Prompts after formatting:")
        # for prompt in prompts:
        #     st.write(prompt)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.write(self.tokens_stream)
        # self.sidebar.write(self.tokens_stream)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        # class_name = serialized["name"]
        # st.sidebar.write(f"Entering new {class_name} chain...")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        # st.sidebar.write("Finished chain.")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Print out the log in specified color."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        # st.sidebar requires two spaces before a newline to render it
        st.markdown(action.log.replace("\n", "  \n"))

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        # verses_with_refs = output
        # try:
        #     # assuming output is your string
        #     matches = re.findall(
        #         r"metadata.: \{.source.: .(.*?)., .usfm.: ..*?.\}, page_content=.(.*?).",
        #         output,
        #     )

        #     # matches will be a list of tuples where the first element of the tuple is the source and the second element is the page_content
        #     verses_with_refs = [match[0] + ": " + match[1] + "\n" for match in matches]
        #     verses_with_refs = "  \n- ".join(verses_with_refs)

        #     # st.sidebar.markdown(verses_with_refs)
        # except Exception as e:
        #     st.write("error on tool end", e)
        with st.expander("See Sources"):
            st.markdown(
                f"**<span style='color:blue'>Checked these sources</span>:**\n{output}",
                unsafe_allow_html=True,
            )
        # st.write(f"output: /{output}/")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on text."""
        # st.sidebar requires two spaces before a newline to render it
        st.sidebar.write(text.replace("\n", "  \n"))

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        # st.sidebar requires two spaces before a newline to render it
        st.sidebar.write(finish.log.replace("\n", "  \n"))


# class StreamlitDropdownCallbackHandler(StreamlitSidebarCallbackHandler):
#     """Callback handler to put agent reasoning into a dropdown."""


class StreamingSocketIOCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        print('emitting emit("activity", {"loading": True})')
        emit("activity", {"loading": True})

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        print('emitting emit("activity", {"stdout": token})')
        emit("activity", {"stdout": token})

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        print('emitting emit("activity", {"loading": False})')
        emit("activity", {"loading": False})

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        print('emitting emit("activity", {"error": str(error)})')
        emit("activity", {"error": str(error)})

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        print('emitting emit("activity", {"tool_start": serialized})')
        emit("activity", {"tool_start": serialized})

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        print('emitting emit("activity", {"action": action.text})')
        emit("activity", {"action": action.text})

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        print('emitting emit("activity", {"result": finish, "loading": False})')
        emit("activity", {"result": finish, "loading": False})


import sys
from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(self) -> None:
        """Initialize the callback handler."""
        self.tokens_area = st.sidebar.empty()
        # self.tokens_area = tokens_area
        # self.tokens_area = st.empty()
        self.tokens_stream = "Agent Reasoning\n\n"

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(token)
        sys.stdout.flush()

        if token == ".":
            token = ".\n\n"

        self.tokens_stream += token
        # Replace 'Input:', 'Final Answer:', 'Action:', with '\n' + string in tokens_stream
        # self.tokens_stream = (
        #     self.tokens_stream.replace("Input:", "\nInput:")
        #     .replace("Final Answer:", "\nFinal Answer:")
        #     .replace("Action:", "\nAction:")
        # )

        # self.tokens_area.markdown(self.tokens_stream)

        if token == ".":
            token = ".\n\n"

        # Replace 'Input:', 'Final Answer:', 'Action:', with '\n' + string in tokens_stream
        formatted_tokens_stream = (
            self.tokens_stream.replace(
                "Input:", "\n<span style='color:red'>Input</span>"
            )
            .replace("Final Answer:", "\n<span style='color:green'>Final Answer</span>")
            .replace("Action:", "\n<span style='color:blue'>Action</span>")
        )

        self.tokens_area.markdown(formatted_tokens_stream, unsafe_allow_html=True)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        # clear self.tokens_area

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        # tokens_area += f"OUTPUTS: {outputs}\n"

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        # st.write(f"Tool end output: {output}")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""


# from langchain.callbacks.streamlit import StreamlitCallbackHandler

# ## Define custom tools for an agent

# In[31]:


# Import things that are needed generically
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

# import term2md

# human_tool = load_tools(["human"])[0]

# In[32]:

# global_resources_consulted_log = []  # keep track of each time a resource is consulted


# def get_similar_resource(db, query, k=3):
#     get_result = lambda x: db.similarity_search(x, k=k)
#     # global_resources_consulted_log.append(result)
#     result = get_result(query)
#     print("DEBUG", result)
#     result_string = [
#         i.metadata["source"] + i.page_content + "\n" for i in result.split("Document")
#     ]
#     st.write(f"Found {len(result)} results")
#     st.write(f"Searching {db.name} for {query}...")
#     st.markdown(f"## Results\n{result_string}")

#     return result


# @tool
# def bible_verse_tool(input):
#     """Tool for finding similar verses in the Bible."""
#     return get_similar_resource(bible_chroma, input, k=2)


tools = [
    Tool(
        name="Bible Verse Reader Lookup",
        # Use the
        # func=lambda x: bible_chroma.search(x, search_type="similarity", k=2),
        # func=bible_verse_tool.run,
        func=lambda x: bible_chroma.search(x, search_type="similarity", k=3),
        description="useful for finding verses that are similar to the user's query, not suitable for complex queries",
        callbacks=[StreamlitSidebarCallbackHandler()],
    ),
    # Tool(
    #     name="Bible Words Lookup",
    #     func=macula_greek_agent.run, # Note: using the NT-only agent here
    #     description="useful for finding information about biblical words from the Macula TSV data, which includes glosses, lemmas, normalized forms, and more. This tool is not useful for grammar and syntax questions (about subjects, objects, verbs, etc.), but is useful for finding information about the words themselves",
    # ),
    Tool(
        name="Bible Verse Dataframe Tool",
        func=macula_greek_agent.run,  # Note: using the NT-only agent here
        description="useful for finding information about Bible verses in a dataframe in case counting, grouping, aggregating, or list building is required. This tool is not useful for grammar and syntax questions (about subjects, objects, verbs, etc.), but is useful for finding information about the words (English or Greek or Greek lemmas) themselves",
        callbacks=[StreamlitSidebarCallbackHandler()],
    ),
    Tool(
        name="Linguistic Data Lookup",
        func=lambda x: context_chroma.similarity_search(x, k=1),
        # func=lambda query: get_similar_resource(context_chroma, query, k=2),
        callbacks=[StreamlitSidebarCallbackHandler()],
        description="useful for finding linguistic data about the user's query. Use this if the user is asking a question that relates to the linguistic discourse, situational context, participants, semantic roles (source, process, goal, etc.), or who the speakers are in a passage",
    ),
    # Tool(
    #     name="Context for Most Relevant Passage", # NOTE: this tool isn't working quite right. Needs some work
    #     func=get_context_for_most_relevant_passage.run,
    #     description="useful for when you need to find relevant linguistic context for a Bible passage. Input should be 'situation for' and the original user query",
    # callbacks=[StreamlitSidebarCallbackHandler()],
    # ),
    Tool(
        name="Syntax Data Lookup",
        func=lambda x: get_syntax_for_query(x),
        description="useful for finding syntax data about the user's query. Use this if the user is asking a question that relates to a sentence's structure, such as 'who is the subject of this sentence?' or 'what are the circumstances of this verb?'",
        callbacks=[StreamlitSidebarCallbackHandler()],
    ),
    Tool(
        name="Theological Data Lookup",
        func=lambda x: theology_chroma.search(x, search_type="similarity", k=2),
        # func=lambda query: get_similar_resource(theology_chroma, query, k=2),
        callbacks=[StreamlitSidebarCallbackHandler()],
        description="useful for finding theological data about the user's query. Use this if the user is asking about theological concepts or value-oriented questions about 'why' the Bible says certain things. Always be sure to cite the source of the data",
    ),
    Tool(
        name="Encyclopedic Data Lookup",
        func=lambda x: encyclopedic_chroma.similarity_search(x, k=2),
        # func=lambda query: get_similar_resource(encyclopedic_chroma, query, k=2),
        callbacks=[StreamlitSidebarCallbackHandler()],
        description="useful for finding encyclopedic data about the user's query. Use this if the user is asking about historical, cultural, geographical, archaeological, or other types of information from secondary sources",
    ),
    Tool(
        name="Any Other Kind of Question Tool",
        func=lambda x: "Sorry, I don't know!",
        description="This tool is for vague, broad, ambiguous questions",
        callbacks=[StreamlitSidebarCallbackHandler()],
    ),
    # human_tool,
    # Tool(
    #     name="Get Human Input Tool",
    #     func=lambda x: input(x),
    #     description="This tool is for vague, broad, ambiguous questions that require human input for clarification",
    # ),
]

agent = initialize_agent(
    tools,
    # OpenAI(temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()]),
    OpenAI(
        temperature=0,
        streaming=True,
        # callbacks=[StreamlitSidebarCallbackHandler(), StreamingStdOutCallbackHandler()],
        callbacks=[StreamingStdOutCallbackHandler()],
    ),
    # OpenAI(
    #     temperature=0, streaming=True, callbacks=[StreamingSocketIOCallbackHandler()]
    # ),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    # reduce_k_below_max_tokens=True,
)


# # Flask UI - non streaming

# In[ ]:


# from flask import Flask, render_template, request
# from flask_socketio import SocketIO, emit

# app = Flask(__name__)
# socketio = SocketIO(app)


# def agent_run(inputs):
#     return agent.run(inputs)


# @app.route("/")
# def index():
#     return render_template("index.html")


# @socketio.on("run_agent")
# def handle_run_agent(message):
#     # Run your agent here
#     result = agent_run(message["data"])

#     # Emit the result to the client
#     emit("agent_result", {"result": result})


# if __name__ == "__main__":
#     socketio.run(app, port=5001)

# Flask UI - streaming

# from flask import Flask, render_template, request
# from flask_socketio import SocketIO, emit
# import sys
# from io import StringIO

# app = Flask(__name__)
# socketio = SocketIO(app)


# # Capture stdout
# class CapturedOutput:
#     def __enter__(self):
#         self._stdout = sys.stdout
#         sys.stdout = self._stringio = StringIO()
#         return self

#     def __exit__(self, type, value, traceback):
#         sys.stdout = self._stdout

#     def getvalue(self):
#         return self._stringio.getvalue()


# def agent_run(inputs):
#     with CapturedOutput() as output:
#         # Run the agent with the user's input
#         final_output = agent.run(inputs)
#     return final_output, output.getvalue()


# @app.route("/")
# def index():
#     return render_template("index.html")


# @socketio.on("run_agent")
# def handle_run_agent(message):
#     # Run your agent here
#     final_output, stdout = agent_run(message["data"])

#     # Stream the stdout to the client
#     for token in stdout.split():
#         emit("agent_stdout", {"stdout": token})

#     # Emit the final output to the client
#     emit("agent_result", {"result": final_output})


# if __name__ == "__main__":
#     socketio.run(app, port=5001)

## Flask streaming using socketio callback handler
# from flask import Flask, render_template, request
# from flask_socketio import SocketIO, emit
# import sys
# from io import StringIO
# import logging
# from time import sleep

# app = Flask(__name__)
# socketio = SocketIO(app)

# os.system(
#     "mkdir -p scripts && wget https://raw.githubusercontent.com/drudru/ansi_up/master/ansi_up.js -O scripts/ansi_up.js"
# )


# # Capture stdout
# class CapturedOutput:
#     def __enter__(self):
#         self._stdout = sys.stdout
#         sys.stdout = self._stringio = StringIO()
#         return self

#     def __exit__(self, type, value, traceback):
#         sys.stdout = self._stdout

#     def getvalue(self):
#         return self._stringio.getvalue()


# def agent_run(inputs):
#     with CapturedOutput() as output:
#         # Run the agent with the user's input
#         final_output = agent.run(inputs)
#     return final_output, output.getvalue()


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/log_stream")
# def stream_logs():
#     def generate():
#         with open("job.log") as f:
#             while True:
#                 yield f.read()
#                 sleep(1)

#     return app.response_class(generate(), mimetype="text/plain")


# @socketio.on("run_agent")
# def handle_run_agent(message):
#     # The callback handler will stream the output to the flask app.
#     agent_run(message["data"])


# if __name__ == "__main__":
#     logging.basicConfig(filename="job.log", level=logging.INFO)
#     socketio.run(app, port=5001, debug=True)

# Steamlit UI - Streaming version

# import streamlit as st
# import sys
# from io import StringIO
# from streamlit_pills import pills

# # Create a text input for user input
# user_input = st.text_input("Enter your message:")


# # Capture stdout
# class CapturedOutput:
#     def __enter__(self):
#         self._stdout = sys.stdout
#         sys.stdout = self._stringio = StringIO()
#         return self

#     def __exit__(self, type, value, traceback):
#         sys.stdout = self._stdout

#     def getvalue(self):
#         return self._stringio.getvalue()


# # Create a selection for streaming mode
# selected = pills("", ["NO Streaming", "Streaming"], ["🎈", "🌈"])

# if user_input:
#     if selected == "Streaming":
#         with CapturedOutput() as output:
#             # Run the agent with the user's input
#             final_output = agent.run(user_input)

#         # Display the final output in the main app
#         st.write(f"Chatbot: {final_output}")

#         # Stream the stdout in the sidebar
#         for token in output.getvalue().split():
#             st.sidebar.write(token)
#     else:
#         with CapturedOutput() as output:
#             # Run the agent with the user's input
#             final_output = agent.run(user_input)

#         # Display the final output in the main app
#         st.write(f"Chatbot: {final_output}")

#         # Display the stdout in the sidebar
#         st.sidebar.write(output.getvalue())

st.header("Genesis Demo")
st.image("/Users/ryderwishart/Downloads/DALL·E 2023-06-14 11.05.52.png", width=600)
from streamlit_chat import message
from datetime import datetime
import random

# Initialize the chat history in session_state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Create a container for the chat history
chat_container = st.expander("Question History")

# Create a container for the user input
input_container = st.container()


# Display the chat history
with chat_container:
    for sender, text in reversed(st.session_state["chat_history"]):
        is_user = sender == "user"
        message(text, is_user=is_user, avatar_style="icons")

# Create a text input for user input in the input container
with input_container:
    user_input = st.text_input("Ask a question:")

    if user_input:
        chat_container.empty()  # clear the chat history widget
        ai_response = st.empty()
        st.empty()  # clear the input field

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Add the user's input to the chat history
        st.session_state["chat_history"].append(("user", user_input))
        message(
            user_input,
            is_user=True,
            avatar_style="icons",
            key=f"user-{timestamp}_{str(random.randint(0, 1000))}",
        )

        # Run the agent with the user's input
        try:
            result = agent.run(user_input)
        except Exception as e:
            result = "Sorry, I don't know! I hit an error: " + str(e)

        # Add the agent's response to the chat history
        st.session_state["chat_history"].append(("agent", result))

        # Display the final output in the main app
        message(
            result,
            # avatar_style="icons",
            key=f"agent-{timestamp}_{str(random.randint(0, 1000))}",
        )
