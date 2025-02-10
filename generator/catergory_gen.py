from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
from loguru import logger
import os

# logger.remove()
# Configure logger to write to a file
logger.add("logs/category_gen_logs.log", rotation="1 MB")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY"),
)


def clean_json(json_string):
    return json.loads(json_string.replace("```json", "").replace("```", ""))

def category_generator(prompt, tools, quantity):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are assigned to a task of generating category for generating each part of a conversation between user and a chatbot from a given prompt. For example ```if the prompt is about a call center agent of a supershop then the topics you can refer is 1. Greetings from user, 2. user asking about various products 3. User asking about a particular product like a dinner set or something else but in depth. 4. User wants to verify his address. Sometime he forgets his address. Or etc.```\nRemember you reponse in english and in a valid json structure. The json structure will contain 2 items. 1. topic_name and 2. topic_description. For example [{{\"topic_name\": \"Greetings\", \"topic_description\": \"This part is about all categories related to greetings from the user. User will send greeting message and assistant will respond regarding that.\"}}, {{\"topic_name\": \"send_email\", \"topic_description\": \"This qa pairs are about user asking to send an email to a particular email address.\"}}]. \nThe topic name should be in one word and topic description should be a short description of how user will interact with the chatbot. Remember the topics you will generate will of course be relevent to the prompt provided by the human below. You have to understand the tools from the given prompt by the human. Greetings and ending the call are also important topics. Remember be creative as much as possible, do not generate such topics \nGenerate total {quantity} topics for the given prompt."
            ),
            (
                "human", "## This is the prompt for generating categories:\n {input}\n## End of prompt\n## Tools provided to the chatbot are {tools}\n## End of Tools\n\n## Your response:\n"
            ),
        ]
    )
    chain = prompt | llm
    response = clean_json(chain.invoke(
        {
           "input": prompt,
           "tools": tools,
           "quantity": quantity,
        }
    ).content)
    logger.info(f"Generated categories: {response}")
    return response