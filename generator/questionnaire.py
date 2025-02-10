import os
import random
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

# logger.remove()
# Configure logger to write to a file
logger.add("logs/questionnaire_eval_logs.log", rotation="1 MB")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=random.uniform(0.1, 0.9),
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY"),
)

def questionnaire_generator(prompt: str, tools: list, category_description: str, quantity: int) -> dict:
    """
    Generate a set of questions and answers which can be used to test the chatbot.
    Args:
        prompt (str): The system prompt of the chatbot.
        tools (list): The tools associated with the chatbot.
        category_description (str): The category description of the questions and answers.
        quantity (int): The number of questions and answers to be generated.
    Returns:
        dict: A dictionary containing the questions and answers.
    """
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a questionnaire generator and a trainer for a chatbot. You are given a system prompt of a chatbot. You need to generate a set of user message, bot answers and corresponding tool calls from the bot which can be used to test the chatbot. If for any question, the tool_call should be not be called then you can keep that tool call as null. You will be also given a category description of what the generated QA pairs will be about. You only generate those types of QA pairs. For example if it is said to generate 5 QA pairs for the topics end_call, then your all 5 QA pairs will be regarding ending the call or bye bye messages. Remember it is very important your generated set of question and answers will be in bangla language. Remem \nFor example here is a system prompt below: \n## Example system prompt: ```You are a call center agent of a barbar shop. Your task is to ask the caller what service he want. Then schedule an appointment for the caller. User schedule_appointment tool to book the time. And then let the customer know about it and end the call. Tools: {{\"name\": \"schedule_appointment\": \"Description\": \"This tool is used to book the time for the caller.\"}} \nCatergory: Appointment scheuling.``` ## End of example. \nYou need to generate a set of questions and answers only for appointment booking. Be as much creative as you can. All questions will be appointment booking related but all will be highly different. Your answer will be in a valid json format. The json format is as follows: [{{\"question\": \"content of question 1\", \"answer\": \"content of answer 1\", \"tool_call\": null}}, {{\"question\": \"content of question 2\", \"answer\": content of answer 2, \"tool_call\": \"tool_name\"}}, {{\"question\": \"content of question 3\", \"answer\": \"content of answer 3\", \"tool_call\": null}}]. \nIt is not mandatory that the question field will contain a question everytime. It sometime can be a statement. For example: \"হ্যালো\" or \"আমি ভাল আছি\"or \"আজকে দুপুরে বৃষ্টি হয়েছিল এখানে। \". Remember these QA pairs are not for a conversation testing. These are for tesing the chatbot with single message. So no previous conversation history will be there in the chatbot memory. So do not generate any pair which will be dependent on one another. Also the generated user messages or questions and the assistant responses or answers should be as informative as possible for example instead of \"এটা কিনব\" generate \"আমি দুইটা কালো কালারের স্লিম টাইপ সামসাং ব্যান্ড গুলা কিনতে চাই, দাম কত পরবে?\"  \nNow generate total few numbers of questions answers and tool call sets that is mentioned by the user."
        ),
        (
            "human", "## A system prompt from where questions and answers need to be generated:\n {input}\n## End of system prompt\n## Tools associated with the system prompt:\n {tools}\n ##End of tools\n## Category description:\n {category}\n## Quantity of questions and answers to be generated: {quantity}\n ## Your response:\n"
        ),
    ]
    )

    chain = prompt | llm
    return chain.invoke(
        {
           "input": prompt,
           "tools": tools,
           "quantity": quantity,
           "category": category_description
        }
    ).content

def qa_pairs_evaluator(qa_pairs_draft: dict, prompt: str, ground_tools: list) -> dict:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an evaluator of bengali conversations between user and an assistant. You are given a conversation as well as a prompt of the assistant where there is a set of questions and answers and tool calls in a json structure where questions contains the user message and answers contains the assistant message and tool calles are always be there. You need to ensure that all the questions and answers are relevant and unique. The question will be asked by user and the answer will be given by a chatbot. You need to evaluate if the question and answer is correct or not based on the prompt and ground tools of the chatbot assistant. You will respond in a valid json structure where you will add the key-value pair number and a reason for removing that pair. The json structure is as follows: [{{\"key_value_pair\": 0, \"reason\": \"reason for removing that pair\"}}, {{\"key_value_pair\": 2, \"reason\": \"reason for removing that pair\"}}, {{\"key_value_pair\": 3, \"reason\": \"reason for removing that pair\"}}]. Some example reason can be: question is abnormal, answer is not relevant, user wanted to help assistant, etc. If there is no pair to remove then you can keep the key-value pair as null. For example [{{\"key_value_pair\": null, \"reason\": null}}].\n\nCommon mistakes you have find are: - Sometime user acts like the assistant, - Sometime there is needed a tool call but the tool is not called, - Sometime user wants irrelevent query and assisatnt also wants to help in that segment. Remember if any question answer pair seems conflicting with the main task of the chatbot then remove it. Like if the bot is a weather assistant and user asks for hair cut service then it is conflicting. But do not remove greeting or bye bye messages, these are not conflicting."
            ),
            (
                "human", "## A set of questions answers and tool calls are:\n {input}\n## End of set of questions and answers\n## The prompt and ground tools are:\n- **Prompt**: {prompt}\n- **Ground Tools**: {ground_tools}\n## End of prompt and ground tools\n## Your response:\n"
            ),
        ]
    )
    chain = prompt | llm
    return chain.invoke(
        {
           "input": qa_pairs_draft,
           "prompt": prompt,
           "ground_tools": ground_tools
        }
    ).content


def generate_qa_pairs(prompt: str, tools: list, category_description: str, quantity: int) -> dict:
    """
    Generate and filter QA pairs based on evaluator feedback.
    Args:
        prompt (str): The system prompt of the chatbot.
        tools (list): The tools associated with the chatbot.
        category_description (str): The category description of the questions and answers.
        quantity (int): The number of QA pairs to be generated.
    Returns:
        dict: Filtered QA pairs in JSON format.
    """
    def clean_json_response(response):
        return json.loads(response.replace("```json", "").replace("```", ""))

    qa_pairs_draft = clean_json_response(questionnaire_generator(prompt, tools, category_description, quantity))
    eval_qa_pairs = clean_json_response(qa_pairs_evaluator(qa_pairs_draft, prompt, tools))
    logger.info(f"Evaluation of QA pairs: {eval_qa_pairs}")
    if eval_qa_pairs is None:
        return qa_pairs_draft

    key_value_pairs = {pair['key_value_pair'] for pair in eval_qa_pairs if pair['key_value_pair'] is not None}
    qa_pairs = [pair for i, pair in enumerate(qa_pairs_draft) if i not in key_value_pairs]
    return qa_pairs
