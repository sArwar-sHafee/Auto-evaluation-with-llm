import os
from openai import OpenAI
from user_prompt import human_caller_prompt


client = OpenAI()


def get_user_response(messages, init_user_message="হ্যালো", model_name="gpt-4o"):
    if len(messages) == 2:
        return init_user_message

    completion = client.chat.completions.create(model=model_name, messages=messages)
    auto_user_response = completion.choices[0].message.content

    return auto_user_response
