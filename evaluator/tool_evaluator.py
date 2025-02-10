import json

# def evaluate_tools(tool_response: list, ground_tool_name: str) -> dict:
#     """
#     Evaluate the tools.
#     """
#     if tool_response == [] and ground_tool_name == None:
#         return True, []
    
#     elif tool_response != []:
#         tool_names = []
#         for tool in tool_response:
#             tool_names.append(tool["tool_calls"][0]["function"]["name"])
#         if ground_tool_name in tool_names:
#             return True, tool_names
#         else:
#             return False, tool_names
#     return False, tool_response

def evaluate_tools(tool_response: list, ground_tool_name: str) -> dict:
    """
    Evaluate the tools.
    """
    if tool_response == [] and ground_tool_name == None:
        return True, []
    elif tool_response != []:
        tool_name = tool_response[0]["tool_calls"][0]["function"]["name"]    
        if ground_tool_name == tool_name:
            return True, tool_name
        elif "to_" in tool_name:
            if ground_tool_name == tool_name.replace("to_", ""):
                return True, tool_name
            else:
                return False, tool_name
        else:
            return False, tool_name
    return False, tool_response
