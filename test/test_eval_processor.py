import os
import json
import uuid
import sys
import asyncio

sys.path.append("../")

from eval_server.eval_processor import EvalProcessor

eval_processor = EvalProcessor()


async def main():

    with open(
        "../graph_data/bright_dental_appointment.json", "r"
    ) as f:
        graph_data = json.load(f)

    user_messages = [
        "একটা ডেন্টাল বেডে আমার কার্যক্রম কি হবে?",
        "আমি আমার ঠিকানা যাচাই করতে চাই।"
        "আমি কি ভাবে আমার ঠিকানা যাচাই করতে পারি?",
        "কেমন আছেন?",
        "আমার ঠিকানা মনে নেই, কি করব?",
        "আমি একটি অ্যাপয়েন্টমেন্ট বুক করতে চাই।",
        "আমি গত সপ্তাহে আপনার সেবার জন্য একটি পর্যালোচনা দিতে চাই।",
        "আমি আগামীকাল সকাল ১০টায় আসতে চাই।",
        "আপনার সেবার মান নিয়ে আমি খুব সন্তুষ্ট।",
        "আমি আপনার সেবার সম্পর্কে কিছু প্রশ্ন করতে চাই।"
    ]

    results = []
    session_id = str(uuid.uuid4())
    for user_message in user_messages:
        result = await eval_processor.process_events(
            user_message, session_id, graph_data
        )
        results.append(result)

    with open("../script/logs/eval_processor_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())