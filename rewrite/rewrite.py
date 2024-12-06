import os.path
import sys
sys.path.append('..')
from tool.util import *
from constant import *


def icl_rewrite(prev_turns,question):
    '''
    :param prev_turns: format like [q_0,q_1,q_2,.....q_t-1]
    :param question: q_t
    :return: q_rewrite
    '''
    prompt_name = "icl_rewrite_v1.txt"
    with open(prompt_name, "r", encoding="utf-8") as f:
        final_prompt = f.read().format(
            prev_turns=prev_turns,
            question=question,
        )
    rewrite_question = talk_llm([final_prompt],llm_name="gpt-4o-mini")
    return rewrite_question