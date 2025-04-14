"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
"""
This file implements prompt template for llama based models. 
Modify the prompt template based on the model you select. 
"""

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


system_prompt = """You are a helpful assistant tasked with answering user questions based on the provided context. Always carefully read and analyze the given context before responding. If a question cannot be answered using the provided context, inform the user politely. Strive to provide detailed and thorough answers to all questions"""


def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=False):
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

        memory = ConversationBufferMemory(input_key="question", memory_key="history")

        print(f"Here is the prompt used: {prompt}")

        return (
        prompt,
         memory,
         )
