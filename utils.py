import os
import re
from dotenv import load_dotenv
load_dotenv()
from pprint import pprint
from openai import OpenAI

openai_client = OpenAI()

def get_query_variation(query, no_of_variation):
    SYSTEM_PROMPT = """
        You are an helpful AI Assistant who gives me a variation of queries which user asked to you. You will have a query and number of variation for
        how many optimize queries need to create.

        Query: {{query}},
        No of variation: {{no_of_variation}}

        For example:
        I will give you query like 'what is sniffing' and no of variation will be '3'. So you need to respond with three query which will be enhanced version of it and more clear.
    """
    # print(SYSTEM_PROMPT)

    result = openai_client.chat.completions.create(
        model=os.getenv('OPENAI_API_GPT_MODEL'),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{query}, create {no_of_variation} queries."}
        ]
    )
    content = list(map(lambda x: re.sub(r'^\d+\.\s*', '', x.strip()), result.choices[0].message.content.split('\n')))
    return content


def get_relevant_chunks(retriever, optimize_querys: list):
    data = {}
    for i, query in enumerate(optimize_querys):
        data[i] = {
            "query": query,
            "chunks": retriever.similarity_search(query=query)
        }
    return data


from langchain_core.documents import Document

from langchain_core.documents import Document

def get_unique_chunks(results_dict: dict[int, dict[str, list[Document]]]) -> list[Document]:
    seen_ids = set()
    seen_contents = set()
    unique_chunks = []

    for result in results_dict.values():
        for chunk in result['chunks']:
            chunk_id = chunk.metadata.get('_id')
            content = chunk.page_content.strip()

            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)
            elif content not in seen_contents:
                seen_contents.add(content)
                unique_chunks.append(chunk)

    return unique_chunks

if __name__ == "__main__":
    print("# utils file")
    # get_query_variation('')
