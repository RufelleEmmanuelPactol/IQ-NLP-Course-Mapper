import csv
import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel
import json
from mysql.connector import Connect
from os import system
from chromadb import PersistentClient
import pandas as pd

load_dotenv()
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

db = Connect(host=os.getenv('HOST'), port=3306, user=os.getenv('DBUSER'), password=os.getenv('PASSWORD'),
             database=os.getenv('DB'))
os.environ['TERM'] = 'xterm'
chroma = PersistentClient(path='embeddings')
course_sections = chroma.get_or_create_collection(name='course-sections', metadata={"hnsw:space": "cosine"})
solutions = chroma.get_or_create_collection(name='solutions', metadata={"hnsw:space": "cosine"})


def generate_attributions():
    query = """
    SELECT sum(vote) as upvote, q.id, concat('interviewquery.com/questions/', slug) as 'link', q.title FROM content_upvotes cu
    left join questions q on q.id = cu.content_id
    where content_type = 'question'
    group by q.id order by upvote desc limit 30
    """
    top = []
    with db.cursor() as cursor:
        cursor.execute(query)
        all = cursor.fetchall()
        for entry in all:
            top.append(entry[1])
    return top

def get_solution(question_id):
    question = solutions.get(ids=[str(question_id)])
    if len(question['ids']) == 0:
        return None
    value = {
        'id': question['ids'][0],
        'link' : question['metadatas'][0]['link'],
        'text': question['metadatas'][0]['text']
    }
    return value

cache_store = {}
def main():
    solutions = {
        'solution_link': [],
        'related_course_section' : [],
        'difference': []
    }
    questions = generate_attributions()
    for question in questions:
        sol = get_solution(question)
        cache_store[sol['id']] = sol



if __name__ == '__main__':
    main()
