import csv
import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel
import json
from mysql.connector import Connect
from os import system
from chromadb import PersistentClient

load_dotenv()
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

db = Connect(host=os.getenv('HOST'), port=3306, user=os.getenv('DBUSER'), password=os.getenv('PASSWORD'),
             database=os.getenv('DB'))
os.environ['TERM'] = 'xterm'
chroma = PersistentClient(path='embeddings')
course_sections = chroma.get_or_create_collection(name='course-sections', metadata={"hnsw:space": "cosine"})
solutions = chroma.get_or_create_collection(name='solutions', metadata={"hnsw:space": "cosine"})


def string_to_embeddings(text):
    """
    Convert a string to SciBERT embeddings.

    Parameters:
    text (str): The input string.

    Returns:
    torch.Tensor: The embeddings of the input text.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :]

    return embeddings.numpy().tolist()


def cache_ifn(map, collection=course_sections):
    if len(collection.get(ids=[map['id']])['ids']) == 0:
        collection.add(
            documents=[map['text']],
            embeddings=string_to_embeddings(map['text'])[0],
            metadatas=[map],
            ids=[str(map['id'])]
        )

    else:
        pass


def load_cms_modules() -> list:
    query = """
    SELECT s.id, s.markdown, s.slug as 'section_slug', c.slug as 'course_slug', lp.slug as 'learning_path_slug', csm.course_id FROM sections s
    INNER JOIN course_section_mapping csm on s.id = csm.section_id
    INNER JOIN courses c on c.id = csm.course_id
    INNER JOIN learning_path_courses_mapping lpcm on lpcm.course_id = c.id
    INNER JOIN learning_paths lp on lpcm.learning_path_id = lp.id
    WHERE s.question_id = 0 and s.is_locked=0 and s.is_published=1 and c.slug <> 'dummy-lesson';
    """
    with db.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchall()
        length = len(result)
        ctr = 0
        for row in result:
            link_gen = 'interviewquery.com/learning-paths/' + row[4] + '/' + row[3] + '/' + row[2]
            details = {
                'id': str(row[0]),
                'text': row[1],
                'link': link_gen
            }
            cache_ifn(details)
            ctr += 1
            clear()
            print(f"Loaded ", "%.2f%%" % (100.00 * (ctr / length)))


def store_and_generate_solutions():
    with db.cursor() as cursor:
        query = """SELECT concat('interviewquery.com/questions/', slug) as 'link', answer_markdown, id FROM questions;"""
        cursor.execute(query)
        result = cursor.fetchall()
        length = len(result)
        ctr = 0
        for row in result:
            cache_ifn({
                'id': str(row[2]),
                'link': str(row[0]),
                'text': str(row[1])
                }, collection=solutions)
            ctr += 1
            print(f"Loaded ", "%.2f%%" % (100.00 * (ctr / length)))

def initialize_cache_store():
    load_cms_modules()
    store_and_generate_solutions()


def clear():
    system('clear')


def main():
    data = course_sections.get()
    initialize_cache_store()




if __name__ == "__main__":
    main()
