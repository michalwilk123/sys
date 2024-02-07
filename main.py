import csv
import json
from beir_utils import evaluate

import tqdm
import logging

from implementation import _WEBM25Implementation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.propagate = False
file_handler = logging.FileHandler("logfile1.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


class WEBM25Search(_WEBM25Implementation):
    ...


def run_test(searcher, queries, qrels, name="unknown"):
    runs02plus = {}
    runs02 = {}
    runs03 = {}
    runs04 = {}
    runs06 = {}
    runs08 = {}
    runs10 = {}

    for query_id, scores in tqdm.tqdm(qrels.items(), desc=f"Running search: on dataset: {name}"):
        query = queries[query_id]
        runs02plus[query_id] = searcher.search(
            query, topk=25, threshhold=0.2, return_dict=True, num_of_max_sim_words=15,
        )
        runs02[query_id] = searcher.search(
            query, topk=25, threshhold=0.2, return_dict=True, 
        )
        runs03[query_id] = searcher.search(
            query, topk=25, threshhold=0.3, return_dict=True
        )
        runs04[query_id] = searcher.search(
            query, topk=25, threshhold=0.4, return_dict=True
        )
        runs06[query_id] = searcher.search(
            query, topk=25, threshhold=0.6, return_dict=True
        )
        runs08[query_id] = searcher.search(
            query, topk=25, threshhold=0.8, return_dict=True
        )
        runs10[query_id] = searcher.search(query, topk=25, threshhold=0.99, return_dict=True)

        logger.info("t:0.2, q:{} Best documents scores: {}".format(query_id, runs02[query_id]))
        logger.info("t:0.2, q:{} Correct docs {}".format(query_id, runs02[query_id].keys() & scores.keys()))

        logger.info("t:0.3, q:{} Best documents scores: {}".format(query_id, runs03[query_id]))
        logger.info("t:0.3, q:{} Correct docs {}".format(query_id, runs03[query_id].keys() & scores.keys()))

        logger.info("t:0.4, q:{} Best documents scores: {}".format(query_id, runs04[query_id]))
        logger.info("t:0.4, q:{} Correct docs {}".format(query_id, runs04[query_id].keys() & scores.keys()))

        logger.info("t:0.6, q:{} Best documents scores: {}".format(query_id, runs06[query_id]))
        logger.info("t:0.6, q:{} Correct docs {}".format(query_id, runs06[query_id].keys() & scores.keys()))

        logger.info("t:0.8, q:{} Best documents scores: {}".format(query_id, runs08[query_id]))
        logger.info("t:0.8, q:{} Correct docs {}".format(query_id, runs08[query_id].keys() & scores.keys()))

        logger.info("t:1.0(bm25), q:{} Best documents scores: {}".format(query_id, runs10[query_id]))
        logger.info("t:1.0, q:{} Correct docs {}".format(query_id, runs10[query_id].keys() & scores.keys()))

        if len(corr1 := (runs02[query_id].keys() & scores.keys())) + 2 < len(
            corr2 := (runs10[query_id].keys() & scores.keys())
        ):
            logger.info(
                "CORRECT: query: {}, scores: {} total02: {} totalbm: {}".format(
                    queries[query_id], scores, len(corr1), len(corr2)
                )
            )
            logger.info("threshold 0.2: {} bm25: {}".format(corr1, corr2))
            logger.info("==============================")

    return runs02plus, runs02, runs03, runs04, runs06, runs08, runs10


corpus = {
    "1": {"text": "The young quick brown fox jumps over the lazy dogs", "title": "Dogs"},
    "0": {
        "text": "The young quick white fox jumps over the lazy dog",
        "title": "Dogs title and stuff",
    },
    "2": {"text": "A stitch in time saves nine", "title": "something"},
    "3": {"text": "Actions speak louder than words", "title": "barking"},
    "4": {"text": "All's well that ends well", "title": "cat"},
    "5": {"text": "Barking dogs seldom bite", "title": ""},
    "6": {"text": "Curiosity killed the cat", "title": "curiosity cat"},
    "7": {
        "text": "Don't count your quick quick chickens before they hatch",
        "title": "quick chicken",
    },
    "8": {
        "text": "Early to bed and early to rise makes a man healthy, wealthy, and wise",
        "title": "wealth",
    },
    "9": {"text": "Fools rush in where angels fear to tread", "title": ""},
    "10": {"text": "Great minds think alike", "title": ""},
    "11": {
        "text": "Alterations of the architecture of cerebral white matter in the developing human brain can affect beans",
        "title": "",
    },
}

DATASET = "nfcorpus"
CORPUS_FNAME = DATASET + "/corpus.jsonl"
QUERIES_FNAME = DATASET + "/queries.jsonl"
QRELS_FNAME = DATASET + "/qrels/test.tsv"

with open(CORPUS_FNAME) as f:
    corpus = {}

    for line in f.readlines():
        doc = json.loads(line)
        key, title, text = doc["_id"], doc.get("title", ""), doc.get("text", "")
        corpus[key] = {
            "title": title,
            "text": text,
        }

with open(QUERIES_FNAME) as f:
    queries = {}

    for line in f.readlines():
        doc = json.loads(line)
        key, text = doc["_id"], doc["text"]
        queries[key] = text

reader = csv.reader(open(QRELS_FNAME, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
qrels = {}
next(reader)

for id, row in enumerate(reader):
    query_id, corpus_id, score = row[0], row[1], int(row[2])

    if query_id not in qrels:
        qrels[query_id] = {corpus_id: score}
    else:
        qrels[query_id][corpus_id] = score

RUN_TEST = 1

searcher = WEBM25Search(tie_breaker=0.5, log_level=logging.INFO if RUN_TEST else logging.DEBUG)
searcher.add_documents(corpus)


if RUN_TEST:
    # with cProfile.Profile() as pr:
    #     runs02, runs03, runs04, runs06, runs08, runs10 = run_test(
    #         searcher, queries, qrels, name=CORPUS_FNAME
    #     )
    #     pr.print_stats(sort="cumulative")
    runs02plus, runs02, runs03, runs04, runs06, runs08, runs10 = run_test(
        searcher, queries, qrels, name=CORPUS_FNAME
    )

    ks = [1, 5, 10, 25]
    metrics02p = evaluate(qrels, runs02plus, ks)
    metrics02 = evaluate(qrels, runs02, ks)
    metrics03 = evaluate(qrels, runs03, ks)
    metrics04 = evaluate(qrels, runs04, ks)
    metrics06 = evaluate(qrels, runs06, ks)
    metrics08 = evaluate(qrels, runs08, ks)
    metrics10 = evaluate(qrels, runs10, ks)

    print("THRESH: 0.2 custom: wieksze topk")
    print(metrics02p)

    print("THRESH: 0.2")
    print(metrics02)

    print("\n===========\nTHRESHOLD: 0.3")
    print(metrics03)

    print("\n===========\nTHRESHOLD: 0.4")
    print(metrics04)

    print("\n===========\nTHRESHOLD: 0.6")
    print(metrics06)

    print("\n===========\nTHRESHOLD: 0.8")
    print(metrics08)

    print("\n===========\nTHRESHOLD: 1 (bm25)")
    print(metrics10)

else:
    topk=15
    print(searcher.search("prevent", 40, threshhold=0.20, num_of_max_sim_words=topk))

    while True:
        try:
            print()
            query = input("Search: ")

            documents_ids99 = searcher.search(query, 40, threshhold=0.99, num_of_max_sim_words=15)
            documents_ids02 = searcher.search(query, 40, threshhold=0.2, num_of_max_sim_words=15)
            print("DOCUMENTS02: ", documents_ids02)
            print()
            print("DOCUMENTS99: ", documents_ids99)
            print()
            # print(
            #     f"THRESH {tr}",
            #     {doc_id: searcher.documents[doc_id][:200] for doc_id in documents_ids},
            # )
            # print()
        except KeyboardInterrupt:
            print("\nBye :)")
            break
