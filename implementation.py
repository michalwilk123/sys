import base64
import bisect
import hashlib
import logging
import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import TypedDict
from itertools import chain
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

import faiss
import numpy as np
import spacy
import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
file_handler = logging.FileHandler("logfile1.log")
logger.addHandler(file_handler)


class Document(TypedDict):
    title: str
    text: str


class Corpora(TypedDict):
    id: Document


@lru_cache(maxsize=10_000)
def tokenize_text(nlp_model, text):
    doc = nlp_model(text)
    lemms = []

    for token in doc:
        if token.is_stop or not token.is_alpha:
            continue

        lem = token.lemma_.lower()
        lemms.append(lem)

    return lemms


def custom_tokenizer(nlp):
    inf = list(nlp.Defaults.infixes)               # Default infixes
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
    inf = tuple(inf)                               # Convert inf to tuple
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x] # Remove - between letters rule
    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

def _expand_query(tokens: list[str], query_dictionary=None, max_ngram=2, condition_fn=None):
    query_dictionary = query_dictionary or {}
    condition_fn = condition_fn or (lambda token: token in query_dictionary)
    expanded = []
    tokens = tokens.copy()
    current_ngram_size = max_ngram

    while tokens:
        current_ngram_size = min(len(tokens), current_ngram_size)
        ngram = tokens[:current_ngram_size]
        generated_ngram = " ".join(ngram)

        if current_ngram_size == 1 or condition_fn(generated_ngram):
            tokens = tokens[current_ngram_size:]
            expanded.extend(query_dictionary.get(generated_ngram, [generated_ngram]))
            current_ngram_size = max_ngram
        else:
            current_ngram_size -= 1

    return expanded


class _WmdVectorLookupDictionary:
    class _WmdDocumentSortedList(list):
        def add(self, doc_id, num_of_occurrences):
            # only works is python 3.10+
            bisect.insort_left(self, (doc_id, num_of_occurrences), key=lambda x: -x[1])

    def __init__(self):
        self.hash_lookup = {}
        self.name_lookup = {}
        self.hash_name_lookup = {}

    def add(self, vector_hash, name):
        documents_object = self._WmdDocumentSortedList()

        self.hash_lookup[vector_hash] = documents_object
        self.name_lookup[name] = documents_object
        self.hash_name_lookup[vector_hash] = name

    def __contains__(self, word):
        return word in self.name_lookup

    def by_hash(self, vector_hash):
        return self.hash_lookup[vector_hash]

    def by_name(self, name):
        return self.name_lookup[name]

    def get_name_from_vector(self, vec):
        return self.hash_name_lookup[vec]


class _WEBM25Implementation:
    def __init__(self, tie_breaker=0.5, fields=["text", "title"], log_level=logging.INFO) -> None:
        self.vector_lookup: dict[str, _WmdVectorLookupDictionary] = {
            field: _WmdVectorLookupDictionary() for field in fields
        }
        self.dim = 300
        self.faiss_index: dict[str, faiss.IndexFlatIP] = {
            field: faiss.IndexFlatIP(self.dim) for field in fields
        }
        self.fields = fields

        self.documents: Corpora = {}

        self.nlp = spacy.load("en_core_web_sm")
        # self.nlp.tokenizer = custom_tokenizer(self.nlp)
        self.floret_model = spacy.load("en_vectors_floret_lg")
        # self.floret_model = spacy.load("en_vectors_floret_md")
        self.query_inverse_df = {}
        self.query2vec = {}

        self.max_ngrams = 2
        self.topk = 10
        self.number_of_docs = 0
        self.tie_breaker = tie_breaker

        # okapi params
        self.okapik = 1.2
        self.okapib = 0.75
        self.maxokapi = 2
        self.okapiavgdl = {field: 0 for field in self.fields}
        self.okapidelta = 0  # bm25+ extension

        self._idf_cache = {}
        self._simil_cache = {}

        logger.setLevel(log_level)
        file_handler.setLevel(log_level)

    def _tokenize_text(self, text: str):
        return tokenize_text(self.nlp, text)

    def _create_word_vector_representation(self, word, normalize=True):
        vec = self.floret_model.vocab[word].vector

        if normalize:
            vector_norm = np.linalg.norm(vec)
            return vec / vector_norm

        return vec

    def _create_str_hash(self, text):
        h = hashlib.blake2b(text.encode(), digest_size=24)
        return base64.b64encode(h.digest()).decode("utf-8")

    def _create_arr_hash(self, arr):
        h = hashlib.blake2b(arr.tobytes(), digest_size=32)
        for dim in arr.shape:
            h.update(dim.to_bytes(4, byteorder="big"))
        return h.digest()

    def _populate_lookup(self, docs: Corpora):
        vectors_to_add = {field: [] for field in self.fields}
        prev_number_of_documents = len(self.documents)
        current_token_sum = {field: 0 for field in self.fields}

        assert all(field in next(iter(docs.values())) for field in self.fields), (
            "Document does not contain all necessary fields: %s" % self.fields
        )

        for doc_id, document in tqdm.tqdm(docs.items(), f"Indexing new documents: {len(docs)}"):
            assert doc_id not in self.documents, "Document already present in dataset! {}".format(
                doc_id
            )

            self.documents[doc_id] = document.copy()

            tokens = {field: self._tokenize_text(document.get(field, "")) for field in self.fields}
            token_counters = {field: Counter(tokens[field]) for field in self.fields}

            for field, token_counter in token_counters.items():
                current_token_sum[field] += len(tokens[field])
                self.documents[doc_id][f"_{field}_num_of_toks"] = len(tokens[field])

                for word, occurence in token_counter.items():
                    if word not in self.vector_lookup[field]:
                        vector = self._create_word_vector_representation(word)
                        word_hash = self._create_arr_hash(vector)
                        self.vector_lookup[field].add(word_hash, word)
                        vectors_to_add[field].append(vector)

                    self.vector_lookup[field].by_name(word).add(doc_id, occurence)


        for field in self.fields:
            self.okapiavgdl[field] = (
                (self.okapiavgdl[field] * prev_number_of_documents) + current_token_sum[field]
            ) / len(self.documents)

        logger.info(
            "Average length of document: %s"
            % "|".join("%s: %s" % (f, round(self.okapiavgdl[f])) for f in self.fields)
        )
        logger.info(
            "Number of new words to add: %s"
            % [(field, len(vecs)) for field, vecs in vectors_to_add.items()]
        )

        return vectors_to_add

    def expand_query(self, tokens):
        return _expand_query(tokens, self.query2vec, self.max_ngrams)

    def get_similiar_words(self, field, term, topk, threshhold):
        floret_vector = self._create_word_vector_representation(term).reshape(1, self.dim)
        _lims, distances, indices = self.faiss_index[field].range_search(floret_vector, threshhold)
        idxs = np.argsort(distances)[: -topk - 1 : -1]
        distances, indices = distances[idxs], indices[idxs]

        vectors = self.faiss_index[field].reconstruct_batch(indices[:topk])

        similiar_words = {}

        for vec, distance in zip(vectors, distances):
            vector_hash = self._create_arr_hash(vec)
            similiar_words[self.vector_lookup[field].get_name_from_vector(vector_hash)] = distance

        return similiar_words

    def calculate_idf(self, field, word_dict):
        idf_dict = {}

        most_similar_word = max(word_dict.items(), key=lambda x: x[1])[0]

        docs_with_phrase = set()

        docs_with_phrase.update(item[0] for item in 
                                self.vector_lookup[field].by_name(most_similar_word))

        max_num_of_docs = len(docs_with_phrase)

        for similiar in word_dict:
            docs_with_phrase = set()

            docs_with_phrase.update(item[0] for item in self.vector_lookup[field].by_name(most_similar_word))

            num_of_docs = len(docs_with_phrase)
            num_of_docs = max(num_of_docs, max_num_of_docs)

            idf = math.log(((self.number_of_docs - num_of_docs + 0.5) / (num_of_docs + 0.5)) + 1)
            idf_dict[similiar] = idf

        return idf_dict

    def get_similiar_words_for_words(self, field, query_terms, topk, threshhold):
        expanded_query = {}

        for term in query_terms:
            expanded_query[term] = self.get_similiar_words(field, term, topk, threshhold)

        return expanded_query

    def add_documents(self, docs):
        vectors_dict = self._populate_lookup(docs)

        vectors_dict = {
            field: np.stack(vectors) if vectors else np.empty(0)
            for field, vectors in vectors_dict.items()
        }

        for field in self.fields:
            if vectors_dict[field].any():
                self.faiss_index[field].add(vectors_dict[field])

        self.number_of_docs += len(self.documents)

    def calculate_term_importance(self, field, name, term_dict):
        """
        For tf-idf we calculate idf part here
        """
        if not term_dict:
            return {}

        idf_dict = self.calculate_idf(field, term_dict)
        logger.debug(f"IDF {name}: {idf_dict}")

        importances = {}
        qidf = self.query_inverse_df.get(name, math.inf) # soon :)

        for similiar_name in term_dict:
            importances[similiar_name] = min(
                idf_dict[similiar_name], qidf
            )

        return importances

    def measure_okapi(self, field, occurances, doc_id):
        okapik = self.okapik
        okapi_score = (occurances * (okapik + 1)) / (
            occurances
            + okapik
            * (
                1
                - self.okapib
                + self.okapib * (self.documents[doc_id][f"_{field}_num_of_toks"] / self.okapiavgdl[field])
            )
        ) + self.okapidelta
        return okapi_score

    def fetch_documents(self, term_dict, idf_dict, topk):
        columns = [("DocId", "U40"), ("Score", float)]
        column_field_names = {}

        for field in self.fields:
            column_field_names[field] = ["F%s_%s" % (field, term) for term in term_dict[field]]
            columns += [(colname, float) for colname in column_field_names[field]]

        candidates = {}
        docs_ids = set()

        for field in self.fields:
            assert len(term_dict[field]) == len(idf_dict[field])

            for colname, similar_words in term_dict[field].items():
                term_name = "F%s_%s" % (field, colname)
                candidates[term_name] = {}

                for similar_word, distance in similar_words.items():
                    topk_docs = self.vector_lookup[field].by_name(similar_word)[:1000]

                    for doc_id, occurences in topk_docs:
                        okapi_score = self.measure_okapi(field, occurences, doc_id)
                        candidates[term_name][doc_id] = max(
                            candidates[term_name].get(doc_id, 0),
                            # candidates[term_name].get(doc_id, 0) + 
                            # okapi_score * idf_dict[field][colname][similar_word] * coeff * d
                            okapi_score * idf_dict[field][colname][similar_word] * (distance**2)
                        )
                        

                docs_ids |= candidates[term_name].keys()

        docs_ids = list(docs_ids)
        array = np.empty(len(docs_ids), dtype=columns)
        array["DocId"] = docs_ids

        for colname, docs in candidates.items():
            freq_array = []

            for doc_id in docs_ids:
                freq_array.append(docs.get(doc_id, 0))

            array[colname] = freq_array

        subsum = np.zeros((len(self.fields), array.shape[0]))

        for idx, field in enumerate(self.fields):
            for colname in column_field_names[field]:
                subsum[idx] += array[colname]

        maxim_fields = subsum.max(axis=0) * (1 - self.tie_breaker)
        array["Score"] = subsum.sum(axis=0) * self.tie_breaker + maxim_fields

        best_documents_indices = (-array["Score"]).argsort()[:topk]
        best_docs_dict = {array["DocId"][i]: array["Score"][i] for i in best_documents_indices}

        logger.debug("Best documents scores: {}".format(best_docs_dict))

        if logger.getEffectiveLevel() == logging.DEBUG:
            for i, indice in enumerate(best_documents_indices):
                row_string = f"{i+1}) doc: {array[indice]['DocId']} "

                for colname in chain(*column_field_names.values()):
                    row_string += f"|{colname[-10:]}:{array[indice][colname]:.4f}"

                row_string += f"||SCORE:{array[indice]['Score']}|"
                print(row_string)

        return best_docs_dict
        # return array["DocId"][best_documents_indices]
    
    def search(self, query, topk=5, num_of_max_sim_words=10, threshhold=0.7, return_dict=False, bm25plus=False):
        self.okapidelta = 1 if bm25plus else 0
        tokens = self._tokenize_text(query)
        logger.debug(f"Tokens taken from query: {tokens}")

        expanded_query = self.expand_query(tokens)
        logger.debug(f"Expanded tokens from query: {expanded_query}")

        expanded_query_with_similiar_words = {}

        for field in self.fields:
            expanded_query_with_similiar_words[field] = {
                word: similiar_word_dict
                for word, similiar_word_dict in self.get_similiar_words_for_words(
                    field, expanded_query, num_of_max_sim_words, threshhold
                ).items()
                if similiar_word_dict
            }

        logger.debug(f"Similiar words from query: \n{expanded_query_with_similiar_words}")

        term_importances = {
            field: {
                term: self.calculate_term_importance(field, term, similar_terms_dict)
                for term, similar_terms_dict in expanded_query_with_similiar_words[field].items()
            }
            for field in self.fields
        }
        logger.debug("Term importances: {}".format(term_importances))

        documents_ids_dict = self.fetch_documents(
            expanded_query_with_similiar_words,
            term_importances,
            topk,
        )

        return documents_ids_dict if return_dict else list(documents_ids_dict)
