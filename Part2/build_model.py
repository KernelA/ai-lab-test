import gzip
import json
import logging
import os
import pickle
import re
import string
import unicodedata
import sys
from typing import Dict, Any

from nltk.tokenize import TweetTokenizer

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)

FORMAT = "%(asctime)s %(name)s [%(levelname)s] {%(funcName)s} %(message)s"
logging.basicConfig(format=FORMAT)

RUSSIAN_LETTERS = frozenset("абвгдежзийклмнопрстуфхцчшщъыьэюя")
PUNCTS = frozenset(string.punctuation)
PUNCT_WITHOUT_DASH_REGEX = "[{}]+".format("".join(PUNCTS - set("-")))
PUNCT_REGEX = f"[{string.punctuation}]+"
REPLACE_REGEX = "\[(.+?)\]"
REPLACE_TARGET = {"male": 1, "female": -1}

DUMP_DIR = "dumps"

if not os.path.exists(os.path.join(".", DUMP_DIR)):
    os.mkdir(os.path.join(".", DUMP_DIR))

AUTHOR_GENDERS_DUMP = os.path.join(".", DUMP_DIR, "author-genders.pickle")
AUTHOR_TEST_DUMP = os.path.join(".", DUMP_DIR, "author-tests.pickle")


def save_pickle_dump(path_to_dump: str, obj):
    LOGGER.info("Save a dump to: {}".format(path_to_dump))
    with open(path_to_dump, "wb") as dump_file:
        pickle.dump(obj, dump_file)


def load_pickle_dump(path_to_dump: str):
    LOGGER.info("Load a dump from: {}".format(path_to_dump))
    with open(path_to_dump, "rb") as dump_file:
        obj = pickle.load(dump_file)

    return obj


def load_genders(path_to_gzip: str) -> Dict[int, str]:
    if os.path.exists(AUTHOR_GENDERS_DUMP):
        return load_pickle_dump(AUTHOR_GENDERS_DUMP)
    else:
        with gzip.open(path_to_gzip, "rt", encoding="utf-8") as file:
            lines = tuple(map(json.loads, file.readlines()))

        res = {line["author"]: line["gender"] for line in lines}

        save_pickle_dump(AUTHOR_GENDERS_DUMP, res)
        LOGGER.info("Total authors: {}".format(len(lines)))
        LOGGER.info("Total unique authors: {}".format(len(res)))
        LOGGER.info("Total count = unique count {}".format(len(res) == len(lines)))
    return res


def load_test_authors(path_to_gzip: str) -> set:
    if os.path.exists(AUTHOR_TEST_DUMP):
        return load_pickle_dump(AUTHOR_TEST_DUMP)
    else:
        with gzip.open(path_to_gzip, "rt", encoding="utf-8") as file:
            lines = map(json.loads, file.readlines())

        res = {line["author"] for line in lines}
        save_pickle_dump(AUTHOR_TEST_DUMP, res)
    return res


def extract_features(author_posts: Dict[str, Any]) -> Dict[str, float]:
    words = author_posts["words"]

    def normalize(features_dict: Dict[str, float], *keys, norm_key: str):
        for key in keys:
            features_dict[key] = 0 if features_dict[norm_key] == 0 else features_dict[key] / features_dict[norm_key]

    features = {
        "tot_char": 0,
        "tot_punct": 0,
        "ratio_lc": 0,
        "ratio_uc": 0,
        "ratio_comma": 0,
        "ratio_colon": 0,
        "ratio_semicolon": 0,
        "ratio_question": 0,
        "ratio_exclam": 0,
        "ratio_period": 0,
        "ratio_left_brace": 0,
        "ratio_right_brace": 0,
        "tot_words": 0,
        "avg_char_per_word": 0
    }

    features["author"] = author_posts["author"]


    for word in words:
        features["tot_char"] += len(word)
        if re.fullmatch(f"[a-я][a-я-]*[a-я]", word, re.IGNORECASE) is not None:
            for char in word:
                if char.isalpha():
                    if char.islower():
                        features["ratio_lc"] += 1
                    else:
                        features["ratio_uc"] += 1

            features["tot_words"] += 1
            features["avg_char_per_word"] += len(word)

        else:
            for char in word:
                if char in PUNCTS:
                    features["tot_punct"] += 1
                if char == "(":
                    features["ratio_left_brace"] += 1
                elif char == ")":
                    features["ratio_right_brace"] += 1
                elif char == ",":
                    features["ratio_comma"] += 1
                elif char == ":":
                    features["ratio_colon"] += 1
                elif char == ";":
                    features["ratio_semicolon"] += 1
                elif char == ".":
                    features["ratio_period"] += 1
                elif char == "?":
                    features["ratio_question"] += 1
                elif char == "!":
                    features["ratio_exclam"] += 1


    features["ratio_right_brace"] = features["ratio_right_brace"] - features["ratio_left_brace"]

    if features["ratio_right_brace"] < 0:
        features["ratio_right_brace"] = 0

    features["ratio_left_brace"] = features["ratio_left_brace"] - features["ratio_right_brace"]

    if features["ratio_left_brace"] < 0:
        features["ratio_left_brace"] = 0

    normalize(features, "ratio_lc", "ratio_uc", norm_key="tot_char")
    normalize(features, "ratio_comma", "ratio_colon", "ratio_semicolon", "ratio_question",
              "ratio_exclam", "ratio_period", "ratio_left_brace", "ratio_right_brace", norm_key="tot_punct")
    normalize(features, "avg_char_per_word",  norm_key="tot_words")

    return features


def is_word_or_punct(word: str):
    # Слова
    if re.fullmatch(f"[a-я][a-я-]*[a-я]", word, re.IGNORECASE) is not None:
        return True
    # Только пунктуация
    elif re.fullmatch(PUNCT_REGEX, word, re.IGNORECASE) is not None:
        return True
    else:
        False


def extract_test_train_features(path_to_compressed_file: str,
                           path_to_features_train: str,
                           path_to_features_test: str,
                           authors_with_genders: Dict[int, str],
                           authors_test: set):
    tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True)


    with gzip.open(path_to_compressed_file, "rt", encoding="utf-8") as file_in\
        , open(path_to_features_train, "w", encoding="utf-8") as file_train\
        , open(path_to_features_test, "w", encoding="utf-8") as file_test:

        for i, line in enumerate(map(json.loads, file_in)):
            author_id = line["author"]

            if author_id in authors_with_genders or author_id in authors_test:
                str_line = line["text"].strip().replace("ё", "е")
                str_line = "".join(char for char in unicodedata.normalize("NFD", str_line) if unicodedata.category(char) != "Mn")
                str_line = re.sub(REPLACE_REGEX, "", str_line)
                cleaned_words = tuple(filter(is_word_or_punct, tokenizer.tokenize(str_line)))

                features = extract_features({"author": author_id, "words": cleaned_words})
                features_line = ("{key}:{val}".format(key=key, val=features[key]) for key in features if
                                 key != "gender" and key != "author")

                target = -1

                if author_id in authors_with_genders and 5 <= len(cleaned_words):
                    target = REPLACE_TARGET[authors_with_genders[author_id]]
                elif author_id in authors_test:
                    target = 1

                res_line = "{} |num {} |add author={}\n".format(target, " ".join(features_line), author_id)

                if author_id in authors_with_genders and 5 <= len(cleaned_words):
                    file_train.write(res_line)
                elif author_id in authors_test:
                    file_test.write(res_line)


if __name__ == "__main__":
    path_to_genders_raw = os.path.join(".", "public.jsonlines.gz")

    if not os.path.exists(path_to_genders_raw):
        print("Path '{}' does not exist.".format(path_to_genders_raw))
        sys.exit(1)

    author_with_genders = load_genders(path_to_genders_raw)

    path_to_authors_test = os.path.join(".", "private.jsonlines.gz")

    if not os.path.exists(path_to_authors_test):
        print("Path '{}' does not exist.".format(path_to_authors_test))
        sys.exit(1)

    test_authors = load_test_authors(path_to_authors_test)

    path_to_post_raw = os.path.join(".", "messages.jsonlines.gz")

    if not os.path.exists(path_to_post_raw):
        print("Path '{}' does not exist.".format(path_to_post_raw))
        sys.exit(1)

    path_to_train_features = os.path.join(".", "train.vw")
    path_to_test_features = os.path.join(".", "test.vw")

    extract_test_train_features(path_to_post_raw, path_to_train_features, path_to_test_features
                                , author_with_genders, test_authors)


