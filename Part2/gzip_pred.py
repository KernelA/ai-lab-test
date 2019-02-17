import gzip
import os
import json

PATH_TO_PRED = os.path.join(".", "pred.vw")
PATH_TO_PRIVATE = os.path.join(".", "private-res.jsonlines.gz")
PATH_TO_TEST = os.path.join(".", "test.vw")

if os.path.exists(PATH_TO_PRIVATE):
    os.rename(PATH_TO_PRIVATE, "{}-{backup}".fomrat(PATH_TO_PRIVATE))

with open(PATH_TO_PRED, "r", encoding="utf-8") as pred_file\
     ,open(PATH_TO_TEST, "r", encoding="utf-8") as test_features\
    , gzip.open(PATH_TO_PRIVATE, "wt", encoding="utf-8") as file_res:

    pred = tuple(map(int, pred_file.readlines()))
    for i, line in enumerate(test_features):
        author = line.split("|")[2]
        author = int(author.split("=")[1])
        file_res.write("{}\n".format(json.dumps({"author": author, "gender": "male" if pred[i] == 1 else "female"})))
