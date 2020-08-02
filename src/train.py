import os

import dill as pickle
from tqdm import tqdm

from smt_utils import detokenize_od, tokenize_en, tokenize_od, train_ibmmodel2, translate

if __name__ == "__main__":
    data_dir = os.path.join("../data", "01_01_2020")
    print(f"Data directory: {data_dir}")

    filepaths = {
        "train_en": os.path.join(data_dir, "train.en"),
        "train_od": os.path.join(data_dir, "train.od"),
        "val_en": os.path.join(data_dir, "val.en"),
        "val_od": os.path.join(data_dir, "val.od"),
        "test_en": os.path.join(data_dir, "test.en"),
        "test_od": os.path.join(data_dir, "test.od"),
    }
    print(f"File-paths: {filepaths}")

    for data_type in filepaths:
        if not os.path.isfile(filepaths[data_type]):
            raise FileNotFoundError

    text = {}
    for data_type in filepaths:
        filepath = filepaths[data_type]
        with open(filepath, "r", encoding="utf-8") as f:
            text[data_type] = list(map(str.strip, f.readlines()))

    if (
        (len(text["train_en"]) != len(text["train_od"]))
        or (len(text["val_en"]) != len(text["val_od"]))
        or (len(text["test_en"]) != len(text["test_od"]))
    ):
        print("Length count mismatched between the data types")
        raise AssertionError

    text_tokenized = {
        "train_en": [tokenize_en(sent) for sent in text["train_en"]],
        "train_od": [tokenize_od(sent) for sent in text["train_od"]],
        "val_en": [tokenize_en(sent) for sent in text["val_en"]],
        "test_en": [tokenize_en(sent) for sent in text["test_en"]],
    }

    # train IBM model 2
    print("IBM model training started..")
    ibm_model = train_ibmmodel2(
        src_text=text_tokenized["train_en"], trg_text=text_tokenized["train_od"], iterations=5
    )
    print("IBM model training completed.")

    # dump trained model
    os.makedirs("../models", exist_ok=True)
    with open(os.path.join("../models", "model.pkl"), "wb") as f:
        pickle.dump(ibm_model, f)
    print("Models dumped")

    # load model from file
    with open(os.path.join("../models", "model.pkl"), "rb") as f:
        ibm_model_loaded = pickle.load(f)

    # translate
    translations = {"train": [], "val": [], "test": []}

    for data_type in translations.keys():
        print(f"Translation process started for data type: {data_type}")
        for toks in tqdm(text_tokenized[data_type + "_en"]):
            translation_toks = translate(ibm_model=ibm_model_loaded, src_tokens=toks)
            translation = detokenize_od(translation_toks)
            translations[data_type].append(translation)

    # write translations to files
    translation_filenames = {"train": "train.out.od", "val": "val.out.od", "test": "test.out.od"}
    for data_type in translation_filenames:
        with open(
            os.path.join("../", translation_filenames[data_type]), "w", encoding="utf-8"
        ) as f:
            f.writelines(list(map(lambda x: x + "\n", translations[data_type])))
