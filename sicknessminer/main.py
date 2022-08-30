#  This file is part of SicknessMiner.
#
#  SicknessMiner is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  SicknessMiner is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with SicknessMiner. If not, see <https://www.gnu.org/licenses/>.

"""
The main entry point.
"""

import argparse
import datetime
import os
import traceback
from os.path import expanduser
from subprocess import check_output
from typing import Iterator

import bert
import numpy as np
import tensorflow as tf
from bert.tokenization.bert_tokenization import FullTokenizer
# from tensorflow.python import keras
from tensorflow import keras

bert.loader.trace = lambda *a, **k: None

from sicknessminer.pubtator import normco2pubtator
from sicknessminer.pubtator.conll2pubtator import ConllIterator, \
    build_state_machine
from sicknessminer.pubtator.utils import PubTatorIterator, RawTextToPubTator
from sicknessminer.util import split_sentences

# added by me
import pandas as pd # had to install
import networkx as nx # had to install
import matplotlib.pyplot as plt
import seaborn as sns # had to install
import shutil
from metapub import PubMedFetcher # pip install metapub
import datetime

sns.set_theme()

CLASS_NAME = "Disease"

BERT_MODEL_PATH = "models/NER_SicknessMiner"
TEMP_DIRECTORY = "tmp"

START_TOKEN = "[CLS]"
END_TOKEN = "[SEP]"
PAD_TOKEN = "[PAD]"

NER_LABELS = ["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"]
DEFAULT_LABEL = "O"
VALID_LABEL_INDICES = {NER_LABELS.index("B"), NER_LABELS.index("I")}

NORMCO_PATH = "NormCo-deep-disease-normalization"
NORMCO_HEADER = "MeSH IDs\tMentions\tSpans\tPMID\n"

NORMCO_COMMAND = [
    expanduser("~/anaconda3/envs/entity_normalization/bin/python"),
    "NormCo-deep-disease-normalization/entity_normalization/normalize.py",
    "--weight_init",
    "models/NEN_SicknessMiner/model.pth_bestacc_00026",
    "--disease_dict",
    "models/NEN_SicknessMiner/preprocessed_data/NCBID/concept_dict.tsv",
    "--vocab_file",
    "models/NEN_SicknessMiner/preprocessed_data/NCBID/vocab.txt",
    "--embeddings_file",
    "models/NEN_SicknessMiner/preprocessed_data/NCBID/word_embeddings_init.npy",
    "--tags_file",
    None,
    "--model",
    "GRU",
    "--sequence_len",
    "20",
    "--scoring_type",
    "euclidean",
    "--output_dim",
    "200"
]

conda_env = os.getenv('CONDA_ENV')
if conda_env is not None:
    NORMCO_COMMAND[0] = conda_env
elif not os.path.isfile(NORMCO_COMMAND[0]):
    path = expanduser("~/.conda/envs/entity_normalization/bin/python")
    if os.path.isfile(path):
        NORMCO_COMMAND[0] = path
    else:
        raise Exception("Conda directory not found! Please set the CONDA_ENV "
                        "variable to the path of your conda python "
                        f"environment. Example {path}")

NORMCO_INPUT_FILE_INDEX = NORMCO_COMMAND.index("--tags_file") + 1


def create_arguments_parser():
    """
    Creates the argument parser.

    :return: the argument parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Runs the SicknessMiner on the PubTator file(s).",
        usage="%(prog)s <PubTator File(s)> <Output File>")

    parser.add_argument(
        "files", metavar="FILES", type=str, nargs="*",
        help="1) the input file(s); 2) the output file.")

    parser.add_argument('--raw', '-r', dest='raw',
                        action="store_true",
                        help="If set, the input files are treated as raw text "
                             "files. Otherwise, they are treated as "
                             "files in the PubTator format.")
    parser.set_defaults(raw=False)

    return parser


def find_last_model(directory, prefix):
    """
    Finds the last model in `directory`, with `prefix`.

    :param directory: the directory
    :type directory: str
    :param prefix: the prefix
    :type prefix: str
    :return: the last model
    :rtype: str or None
    """
    files = os.listdir(directory)
    model_number = 0
    model_name = None
    for f in files:
        if not f.startswith(prefix):
            continue
        number = f[len(prefix):]
        dot_index = number.rfind(".")
        if dot_index > 0:
            number = int(number[:dot_index])
            if number > model_number:
                model_number = number
                model_name = f[:-dot_index - 1]

    if model_name is not None:
        model_name = os.path.join(directory, model_name)

    return model_name


def read_bert_parameters(parameters_path):
    return bert.params_from_pretrained_ckpt(parameters_path)


def create_biobert_model(bert_parameters) -> keras.Model:
    max_seq_len = bert_parameters["max_position_embeddings"]

    # bert_ckpt_file = find_last_model(BERT_MODEL_PATH, "model.ckpt-")
    bert_ckpt_file = "models/NER_SicknessMiner\model.ckpt-1695"
    ckpt_reader = tf.train.load_checkpoint(bert_ckpt_file)

    l_bert = bert.BertModelLayer.from_params(bert_parameters, name="bert")

    l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
    output_layer = l_bert(l_input_ids)  # [batch_size, max_seq_len, hidden_size]

    num_labels = len(NER_LABELS)
    hidden_size = bert_parameters["hidden_size"]

    output_weight = tf.Variable(
        name="output_weights", shape=[num_labels, hidden_size],
        initial_value=ckpt_reader.get_tensor("output_weights"))
    output_bias = tf.Variable(
        name="output_bias", shape=[num_labels],
        initial_value=ckpt_reader.get_tensor("output_bias"))

    output_layer = tf.reshape(output_layer, [-1, hidden_size])
    logits = tf.matmul(output_layer, output_weight, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.reshape(logits, [-1, max_seq_len, num_labels])
    log_probabilities = tf.nn.log_softmax(logits, axis=-1)

    model = keras.Model(inputs=l_input_ids, outputs=log_probabilities)
    model.build(input_shape=(None, max_seq_len))

    bert.load_stock_weights(l_bert, bert_ckpt_file)

    return model


def create_dataset_from_sentences(sentences, tokenizer, maximum_length,
                                  start_id, end_id, pad_id):
    """
    Transforms a sequence of sentences into a dataset to be passed to the
    BERT model.

    :param sentences: the sentences as list of tokens
    :type sentences: List[List[str]]
    :param tokenizer: the tokenizer
    :type tokenizer: FullTokenizer
    :param maximum_length: the maximum sequence length
    :type maximum_length: int
    :param start_id: the id of the first token
    :type start_id: int
    :param end_id: the id of the last token
    :type end_id: int
    :param pad_id: the id of the padding token
    :type pad_id: int
    :return: the dataset
    :rtype: np.ndarray
    """

    dataset = []
    token_type_ids = []
    masks = []
    for tokens in sentences:
        tokens = tokens[0:maximum_length - 1]
        identifiers = tokenizer.convert_tokens_to_ids(tokens)
        sentence_data = [start_id] + identifiers + [end_id]
        sequence_length = len(sentence_data)
        padding_length = max(maximum_length - sequence_length, 0)
        if sequence_length < maximum_length:
            sentence_data += [pad_id] * padding_length
        dataset.append(sentence_data)
        token_type_ids.append([0] * maximum_length)
        masks.append([1] * sequence_length + [0] * padding_length)

    return ([np.array(dataset, dtype=np.int32),
             np.array(token_type_ids, dtype=np.int32)],
            np.array(masks, dtype=np.int32))


def convert_predictions_to_conll(sentences, predictions) -> Iterator[str]:
    """
    Converts the predictions to CoNLL format.

    :param sentences: the sentences as list of tokens
    :type sentences: List[List[str]]
    :param predictions: the predictions
    :type predictions: np.ndarray
    :return: the predictions in the CoNLL format
    :rtype: Iterator[str]
    """
    for i in range(len(sentences)):
        token_sentences = sentences[i]
        if len(token_sentences) == 0:
            continue
        token_predictions = predictions[i]
        prediction_length = len(token_predictions) - 1
        start_index = 1
        buffer = token_sentences[0]
        for j in range(1, len(token_sentences)):
            if token_sentences[j].startswith("##"):
                buffer += token_sentences[j][2:]
            else:
                if start_index < prediction_length:
                    label_index = token_predictions[start_index].argmax()
                    if label_index in VALID_LABEL_INDICES:
                        label = f"{NER_LABELS[label_index]}-{CLASS_NAME}"
                    else:
                        label = DEFAULT_LABEL
                else:
                    label = DEFAULT_LABEL
                yield f"{buffer}\t{label}"
                buffer = token_sentences[j]
                start_index = j + 1
        if start_index < prediction_length:
            label_index = token_predictions[start_index].argmax()
            if label_index in VALID_LABEL_INDICES:
                label = f"{NER_LABELS[label_index]}-{CLASS_NAME}"
            else:
                label = DEFAULT_LABEL
        else:
            label = DEFAULT_LABEL
        yield f"{buffer}\t{label}"
        yield ""


def write_normco_entry(pubtator_entry, writer):
    if len(pubtator_entry.mentions) == 0:
        return

    writer.write("\t")
    writer.write("|".join(map(lambda x: x.text, pubtator_entry.mentions)))
    writer.write("\t")
    writer.write("|".join(map(lambda x: f"{x.begin_index} {x.end_index}",
                              pubtator_entry.mentions)))
    writer.write("\t")
    writer.write(pubtator_entry.identifier)
    writer.write("\n")


def main(input_paths, output_path, raw=False):
    """
    The main method.

    :param input_paths: the input files in PubTator format.
    :type input_paths: List[str]
    :param output_path: the output file
    :type output_path: str
    """
    bert_parameters = read_bert_parameters(BERT_MODEL_PATH)
    max_seq_len = bert_parameters["max_position_embeddings"]
    tokenizer = FullTokenizer(
        os.path.join(BERT_MODEL_PATH, "vocab.txt"), do_lower_case=False)

    start_id = tokenizer.convert_tokens_to_ids([START_TOKEN])[0]
    end_id = tokenizer.convert_tokens_to_ids([END_TOKEN])[0]
    pad_id = tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]

    bert_model = create_biobert_model(bert_parameters)

    timestamp = str(datetime.datetime.now()) \
        .replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")

    os.makedirs(TEMP_DIRECTORY, exist_ok=True)

    ner_output_filepath = \
        os.path.join(TEMP_DIRECTORY, f"ner_output_{timestamp}.txt")
    ner_output_file = open(ner_output_filepath, "w")

    normco_input_filepath = \
        os.path.join(TEMP_DIRECTORY, f"normco_input_{timestamp}.txt")
    normco_input_file = open(normco_input_filepath, "w")
    normco_input_file.write(NORMCO_HEADER)
    try:
        if raw:
            iterator = RawTextToPubTator(input_paths)
        else:
            iterator = PubTatorIterator(input_paths)

        for pubtator_entry in iterator:
            sentences = [tokenizer.tokenize(pubtator_entry.title)]

            sentences.extend(map(lambda x: tokenizer.tokenize(x),
                                 split_sentences(pubtator_entry.abstract)))

            dataset, mask = create_dataset_from_sentences(
                sentences, tokenizer, max_seq_len, start_id, end_id, pad_id)
            predictions = bert_model(dataset, mask=mask).numpy()
            conll_iterator = ConllIterator(
                convert_predictions_to_conll(sentences, predictions))
            char_iterator = \
                pubtator_entry.to_string_without_mentions().__iter__()
            state = build_state_machine(
                conll_iterator, ner_output_file, reset=False)
            while not state.is_final_state():
                state = state.perform_operation(char_iterator)
            mentions = state.parameters.mentions
            pubtator_entry.mentions = mentions
            write_normco_entry(pubtator_entry, normco_input_file)

        ner_output_file.close()
        normco_input_file.close()

        NORMCO_COMMAND[NORMCO_INPUT_FILE_INDEX] = normco_input_filepath
        my_env = os.environ.copy()
        my_env["PYTHONPATH"] = NORMCO_PATH
        normco_result = check_output(NORMCO_COMMAND, env=my_env).decode("utf-8")

        normco2pubtator.main(normco_result.splitlines().__iter__(),
                             ner_output_filepath, output_path,
                             entity_type=CLASS_NAME, override_type=True)
        os.remove(ner_output_filepath)
        os.remove(normco_input_filepath)
    except Exception:
        if not ner_output_file.closed:
            ner_output_file.close()
        if not normco_input_file.closed:
            normco_input_file.close()
        traceback.print_exc()


# Stephanie's extensions


def process_output(output, dict, target_diseases):

    '''
    Creates a dictionary of the disease co-mentions for each target disease in one output file
    output = raw output file, the function will directly modigy this file
    dict = initial dictionary
    target_diseases = list of target diseases
    '''

    output_dir = "./output_files/" + output

    # Delete first two lines of the file (title and abstract)
    lines = []
    with open(output_dir, 'r') as fp:
        lines = fp.readlines()

    with open(output_dir, 'w') as fp:
        for number, line in enumerate(lines):
            if number not in [0, 1]:
                fp.write(line)

    # Replace commas with hyphens (in disease names...?)
    fp = open(output_dir, 'r')
    data = fp.read()
    data = data.replace(', ', '-')
    fp.close()

    # Replace tabs with commas
    fp = open(output_dir, 'r')
    data = fp.read()
    data = data.replace('\t', ',')
    fp.close()

    fp = open(output_dir, 'w')
    fp.write(data)
    fp.close()

    # Convert copy.txt to copy.csv

    # Won't work for diseases with commas -> try to figure out *.tsv
    # Ideas: delete all commas from the *.txt file?

    read_file = pd.read_csv(output_dir)
    read_file.to_csv("current_output.csv", index=None)

    # Put into a dataframe
    names = ['ID', 'Start', 'End', 'Disease', 'Type', 'MESH']
    df = pd.read_csv("current_output.csv", names=names)

    # Extract the Disease column
    diseases = df[df.columns[3]].str.lower() # change all to LOWERCASE

    present_targets = []

    # Check to see which target diseases are in the text
    # REMEMBER TO CONSIDER LOWER AND UPPER CASE LATER -> DONE! (see 6 lines above)
    for target in target_diseases:
        for disease in diseases:
            if target == disease and target not in present_targets:
                present_targets.append(target)

    # Right now, the below will count more than one co-mention for each abstract/title
    for disease in diseases:
        if disease in present_targets:
            continue
        for target in present_targets:
            if disease not in dict[target]:
                dict[target][disease] = 1
            else:
                dict[target][disease] += 1

    return dict


def process_all_outputs(folder, dict, target_diseases):
    '''
    Creates a dictionary for the disease co-mentions of all target diseases in all output files
 	folder = folder with all the output files
  	dict = empty initialized dictionary
  	'''
    for file in os.listdir(folder):
        dict = process_output(file, dict, target_diseases)
        os.remove("current_output.csv")
    return dict


def retrieve_top_outputs(num, dict):
	'''
    Returns a list of the top co-mentioned diseases
    Creates a file of the top co-mentioned diseases
	num = number of top outputs to retrieve
	dict = processed dictionary
	'''
	list = []
	for key in dict.keys():
		for sub_key in dict[key].keys():
			list.append((key, sub_key, dict[key][sub_key]))
	
	sorted_list = sorted(list, key=lambda tup:tup[2], reverse=True)
	top_list = sorted_list[:num]

	with open("top.txt", 'w') as fp:
		for tuple in top_list:
			item_num = 1
			for item in tuple:
				if item_num != 3:
					fp.write("%s\t" % item)
				else:
					fp.write("%s\n" % item)
				item_num += 1
		
	return top_list


def make_graph(target_diseases, num):
    '''
    Creates a knowledge graph of the top disease co-mentions for the target diseases
    target_diseases = list of target diseases
    num = number of top co-mentions to display on the knowledge graph (int)
    '''

    dict = {}
    for target in target_diseases:
        dict[target] = {}

    dict = process_all_outputs("output_files", dict, target_diseases)

    top_list = retrieve_top_outputs(num, dict)

    file = "top.txt"
    names = ['Index_disease', 'Associated_disease', 'Co-mentions']

    df = pd.read_table(file, names=names)

    source = df['Index_disease'].values.tolist()
    target = df['Associated_disease'].values.tolist()
    edge = df['Co-mentions'].values.tolist()

    kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':edge})

    # Create a directed-graph from a dataframe
    G = nx.from_pandas_edgelist(kg_df, "source", "target", 
                                edge_attr=True, create_using=nx.MultiDiGraph())

    plt.figure()
    pos = nx.spring_layout(G)
    # nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos=pos)
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=1500, edge_color=edge, edge_cmap=plt.cm.GnBu, pos=pos)
    plt.show()


def get_inputs(target_diseases, start_year, end_year, max_files):
    '''
    Extracts all titles and abstracts from PubMed based on search criteria
    disease = name of the disease of interest
    start_year = earliest year of publication
    end_year = latest year of publication
    max_files = maximum number of abstracts to extract
    '''

    fetch = PubMedFetcher()

    # Delete all contents of input_files and outputs
    for file in os.listdir('input_files'):
        os.remove(os.path.join('input_files', file))
    for file in os.listdir('output_files'):
        os.remove(os.path.join('output_files', file))
    print("All previous files removed")

    count = 1
    for disease in target_diseases:
        for year in range(start_year, end_year+1): # choose some years

            pmids = fetch.pmids_for_query(disease+" "+str(year)+"/01/01[MDAT] : "+str(year)+"/12/31[MDAT]", retmax=10000000)
            print("Number of articles in " +  str(year) + ': ' + str(len(pmids)))

            for pmid in pmids:
                article = fetch.article_by_pmid(pmid)
                title = article.title
                abstract = article.abstract
                if (title == None) or (abstract == None):
                    continue
                else:
                    # Write into a text file
                    file_path = 'input_files/input' + str(count) + '.txt'
                    with open(file_path, 'w') as text_file:
                        text = title + abstract
                        text = str(text.encode("utf-8")) # this line and the next don't work perfectly for fixing the error that occurs when a title/abstract has special characters
                        text = text[2:len(text)-1] # when converting back to string, a b"" is added around the text
                        text_file.write(text)
                    count += 1
                    if count > max_files:
                        break # stop when we have 100 articles
            if count > max_files:
                print("Early stop at " + str(max_files) + " articles")
                break

    print("Total number of successful articles:", count-1)


def put_it_together(target_diseases, start_year, end_year, max_files, num, input_folder, output_folder):
    get_inputs(target_diseases, start_year, end_year, max_files)

    # Run the code for each abstract
    count = 1
    for file in os.listdir(input_folder[0]): 
        input_file = input_folder[0] + "/" + file
        output_file = output_folder + "/output" + str(count) + ".txt"
        main(input_file, output_file, raw=raw_files)
        count += 1

    # Create the knowledge graph based on the output files in the outputs folder
    make_graph(target_diseases, num)


if __name__ == '__main__':
    st = datetime.datetime.now()

    parser = create_arguments_parser()
    arguments = parser.parse_args()

    if len(arguments.files) < 2:
        raise Exception(f"At least two arguments are expected. Found:\t"
                        f"{arguments.files}")

    '''
    input_files = arguments.files[:-1]
    output_file = arguments.files[-1]
    raw_files = arguments.raw
    main(input_files, output_file, raw=raw_files)
    '''

    input_folder = arguments.files[:-1] 
    output_folder = arguments.files[-1]
    raw_files = arguments.raw

    # An example
    target_diseases = ["alzheimer's", "ad"]
    num = 50
    start_year = 2012
    end_year = 2022
    max_files = 10000

    put_it_together(target_diseases, start_year, end_year, max_files, num, input_folder, output_folder)
    
    et = datetime.datetime.now()

    elapsed_time = et - st 
    print("Total elapsed time:", elapsed_time)

# python sicknessminer/main.py --raw input_files output_files