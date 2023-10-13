"""
README
# Alignment Script

This script performs word alignment between parallel corpora using the IBM-1 model and generates visualizations of the alignments.

## Prerequisites

- Python 3.x

## Usage

1. Open a terminal or command prompt.
2. Navigate to the directory where the `align.py` script is located.
3. Run the script with the following command:

    python3 align.py <source_folder> <target_folder>
    
   Replace `<source_folder>` and `<target_folder>` with the paths to the folders containing the source and target corpora respectively.
   NOTE: both of these folders must contain a Settings.xml. For examples of what these folders should contain, see https://github.com/sillsdev/machine.py/blob/main/samples/word_alignment.ipynb
   You will then need to change the Naming, FileNameBookNameForm, FileNamePostPart, and BooksPresent tags.
4. The script will perform word alignment using the IBM-1 model and generate alignment visualizations for each segment.
5. The alignment visualizations will be saved as PNG images in the `alignments` directory.
"""
from machine.corpora import ParatextTextCorpus
from machine.tokenization import LatinWordTokenizer
from machine.translation import (
    word_align_corpus,
    SymmetrizationHeuristic,
    SymmetrizedWordAlignmentModelTrainer,
)
from machine.translation.thot import (
    ThotWordAlignmentModelTrainer,
    ThotWordAlignmentModelType,
    ThotIbm1WordAlignmentModel,
    ThotSymmetrizedWordAlignmentModel,
)
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import logging
import sys
import matplotlib

# Set up logging to keep track of error handling in both the terminal and for getting help if needed.
logging.basicConfig(
    filename="alignment.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Create a StreamHandler to display log messages in the terminal
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# Get the root logger and add the StreamHandler
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)


if not os.path.exists("alignments"):
    os.makedirs("alignments")

# arguments must be folders that contain a Settings and .SFM file
parser = argparse.ArgumentParser(description="Generate token alignments")

parser.add_argument(
    "source_folder",
    metavar="S",
    type=str,
    nargs=1,
    help="A folder path that contains a settings file and language data",
)
parser.add_argument(
    "target_folder",
    metavar="T",
    type=str,
    nargs=1,
    help="A folder path that contains a settings file and language data",
)

args = parser.parse_args()


def validate_folder_paths(s, t):
    if not os.path.exists(s):
        logging.error("Invalid Source Folder")
    if not os.path.exists(t):
        logging.error("Invalid Target Folder")


def load_parallel_corpus(source_folder, target_folder):
    validate_folder_paths(s=source_folder, t=target_folder)

    source_corpus = ParatextTextCorpus(source_folder)
    target_corpus = ParatextTextCorpus(target_folder)
    parallel_corpus = source_corpus.align_rows(target_corpus).tokenize(
        LatinWordTokenizer()
    )
    return parallel_corpus


try:
    parallel_corpus = load_parallel_corpus(args.source_folder[0], args.target_folder[0])
except Exception as e:
    logging.error(f"Error generating the parallel corpus: {e}")

try:
    aligned_corpus = word_align_corpus(parallel_corpus.lowercase(), aligner="ibm1")
except Exception as e:
    logging.error(f"Error generating aligned corpus: {e}")

# training a model from scratch
try:
    os.makedirs("out/VBL-WEB-IBM1", exist_ok=True)
    trainer = ThotWordAlignmentModelTrainer(
        ThotWordAlignmentModelType.IBM1,
        parallel_corpus.lowercase(),
        "out/VBL-WEB-IBM1/src_trg",
    )

    trainer.train(
        lambda status: logging.info(
            "Training IBM-1 model: {:.2%}".format(status.percent_completed)
        )
    )
    trainer.save()
    logging.info("IBM-1 model saved")
except Exception as e:
    logging.info(f"Error training model: {e}")

# aligning sentences in batches instead of one at a time
try:
    model = ThotIbm1WordAlignmentModel("out/VBL-WEB-IBM1/src_trg")

    segment_batch = list(parallel_corpus.lowercase())
    alignments = model.align_batch(segment_batch)
except Exception as e:
    logging.error(f"Error creating alignments: {e}")

# symmetrize alignment models for better quality
try:
    src_trg_trainer = ThotWordAlignmentModelTrainer(
        ThotWordAlignmentModelType.IBM1,
        parallel_corpus.lowercase(),
        "out/VBL-WEB-IBM1/src_trg",
    )
    trg_src_trainer = ThotWordAlignmentModelTrainer(
        ThotWordAlignmentModelType.IBM1,
        parallel_corpus.invert().lowercase(),
        "out/VBL-WEB-IBM1/trg_src",
    )
    symmetrized_trainer = SymmetrizedWordAlignmentModelTrainer(
        src_trg_trainer, trg_src_trainer
    )
    symmetrized_trainer.train(
        lambda status: logging.info(f"{status.message}: {status.percent_completed:.2%}")
    )
    symmetrized_trainer.save()

    src_trg_model = ThotIbm1WordAlignmentModel("out/VBL-WEB-IBM1/src_trg")
    trg_src_model = ThotIbm1WordAlignmentModel("out/VBL-WEB-IBM1/trg_src")
    symmetrized_model = ThotSymmetrizedWordAlignmentModel(src_trg_model, trg_src_model)
    symmetrized_model.heuristic = SymmetrizationHeuristic.GROW_DIAG_FINAL_AND
except Exception as e:
    logging.error(f"Error symmetrizing trainer: {e}")

# align on data
try:
    segment_batch = list(parallel_corpus.lowercase())
    alignments = symmetrized_model.align_batch(segment_batch)

    src_trg_model = ThotIbm1WordAlignmentModel("out/VBL-WEB-IBM1/src_trg")
    trg_src_model = ThotIbm1WordAlignmentModel("out/VBL-WEB-IBM1/trg_src")
    symmetrized_model = ThotSymmetrizedWordAlignmentModel(src_trg_model, trg_src_model)
    symmetrized_model.heuristic = SymmetrizationHeuristic.GROW_DIAG_FINAL_AND

    segment_batch = list(parallel_corpus.lowercase())
    alignments = symmetrized_model.align_batch(segment_batch)
except Exception as e:
    logging.error(f"Error aligning data: {e}")


def create_alignment_images(data, verse):
    fig = plt.figure(figsize=(7, 10), dpi=300)
    ax = plt.subplot()

    ncols = 2
    nrows = data.shape[0]

    ax.set_xlim(0, ncols + 1)
    ax.set_ylim(0, nrows)

    positions = [0.75, 2]
    columns = ["source", "target"]

    for i in range(nrows):
        for j, column in enumerate(columns):
            ax.annotate(
                xy=(positions[j], i + 0.5),
                text=data[column].iloc[i],
                ha="center",
                va="center",
            )

    column_names = ["Source\nLanguage", "Target\nLanguage"]
    for i in range(nrows):
        for index, column in enumerate(column_names):
            ax.annotate(
                xy=(positions[index], nrows),
                text=column_names[index],
                ha="center",
                va="bottom",
                weight="bold",
            )

    ax.plot(
        [ax.get_xlim()[0], ax.get_xlim()[1]],
        [nrows, nrows],
        lw=1.5,
        color="black",
        marker="",
        zorder=4,
    )
    ax.plot(
        [ax.get_xlim()[0], ax.get_xlim()[1]],
        [0, 0],
        lw=1.5,
        color="black",
        marker="",
        zorder=4,
    )
    for x in range(1, nrows):
        ax.plot(
            [ax.get_xlim()[0], ax.get_xlim()[1]],
            [x, x],
            lw=1.15,
            color="gray",
            ls=":",
            zorder=3,
            marker="",
        )

    ax.set_axis_off()

    plt.savefig(
        "alignments/alignment_{}.png".format(verse),
        dpi=300,
        transparent=False,
        bbox_inches="tight",
    )

    matplotlib.pyplot.close()


verse = 1

for (source_segment, target_segment), alignment in zip(segment_batch, alignments):
    spa = []
    eng = []

    source = 0

    for i in alignment:
        true_indices = np.where(i)[0]  # Find the indices where the value is True
        if len(true_indices) > 0:
            key1 = source_segment[source]
            key2 = str(true_indices[0])  # Get the first index where the value is True
            value2 = target_segment[int(key2)]

            spa.append(key1)
            eng.append(value2)

        source += 1

    ### create a visualization ###
    table = pd.DataFrame(
        {
            "source": spa,
            "target": eng,
        }
    )

    create_alignment_images(data=table, verse=verse)

    verse += 1

logging.info("Task Completed!")
