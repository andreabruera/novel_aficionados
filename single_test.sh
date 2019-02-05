#!/bin/bash
#set -x
#This script runs the novel preparation pipeline on one novel.

ORIGINAL_FOLDER=$1
BOOK_NUMBER=$2
FILENAME=${BOOK_NUMBER}.txt

python3 scripts/remove_gutenberg_header.py ${ORIGINAL_FOLDER}/original_novel ${BOOK_NUMBER}

CLEAN_NAME=${BOOK_NUMBER}_clean.txt
TEMP_FOLDER=${ORIGINAL_FOLDER}/temp
BOOK_NLP_OUTPUT_FOLDER=${ORIGINAL_FOLDER}/book_nlp_output
PROCESSED_NOVEL_FOLDER=${ORIGINAL_FOLDER}/processed_novel
CLEAN_PATH=${TEMP_FOLDER}/${CLEAN_NAME}

l=($(wc -l ${CLEAN_PATH}))
lines=${l[0]}
count=$(expr $lines / 2 + 1)

echo $count

split -a 1 -l $count ${CLEAN_PATH} ${CLEAN_PATH}_part_

cd ../book_nlp

./runjava novels/BookNLP -doc ../novel_aficionados/${CLEAN_PATH} -printHTML -p ../novel_aficionados/${BOOK_NLP_OUTPUT_FOLDER} -tok ../novel_aficionados/${BOOK_NLP_OUTPUT_FOLDER}/${BOOK_NUMBER}.tokens -f

cd ../novel_aficionados

python3 scripts/prepare_for_n2v.py ${ORIGINAL_FOLDER} ${BOOK_NUMBER} ${CLEAN_PATH}

n2v test --on novels --model /mnt/cimec-storage-sata/users/andrea.bruera/wiki_training/data/wiki_w2v_2018_size400_max_final_vocab250000_sg1 --folder /mnt/cimec-storage-sata/users/andrea.bruera/novel_aficionados/${PROCESSED_NOVEL_FOLDER} --data ${BOOK_NUMBER} --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 0 --simil_out

rm -r ${BOOK_NLP_OUTPUT_FOLDER}
rm -r ${TEMP_FOLDER}
rm -r ${PROCESSED_NOVEL_FOLDER}
python3 scripts/get_damn_evaluation.py ${ORIGINAL_FOLDER} ${BOOK_NUMBER}
