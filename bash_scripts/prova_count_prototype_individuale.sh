#!/bin/bash
#This script runs the novel preparation pipeline on one novel.

TRAINING_FOLDER=$1
NOVEL_FOLDER=$2
WINDOW_SIZE=$3
TOP_CONTEXTS=$4
WEIGHT=$5

ORIGINAL_FOLDER=${TRAINING_FOLDER}/${NOVEL_FOLDER}

FILENAME=$(ls ${ORIGINAL_FOLDER}/original_novel)
BOOK_NUMBER=${FILENAME/'.txt'/''}

python3 scripts/remove_gutenberg_header.py ${ORIGINAL_FOLDER}/original_novel ${BOOK_NUMBER}

CLEAN_NAME=${BOOK_NUMBER}_clean.txt
TEMP_FOLDER=${ORIGINAL_FOLDER}/temp
BOOK_NLP_OUTPUT_FOLDER=${ORIGINAL_FOLDER}/book_nlp_output
PROCESSED_NOVEL_FOLDER=${ORIGINAL_FOLDER}/processed_novel
CLEAN_PATH=${TEMP_FOLDER}/${CLEAN_NAME}

echo 'Starting with novel' ${BOOK_NUMBER}

cd ../book_nlp

./runjava novels/BookNLP -doc ../novel_aficionados/${CLEAN_PATH} -printHTML -p ../novel_aficionados/${BOOK_NLP_OUTPUT_FOLDER} -tok ../novel_aficionados/${BOOK_NLP_OUTPUT_FOLDER}/${BOOK_NUMBER}.tokens -f > /dev/null 2>&1 

cd ../novel_aficionados

echo 'Training on N2V on novel' ${BOOK_NUMBER}

FULL_FOLDER=/mnt/cimec-storage-sata/users/andrea.bruera/novel_aficionados/${ORIGINAL_FOLDER}

n2v test --on count_novels --model /mnt/cimec-storage-sata/users/andrea.bruera/novel_aficionados/count_models/count_wiki_2/count_wiki_2_cooccurrences.pickle --vocabulary /mnt/cimec-storage-sata/users/andrea.bruera/novel_aficionados/count_models/count_wiki/count_wiki_vocabulary_trimmed.pickle --folder ${FULL_FOLDER} --data ${BOOK_NUMBER} --window_size ${WINDOW_SIZE} --prototype --top_contexts 20 --weight 6 --write_to_file > /dev/null 2>&1   

rm -r ${BOOK_NLP_OUTPUT_FOLDER}
#rm -r ${TEMP_FOLDER}
rm -r ${PROCESSED_NOVEL_FOLDER}
echo 'All good with novel' ${BOOK_NUMBER}
