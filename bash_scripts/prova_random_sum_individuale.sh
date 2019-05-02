#!/bin/bash
#This script runs the novel preparation pipeline on one novel.

TRAINING_FOLDER=$1
NOVEL_FOLDER=$2
lambda=$3
window_decay=$4
subsampling_decay=$5
subsampling=$6

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

n2v test --on novels --model /mnt/cimec-storage-sata/users/andrea.bruera/wiki_training/data/wiki_w2v_2018_size400_max_final_vocab250000_sg1 --folder ${FULL_FOLDER} --data ${BOOK_NUMBER} --alpha 1 --neg 3 --window 15 --sample ${subsampling} --epochs 1 --lambda ${lambda} --sample-decay ${subsampling_decay} --window-decay ${window_decay} --simil_out --random_sentences --sum_only > /dev/null 2>&1  

#rm -r ${BOOK_NLP_OUTPUT_FOLDER}
rm -r ${TEMP_FOLDER}
rm -r ${PROCESSED_NOVEL_FOLDER}
python3 scripts/get_damn_evaluation.py ${ORIGINAL_FOLDER} ${BOOK_NUMBER}
echo 'All good with novel' ${BOOK_NUMBER}
