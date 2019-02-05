#!/bin/bash

DOWNLOADED_NOVELS_FOLDER='novels'

#mkdir ${DOWNLOADED_NOVELS_FOLDER}
#echo 'Downloading books...'
#python3 scripts/get_books/download_books.py ${DOWNLOADED_NOVELS_FOLDER} 

mkdir sum_big_test_novels
#mkdir big_test_novels
FOLDER='sum_big_test'
echo 'Created folder:' ${FOLDER}
#TRAINING_FOLDER=big_test_novels/${FOLDER}
TRAINING_FOLDER=sum_big_test_novels/${FOLDER}
mkdir ${TRAINING_FOLDER}
cp -r ${DOWNLOADED_NOVELS_FOLDER}/* ${TRAINING_FOLDER}/
echo 'Starting training'
for NOVEL_FOLDER in $(ls ${TRAINING_FOLDER});
    do
    ./prova_ufficiale_sum_individuale.sh ${TRAINING_FOLDER} ${NOVEL_FOLDER}
done
exit 0
