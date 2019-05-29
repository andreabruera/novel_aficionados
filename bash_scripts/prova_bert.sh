#!/bin/bash

#TRAINING_MODE='bert'
#FOLDER=$1
TRAINING_MODE=$1

DOWNLOADED_NOVELS_FOLDER=dataset/novels/hundred_novels_by_6
#DOWNLOADED_NOVELS_FOLDER=novels_by_6
#mkdir ${DOWNLOADED_NOVELS_FOLDER}
#echo 'Downloading books...'
#python3 scripts/get_books/download_books.py ${DOWNLOADED_NOVELS_FOLDER} 
echo ${DOWNLOADED_NOVELS_FOLDER}
for SIX_NOVELS in $(ls ${DOWNLOADED_NOVELS_FOLDER});
    do
    TRAINING_FOLDER=${TRAINING_MODE}_test_novels
    mkdir -p ${TRAINING_FOLDER}
    echo 'Created folder: '${TRAINING_FOLDER}
    for novel in $(ls ${DOWNLOADED_NOVELS_FOLDER}/${SIX_NOVELS});
        do
        cp -r ${DOWNLOADED_NOVELS_FOLDER}/${SIX_NOVELS}/${novel} ${TRAINING_FOLDER}/
        ./bash_scripts/prova_${TRAINING_MODE}_individuale.sh ${TRAINING_FOLDER} ${novel} &   
        done
    wait
    done

#mkdir ${TRAINING_MODE}_test_novels/${FOLDER}
#cp -ri ${TRAINING_MODE}_test_novels_part_2/${FOLDER}/* ${TRAINING_MODE}_test_novels/${FOLDER}/
#cp -ri ${TRAINING_MODE}_test_novels_part_1/${FOLDER}/* ${TRAINING_MODE}_test_novels/${FOLDER}/
#rm -r ${TRAINING_MODE}_test_novels_part_2 ${TRAINING_MODE}_test_novels_part_1
