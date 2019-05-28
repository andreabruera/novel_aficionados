#!/bin/bash

TRAINING_MODE='bert_random'
TRIALS=(1 2 3 4 5 6 7 8 9 10)
FOLDER=$1

DOWNLOADED_NOVELS_FOLDER=dataset/novels/hundred_novels_by_6
#DOWNLOADED_NOVELS_FOLDER=novels_by_6
#mkdir ${DOWNLOADED_NOVELS_FOLDER}
#echo 'Downloading books...'
#python3 scripts/get_books/download_books.py ${DOWNLOADED_NOVELS_FOLDER} 
echo ${DOWNLOADED_NOVELS_FOLDER}
for trial in ${TRIALS[@]};
    do
    for SIX_NOVELS in $(ls ${DOWNLOADED_NOVELS_FOLDER});
        do
        TRAINING_FOLDER=${TRAINING_MODE}_test_novels_part_${SIX_NOVELS}/${FOLDER}_${trial}
        mkdir -p ${TRAINING_FOLDER}
        echo 'Created folder: '${TRAINING_FOLDER}
        cp -r ${DOWNLOADED_NOVELS_FOLDER}/${SIX_NOVELS}/* ${TRAINING_FOLDER}/
        for NOVEL_FOLDER in $(ls ${TRAINING_FOLDER});
            do
            ./bash_scripts/prova_${FOLDER}_individuale.sh ${TRAINING_FOLDER} ${NOVEL_FOLDER} &   
            done
        wait
        done
    done

#mkdir ${TRAINING_MODE}_test_novels/${FOLDER}
#cp -ri ${TRAINING_MODE}_test_novels_part_2/${FOLDER}/* ${TRAINING_MODE}_test_novels/${FOLDER}/
#cp -ri ${TRAINING_MODE}_test_novels_part_1/${FOLDER}/* ${TRAINING_MODE}_test_novels/${FOLDER}/
#rm -r ${TRAINING_MODE}_test_novels_part_2 ${TRAINING_MODE}_test_novels_part_1
