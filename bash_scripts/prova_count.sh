#!/bin/bash

TRAINING_MODE='count'
FOLDER=$1
WINDOW_SIZES=(7 5 2 10 12)

DOWNLOADED_NOVELS_FOLDER=dataset/novels/hundred_novels_by_6
#DOWNLOADED_NOVELS_FOLDER=novels_by_6
#DOWNLOADED_NOVELS_FOLDER=novel_adv
#mkdir ${DOWNLOADED_NOVELS_FOLDER}
#echo 'Downloading books...'
#python3 scripts/get_books/download_books.py ${DOWNLOADED_NOVELS_FOLDER} 
echo ${DOWNLOADED_NOVELS_FOLDER}
for window_size in ${WINDOW_SIZES[@]};
    do
    for SIX_NOVELS in $(ls ${DOWNLOADED_NOVELS_FOLDER});
        do
        TRAINING_FOLDER=${TRAINING_MODE}_test_novels_part_${SIX_NOVELS}/${FOLDER}_window${window_size}
        mkdir -p ${TRAINING_FOLDER}
        echo 'Created folder: '${TRAINING_FOLDER}
        cp -r ${DOWNLOADED_NOVELS_FOLDER}/${SIX_NOVELS}/* ${TRAINING_FOLDER}/
        for NOVEL_FOLDER in $(ls ${TRAINING_FOLDER});
            do
            ./bash_scripts/prova_${FOLDER}_individuale.sh ${TRAINING_FOLDER} ${NOVEL_FOLDER} ${window_size}  
            done
        wait
        done
    done

#mkdir ${TRAINING_MODE}_test_novels/${FOLDER}
#cp -ri ${TRAINING_MODE}_test_novels_part_2/${FOLDER}/* ${TRAINING_MODE}_test_novels/${FOLDER}/
#cp -ri ${TRAINING_MODE}_test_novels_part_1/${FOLDER}/* ${TRAINING_MODE}_test_novels/${FOLDER}/
#rm -r ${TRAINING_MODE}_test_novels_part_2 ${TRAINING_MODE}_test_novels_part_1
