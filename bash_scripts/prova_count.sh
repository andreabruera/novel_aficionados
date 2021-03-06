#!/bin/bash

TRAINING_MODE=$1
#FOLDER=$1
WINDOW_SIZES=(7 5 2)

DOWNLOADED_NOVELS_FOLDER=dataset/novels/hundred_novels_by_2_wiki
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
        TRAINING_FOLDER=${TRAINING_MODE}_test_novels/${TRAINING_MODE}_window_${window_size}
        mkdir -p ${TRAINING_FOLDER}
        echo 'Created folder: '${TRAINING_FOLDER}
        for novel in $(ls ${DOWNLOADED_NOVELS_FOLDER}/${SIX_NOVELS});
            do
            cp -r ${DOWNLOADED_NOVELS_FOLDER}/${SIX_NOVELS}/${novel} ${TRAINING_FOLDER}/
            ./bash_scripts/prova_${TRAINING_MODE}_individuale.sh ${TRAINING_FOLDER} ${novel} ${window_size} &  
            done
        wait
        done
    done

#mkdir ${TRAINING_MODE}_test_novels/${FOLDER}
#cp -ri ${TRAINING_MODE}_test_novels_part_2/${FOLDER}/* ${TRAINING_MODE}_test_novels/${FOLDER}/
#cp -ri ${TRAINING_MODE}_test_novels_part_1/${FOLDER}/* ${TRAINING_MODE}_test_novels/${FOLDER}/
#rm -r ${TRAINING_MODE}_test_novels_part_2 ${TRAINING_MODE}_test_novels_part_1
