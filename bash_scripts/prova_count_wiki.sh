#!/bin/bash

#TRAINING_MODE='bert'
#FOLDER=$1
TRAINING_MODE=$1
NUMBER_NOVELS_AT_A_TIME=$2
WINDOW_SIZE=(7 5 2)

DOWNLOADED_NOVELS_FOLDER=dataset/novels/hundred_novels_by_${NUMBER_NOVELS_AT_A_TIME}_wiki
#DOWNLOADED_NOVELS_FOLDER=novels_by_6
#mkdir ${DOWNLOADED_NOVELS_FOLDER}
#echo 'Downloading books...'
#python3 scripts/get_books/download_books.py ${DOWNLOADED_NOVELS_FOLDER} 
echo ${DOWNLOADED_NOVELS_FOLDER}
for size in ${WINDOW_SIZE[@]};
    do
    for SIX_NOVELS in $(ls ${DOWNLOADED_NOVELS_FOLDER});
        do
        TRAINING_FOLDER=${TRAINING_MODE}_test_novels/${TRAINING_MODE}_window_${size}
        #mkdir -p ${TRAINING_FOLDER}
        #echo 'Created folder: '${TRAINING_FOLDER}
        for novel in $(ls ${DOWNLOADED_NOVELS_FOLDER}/${SIX_NOVELS});
            do
            cp -r ${DOWNLOADED_NOVELS_FOLDER}/${SIX_NOVELS}/${novel}/original_wikipedia_page ${TRAINING_FOLDER}/${novel}
            ./bash_scripts/prova_${TRAINING_MODE}_wiki_individuale.sh ${TRAINING_FOLDER} ${novel} ${size}   
            done
        wait
        done
    done

#mkdir ${TRAINING_MODE}_test_novels/${FOLDER}
#cp -ri ${TRAINING_MODE}_test_novels_part_2/${FOLDER}/* ${TRAINING_MODE}_test_novels/${FOLDER}/
#cp -ri ${TRAINING_MODE}_test_novels_part_1/${FOLDER}/* ${TRAINING_MODE}_test_novels/${FOLDER}/
#rm -r ${TRAINING_MODE}_test_novels_part_2 ${TRAINING_MODE}_test_novels_part_1
