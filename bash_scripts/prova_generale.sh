#!/bin/bash

TRAINING_MODE=$1

LAMBDA=(50 100)
#LAMBDA=(10 50)
#LAMBDA=(50 70 100)
#WINDOW_DECAY=(1 3)
#SUBSAMPLING_DECAY=(1.1 1.5)
SUBSAMPLING_DECAY=(1.1)
SUBSAMPLING=(1000 10000)
#SUBSAMPLING=(1000)
#ALPHA=(0.5 0.1 0.7) 
ALPHA=(0.1 0.5)
#DOWNLOADED_NOVELS_FOLDER=dataset/novels/hundred_novels_by_6
DOWNLOADED_NOVELS_FOLDER=dataset/novels/hundred_novels_by_3
#mkdir ${DOWNLOADED_NOVELS_FOLDER}
#echo 'Downloading books...'
#python3 scripts/get_books/download_books.py ${DOWNLOADED_NOVELS_FOLDER} 

for alpha in ${ALPHA[@]};
    do 
    for lambda in ${LAMBDA[@]};
        do
        for subsampling_decay in ${SUBSAMPLING_DECAY[@]};
            do 
            for subsampling in ${SUBSAMPLING[@]};
                do
                #FOLDER='Lambda_'${lambda}'_Window_decay_'${window_decay}'_Subsampling_decay_'${subsampling_decay}
                FOLDER=${TRAINING_MODE}'_Lambda_'${lambda}'_Alpha_'${alpha}'_Subsampling_decay_'${subsampling_decay}'_Subsampling_'${subsampling}
                echo 'Created folder:' ${FOLDER}
                for SIX_NOVELS in $(ls ${DOWNLOADED_NOVELS_FOLDER});
                    do
                    TRAINING_FOLDER=${TRAINING_MODE}_test_novels/${TRAINING_MODE}
                    mkdir -p ${TRAINING_FOLDER}
                    for novels in $(ls ${DOWNLOADED_NOVELS_FOLDER}/${SIX_NOVELS});
                        do
                        cp -r ${DOWNLOADED_NOVELS_FOLDER}/${SIX_NOVELS}/${novels} ${TRAINING_FOLDER}/
                        echo 'Starting training'
                        ./bash_scripts/prova_${TRAINING_MODE}_individuale.sh ${TRAINING_FOLDER} ${novels} ${lambda} ${alpha} ${subsampling_decay} ${subsampling} &  
                        done
                    wait
                    done
                wait
                done
            done
        done
    done

#cp -ri ${TRAINING_MODE}_test_novels_part_1/* ${TRAINING_MODE}_test_novels_part_2/
#rm -r ${TRAINING_MODE}_test_novels_part_1
#mv ${TRAINING_MODE}_test_novels_part_2/* ${TRAINING_MODE}_test_novels/
#rm -r ${TRAINING_MODE}_test_novels_part_2
