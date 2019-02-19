#!/bin/bash

#LAMBDA=(1 10 50 100)
#LAMBDA=(50 70 100)
LAMBDA=(150 70 100)
WINDOW_DECAY=(0 1 3 5)
SUBSAMPLING_DECAY=(1 '1.1' '1.5' 2)
SUBSAMPLING=(10000 5000 1000)
DOWNLOADED_NOVELS_FOLDER=novels
#mkdir ${DOWNLOADED_NOVELS_FOLDER}
#echo 'Downloading books...'
#python3 scripts/get_books/download_books.py ${DOWNLOADED_NOVELS_FOLDER} 

mkdir big_test_novels
for lambda in ${LAMBDA[@]};
    do 
    for window_decay in ${WINDOW_DECAY[@]};
        do
        for subsampling_decay in ${SUBSAMPLING_DECAY[@]};
            do 
            for subsampling in ${SUBSAMPLING[@]};
                do
                #FOLDER='Lambda_'${lambda}'_Window_decay_'${window_decay}'_Subsampling_decay_'${subsampling_decay}
                FOLDER='Lambda_'${lambda}'_Window_decay_'${window_decay}'_Subsampling_decay_'${subsampling_decay}'_Subsampling_'${subsampling}
                echo 'Created folder:' ${FOLDER}
                TRAINING_FOLDER=big_test_novels/${FOLDER}
                mkdir ${TRAINING_FOLDER}
                cp -r ${DOWNLOADED_NOVELS_FOLDER}/* ${TRAINING_FOLDER}/
                echo 'Starting training'
                for NOVEL_FOLDER in $(ls ${TRAINING_FOLDER});
                    do
                    ./prova_ufficiale_individuale.sh ${TRAINING_FOLDER} ${NOVEL_FOLDER} ${lambda} ${window_decay} ${subsampling_decay} ${subsampling} &
                    done
                wait
                done
            done
        done
    done
