#!/bin/bash

#LAMBDA=(1 10 50 100)
#LAMBDA=(50 70 100)
LAMBDA=(100)
WINDOW_DECAY=(1)
SUBSAMPLING_DECAY=(1.1)
SUBSAMPLING=(1000)
DOWNLOADED_NOVELS_FOLDER=novels_by_6
#mkdir ${DOWNLOADED_NOVELS_FOLDER}
#echo 'Downloading books...'
#python3 scripts/get_books/download_books.py ${DOWNLOADED_NOVELS_FOLDER} 

#mkdir prototype_test_novels
for lambda in ${LAMBDA[@]};
    do 
    for window_decay in ${WINDOW_DECAY[@]};
        do
        for subsampling_decay in ${SUBSAMPLING_DECAY[@]};
            do 
            for subsampling in ${SUBSAMPLING[@]};
                do
                #FOLDER='Lambda_'${lambda}'_Window_decay_'${window_decay}'_Subsampling_decay_'${subsampling_decay}
                FOLDER='Proto_Lambda_'${lambda}'_Window_decay_'${window_decay}'_Subsampling_decay_'${subsampling_decay}'_Subsampling_'${subsampling}
                echo 'Created folder:' ${FOLDER}
                for SIX_NOVELS in $(ls ${DOWNLOADED_NOVELS_FOLDER});
                    do
                    TRAINING_FOLDER=prototype_test_novels_part_${SIX_NOVELS}/${FOLDER}
                    mkdir -p ${TRAINING_FOLDER}
                    cp -r ${DOWNLOADED_NOVELS_FOLDER}/${SIX_NOVELS}/* ${TRAINING_FOLDER}/
                    echo 'Starting training'
                    for NOVEL_FOLDER in $(ls ${TRAINING_FOLDER});
                        do
                        ./prova_prototype_individuale.sh ${TRAINING_FOLDER} ${NOVEL_FOLDER} ${lambda} ${window_decay} ${subsampling_decay} ${subsampling} & 
                        done
                    wait
                    done
                wait
                done
            done
        done
    done
