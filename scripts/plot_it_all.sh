#!/bin/bash

PLOT_FOLDERS=plot_folders
for SETUP in $(ls ${PLOT_FOLDERS});
    do 
    for NOVEL in $(ls ${PLOT_FOLDERS}/${SETUP});
        do 
        PATH_TO_FILE=${PLOT_FOLDERS}/${SETUP}/${NOVEL}/original_novel
        FILE_NAME=$(ls ${PATH_TO_FILE})
        FILE_NUMBER=${FILE_NAME/'.txt'/''}
        python3 scripts/get_damn_evaluation.py ${PLOT_FOLDERS}/${SETUP}/${NOVEL} ${FILE_NUMBER} 
        done
    done

