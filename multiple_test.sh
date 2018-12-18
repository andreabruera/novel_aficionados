#!/bin/bash
set -e
#This script runs the novel preparation pipeline on one novel.

#cd ../

#for i in $(ls 100_books); do
for i in $(ls test_novel); do
#    ORIGINAL_FOLDER=100_books/${i}
    ORIGINAL_FOLDER=test_novel/${i}

    cd ${ORIGINAL_FOLDER}    

    FILENAME=$(ls)
    BOOK_NUMBER=${FILENAME/'.txt'/''}

    echo $FILENAME    

#    cd ../../novel_aficionados
    cd ../../
#    python3 scripts/remove_gutenberg_header.py ${BOOK_NUMBER} ../${ORIGINAL_FOLDER}
    python3 scripts/remove_gutenberg_header.py ${BOOK_NUMBER} ${ORIGINAL_FOLDER}

#    cd ../

    CLEAN_NAME=${BOOK_NUMBER}_clean.txt
    CLEAN_PATH=${ORIGINAL_FOLDER}/${CLEAN_NAME}

    l=($(wc -l ${CLEAN_PATH}))
    lines=${l[0]}
    count=$(expr $lines / 2 + 1)

    echo $count

    split -a 1 -l $count ${CLEAN_PATH} ${CLEAN_PATH}_part_

    mkdir ${ORIGINAL_FOLDER}/booknlp

#    cd book_nlp
    cd ../book_nlp

#    ./runjava novels/BookNLP -doc ../${CLEAN_PATH} -printHTML -p ../${ORIGINAL_FOLDER}/booknlp -tok ../${ORIGINAL_FOLDER}/booknlp/${BOOK_NUMBER}.tokens -f
    ./runjava novels/BookNLP -doc ../novel_aficionados/${CLEAN_PATH} -printHTML -p ../novel_aficionados/${ORIGINAL_FOLDER}/booknlp -tok ../novel_aficionados/${ORIGINAL_FOLDER}/booknlp/${BOOK_NUMBER}.tokens -f

    echo "characters list created"

    cd ../novel_aficionados/

#    python3 scripts/prepare_for_n2v.py ../${ORIGINAL_FOLDER} ${BOOK_NUMBER} ${CLEAN_PATH}
    python3 scripts/prepare_for_n2v.py ${ORIGINAL_FOLDER} ${BOOK_NUMBER} ${CLEAN_PATH}

#    cd ..
    find ${ORIGINAL_FOLDER} -name '*clean*' | xargs rm
    rm -r ${ORIGINAL_FOLDER}/booknlp

#    cd novel_aficionados
    n2v test --on novels --model /mnt/cimec-storage-sata/users/andrea.bruera/wiki_training/data/wiki_w2v_2018_size400_max_final_vocab250000_sg1 --folder /mnt/cimec-storage-sata/users/andrea.bruera/novel_aficionados/${ORIGINAL_FOLDER} --data ${BOOK_NUMBER} --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --simil_out
    
    python3 scripts/get_damn_evaluation.py ${ORIGINAL_FOLDER} ${BOOK_NUMBER}
#    cd ../
done
