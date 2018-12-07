#!/bin/bash
set -x
#This script runs the novel preparation pipeline on one novel.

ORIGINAL_FOLDER=$1
BOOK_NUMBER=$2
FILENAME=${BOOK_NUMBER}.txt

python3 scripts/remove_gutenberg_header.py ${BOOK_NUMBER} ${ORIGINAL_FOLDER} 

CLEAN_NAME=${BOOK_NUMBER}_clean.txt
CLEAN_PATH=${ORIGINAL_FOLDER}/${CLEAN_NAME}

l=($(wc -l ${CLEAN_PATH}))
lines=${l[0]}
count=$(expr $lines / 2 + 1)

echo $count

split -a 1 -l $count ${CLEAN_PATH} ${CLEAN_PATH}_part_

mkdir ${ORIGINAL_FOLDER}/booknlp
cd ../book_nlp

./runjava novels/BookNLP -doc ../novel_aficionados/${CLEAN_PATH} -printHTML -p ../novel_aficionados/${ORIGINAL_FOLDER}/booknlp -tok ../novel_aficionados/${ORIGINAL_FOLDER}/booknlp/${BOOK_NUMBER}.tokens -f

echo "characters list created"

cd ../novel_aficionados/

python3 scripts/characters_list_from_booknlp.py ${ORIGINAL_FOLDER} ${BOOK_NUMBER}
python3 scripts/prepare_for_n2v.py ${ORIGINAL_FOLDER} ${BOOK_NUMBER} ${CLEAN_PATH}

find ${ORIGINAL_FOLDER} -name '*clean*' | xargs rm
rm -r ${ORIGINAL_FOLDER}/booknlp 
