#!/bin/bash
set -x
#This script runs the novel preparation pipeline on one novel.

NAME=$1

python3 scripts/remove_gutenberg_header.py ${NAME} 

CLEAN_NAME=${NAME}_clean.txt
CLEAN_PATH=novels/${CLEAN_NAME}

l=($(wc -l ${CLEAN_PATH}))
lines=${l[0]}
count=$(expr $lines / 2 + 1)

echo $count

split -a 1 -l $count ${CLEAN_PATH} ${CLEAN_PATH}_part_

cd ../book_nlp

./runjava novels/BookNLP -doc ../novel_aficionados/${CLEAN_PATH} -printHTML -p ../novel_aficionados/novels/${NAME} -tok data/tokens/${CLEAN_PATH}.tokens -f

echo "characters list created"

cd ../novel_aficionados/


python3 scripts/characters_list.py ${NAME}
python3 scripts/prepare_for_n2v.py ${NAME}
