#!/bin/bash

WINDOW_SIZES=(5 7 10 12 15)

for window in ${WINDOW_SIZES[@]};
    do
    echo 'Currently training on window size: '${window}
    python3 -m scripts.count_model --input_type wiki_${window} --filedir /mnt/cimec-storage-sata/users/andrea.bruera/wikiextractor/wiki_for_bert/ --window_size ${window} --write_to_file
    done
