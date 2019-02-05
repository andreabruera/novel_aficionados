#!/bin/bash

for i in $(ls test_novel); do
    FILENAME=$(ls test_novel/${1}/original_novel/)
    BOOK_NUMBER=${FILENAME/'.txt'/''} 
    sh single_test.sh test_novel/${i}/original_novel ${BOOK_NUMBER} &
done
