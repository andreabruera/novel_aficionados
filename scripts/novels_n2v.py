def _test_on_novel(args):
    char_list=get_characters_list(args.dataset)
    books_list=get_books_list(args.dataset)
    for book in books_list:
        with open('novels/{}'.format(book)) as b:
            lines=b.readlines()
            for c in char_list:
                if type(c)!=list:
                   character=str(c)
                elif type(c)==list:
                   character=str(c[0])
                out_file=open('{}.character_vectors'.format(book),'w')
                model=_load_nonce2vec_model(args,character)
                model.vocabulary.nonce=character
                vocab_size=len(model.wv.vocab)
                logger.info('vocab size = {}'.format(vocab_size))
                logger.info('nonce: {}'.format(character))
                del model.wv.index2word[model.wv.vocab[character].index]
                #del wv.vocab[self.nonce]
                for l in lines:
                    line=l.strip('\n').strip('\b').split(' ')
                    if character in line:
                        sentence=[line]
                        logger.info('\n\nsentence: {}\n\n'.format(line))
                        model.build_vocab(sentence, update=True)
                        if not args.sum_only:
                            model.train(sentence, total_examples=model.corpus_count,epochs=model.iter)
                            logger.info('\n\n{}\n\n'.format(model.most_similar(character,topn=10)))
                character_vector=model.wv[character]
                out_file.write('{}\t{}\n'.format(character,character_vector))
