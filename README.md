# Fast mapping for learning semantic representations ofindividuals

By deploying different strategies (e.g.  attention, rehearsal) we are certainly ableto do in one shot what we usually call learning somebody’s name, which involves both storing important information precisely about that individual (e.g.  face features, voice, personality traits) and learning a unique name to go with this representation, which can be subsequently used to retrieve that bundle of semantic information when we need it e.g. when recognizing or describing that individual.  Ideally, fast mapping should allow machine learning systems to dothe same, although this would have to be implemented by means of strategies which will be different from those used by humans, and specific to the chosen algorithm. Within this theoretical context, the aim of this work is to apply a fast mapping technique in order to obtain semantic information about individuals in novels - that is, semantic information about characters, by using the novels in which they appear:  since characters are lacking proper counterparts in the real world, their semantic information can only be extracted by means of the relevant texts.By human standards, learning a semantic representation for a character from a novel may not look like a fast mapping task, as novels are quite long.  However,by neural networks standards, novels constitute extremely small data. Therefore, this can be considered a machine learning-specific fast mapping task.

# The tasks

The final goal is to be able to generate a faithful semantic representation for each character within a novel. The representations would have to be clearly distinguishable from each other, reflecting the unicity of each individual character in a novel. Furthermore, and this is a crucial point, these representations would have to faithfully represent the main semantic features of each entity, reflecting the original properties of the individual character. I’ll start from studying the fast mapping system from Herbelot&Baroni 2017, which applies Nonce2Vec, a modified version of the well known algorithm Word2Vec, to individual sentences in order to extract semantic information about unknown words. I’ll modify it and re-implement it for the extraction of semantic information about characters in novels in English. The novels will be taken from https://www.gutenberg.org/, a library of public domain books which contains around 30000 books in English. All the novels available in English will be used for the task. In order to verify the effectiveness of the model at building semantic representations of the characters, for each novel I will use two tests. For the first one (the reference test) I will first split each text in two parts, obtaining all the characters’ representations twice, one for each part. Then, I’ll check if the system is able to correctly match the representations taken from one part of the novel with those coming from the other part - i.e. if the model is able to correctly match (and distinguish) characters. The second one (the fidelity test) will check to what extent each character’s semantic representation extracted from a novel matches the one extracted from the corresponding character’s description on Wikipedia (hence, an informatively rich description) - i.e. if the model is able to build faithful representations of the characters. This will be done, of course, only for the novels that have a Wikipedia page, and a sufficiently detailed one.

## Results on 12 novels:

### Doppelganger test:

| TRAINING MODE | PROPER NOUNS | COMMON NOUNS |
| --- | --- | --- |
| _**Count**_ | 0.471 - **3.25** | 0.601 - 2.0 |
| _**Bert**_ | **0.482** - 3.5 | **0.926** - **1.0** |
| _**Character2Vec**_ | 0.342 - 5.5 | 0.409 - 4.0 |

### Doppelganger test - PROTOTYPE:

| TRAINING MODE | PROPER NOUNS |
| --- | --- |
| _**Count**_ | 0.371 - 6.0 |
| _**Bert**_ | **0.577** - **2.0** |
| _**Character2Vec**_ | 0.353 - 5.5 |
