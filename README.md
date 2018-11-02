<!DOCTYPE html>
<html>
<strong>Fast Mapping for Novel Aficionados <br><em>learning semantic representations of individuals from limited linguistic data</em></strong><br><br>
<div>
  <p>The final goal is to be able to generate a faithful semantic representation for each character within a novel. The representations would have to be clearly distinguishable from each other, reflecting the unicity of each individual character in a novel. Furthermore, and this is a crucial point, these representations would have to faithfully represent the main semantic features of each entity, reflecting the original properties of the individual character.</p>
</div>

<div>
  <p>I’ll start from studying the fast mapping system from Baroni & Herbelot 2017, which applies Nonce2Vec, a modified version of the well known algorithm Word2Vec (Mikolov et al. 2013), to individual sentences in order to extract semantic information about unknown words. I’ll modify it and re-implement it for the extraction of semantic information about characters in novels in English. The novels will be taken from https://www.gutenberg.org/, a library of public domain books which contains around 30000 books in English. All the novels available in English will be used for the task. </p>
</div>
  <p>In order to verify the effectiveness of the model at building semantic representations of the characters, for each novel I will use two tests.</p>
  <ol>
  <li>For the first one (the reference test) I will first split each text in two parts, obtaining all the characters’ representations twice, one for each part. Then, I’ll check if the system is able to correctly match the representations taken from one part of the novel with those coming from the other part - i.e. if the model is able to correctly match (and distinguish) characters. </li>
    <li>The second one (the fidelity test) will check to what extent each character’s semantic representation extracted from a novel matches the one extracted from the corresponding character’s description on Wikipedia (hence, an informatively rich description) - i.e. if the model is able to build faithful representations of the characters. This will be done, of course, only for the novels that have a Wikipedia page, and a sufficiently detailed one.</li>
  </ol>
</html>
