path_to_dir = path to directory containing train and dev. 
THE DIR MUST CONTAIN BOTH train AND dev FILES with this exact name. 
IF THERE's NO DEV PLEASE COPY THE train FILE AND RENAME ONE COPY TO dev (to avoid run error)
name_prefix = prefix to output dir
path_to_word_vecs = path to wordVectors.txt
path_to_word_vocab = path to vocab.txt

##### training (train and dev) ##### 
### PART 1 ###
python tagger1.py path_do_dir name_prefix 

example to run:
python tagger1.py pos/ pos
python tagger1.py ner/ ner

the outputs containing the (1) model, (2) idx_to_tag, (3) word_to_idx and (4) the graphs 
will be in the following dirs:
pos_output
ner_output

### PART 2 ###
python top_k wordVectors.txt vocab.txt

### PART 3 ###
!! using tagger1.py as suggested !! 
python tagger1.py path_do_dir name_prefix path_to_word_vecs path_to_word_vocab 

example to run:
python tagger1.py pos/ pos wordVectors.txt vocab.txt
python tagger1.py ner/ ner wordVectors.txt vocab.txt

the outputs will be in the following dirs:
pos_output_pre_trained
ner_output_pre_trained

### PART 4 ###
run without pre trained-
python tagger3.py path_do_dir name_prefix 
run with pre trained -
python tagger3.py path_do_dir name_prefix path_to_word_vecs path_to_word_vocab

the outputs will be in the following dirs:
pos_output_prefix_and_suffix
ner_output_prefix_and_suffix
pos_output_prefix_and_suffix_pre_trained
ner_output_prefix_and_suffix_pre_trained

##### test - part 1 and 3 ##### 
MAKE SURE THAT BATCH DIM IS SIMILAR TO MODEL
python test1.py path_to_test path_to_model path_to_word_to_idx path_to_idx_to_tag output_file

##### test - part 4 ##### 
python test3.py path_to_test path_to_model path_to_word_to_idx path_to_idx_to_tag path_to_pref_to_idx path_to_suff_to_idx output_file
