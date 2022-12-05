# Read data
import csv
import re

def get_clean_data(file_name, ignore=False, debug = False):
    DEBUG_ITER = debug
    ITER = 30
    DEBUG_ONCE = False
    DEBUG_FLAG = False
    DEBUG_RANGE = debug
    RANGE_COUNT = 0
    RANGE_START = 88161

    G_RANGE = 2
    if ignore:
        G_RANGE = 1
    
    
    clean_data = []
    with open(train_data) as fd:
        
            
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        sentence = ""
        RE_INT = re.compile(r'^[-+]?([1-9]\d*|0)$')
        entities = []
        
        if DEBUG_ITER:
            debug = 0
            
        count = 0
        START_SENTENCE = False
        END_SENTENCE = False
        TEXT_EMPTY = False
        ROW_EMPTY = False
        HAVE_SUBSTRING = False

        for row in rd:
            if DEBUG_RANGE:
                if RANGE_START <= RANGE_COUNT:
                    pass
                else:
                    RANGE_COUNT += 1
                    continue
            if DEBUG_ITER:
                print(row)
                if debug > ITER:
                    break
            

            #end of parsing
            if len(row) == 0:
                # output one sentence result
                clean_data.append((sentence, {'entities':entities}))
                if DEBUG_ITER:
                    print(clean_data)

                # reset variables
                sentence = ""
                entities = []
                count = 0
                if DEBUG_ONCE:
                    break
                continue

            # start parsing
            elif row[0] == '#':
                sentence = ""
                START_SENTENCE = True
            # middle of sentence
            elif RE_INT.match(row[0]):
                START_SENTENCE = False
                HAVE_SUBSTRING = False
                # Have substring
                if '\t' in row[1]:
                    NEW_ROW_START = False
                    sub_string = re.split('\n',row[1])
                    if DEBUG_ITER:
                        print(f"sub_string:{sub_string}")
                    for _, sub in enumerate(sub_string):
                        elements = re.split('\t', sub)
                        if len(elements) == 0:
                            if NEW_ROW_START:
                                # output one sentence result
                                clean_data.append((sentence, {'entities':entities}))

                                # reset variables
                                sentence = ""
                                entities = []
                                count = 0
                                NEW_ROW_START = False
                                END_SENTENCE = True
                        if DEBUG_ITER:     
                            print(f"elements:{elements}")
                        for index, e in enumerate(elements):
                            if e == '':
                                ROW_EMPTY = True
                            if e == '#':
                                NEW_ROW_START = True
                                START_SENTENE = True
                                count = 0
                                if sentence != "" and len(entities) != 0:
                                    # output one sentence result
                                    clean_data.append((sentence, {'entities':entities}))

                                    # reset variables
                                    sentence = ""
                                    entities = []
                            
                            if RE_INT.match(e):
                                ROW_EMPTY = False
                                HAVE_SUBSTRING = True
                                text = elements[index+1]
                                if text == "":
                                    TEXT_EMPTY = True
                                else:
                                    TEXT_EMPTY = False
                                sentence += text
                                

                                for i in range(G_RANGE):
                                    if index + i + 2 > len(elements) - 1:
                                        continue
                                    pattern = elements[index+i+2]
                                    if pattern != 'O':
                                        entities.append((count, count+len(text)-1, pattern)) #(21, 25, 'PrdName')
                                if count == 0:
                                    count += len(text) -1
                                else:
                                    count += len(text)
                            if DEBUG_ITER and DEBUG_FLAG:
                                print(f"START_SENTENCE:{START_SENTENCE}")
                                print(f"END_SENTENCE:{END_SENTENCE}")
                                print(f"TEXT_EMPTY:{TEXT_EMPTY}")
                                print(f"ROW_EMPTY:{ROW_EMPTY}")
                        if not START_SENTENCE and not END_SENTENCE and not TEXT_EMPTY and not ROW_EMPTY:
                            # Add space count
                            sentence += " "
                            count += 1
                            if DEBUG_ITER:
                                debug +=1
                            START_SENTENCE = False
                            END_SENTENCE = False
                            TEXT_EMPTY = False
                            ROW_EMPTY = False
                
                if not HAVE_SUBSTRING:
                    # add entity
                    for i in range(G_RANGE):
                        if i+2 > len(row) -1:
                            continue
                        if row[i+2] != 'O':
                            entities.append((count, count+len(row[1])-1, row[i+2])) #(21, 25, 'PrdName')

                    
                    # iterate the count and save sentence
                    sentence += row[1]
                    if count == 0:
                        count += len(row[1]) -1
                    else:
                        count += len(row[1])

            if DEBUG_ITER and DEBUG_FLAG:
                print(f"START_SENTENCE:{START_SENTENCE}")
                print(f"END_SENTENCE:{END_SENTENCE}")
                print(f"HAVE_SUBSTRING:{HAVE_SUBSTRING}")
            
            if not START_SENTENCE and not END_SENTENCE and not HAVE_SUBSTRING:
                # Add space count
                sentence += " "
                count += 1
                if DEBUG_ITER:
                    debug +=1
                START_SENTENCE = False
                END_SENTENCE = False
                HAVE_SUBSTRING = False

            
        return clean_data

# Test 
train_data = "NER-de-train.tsv"
dev_data = "NER-de-dev.tsv"
# debug
with open('test.txt', 'w') as fout:
    data = get_clean_data(train_data, ignore = True)
    print(*data, sep="\n", file=fout)
# get_clean_data(train_data, ignore = True)

# Example for training (WIP)
import spacy
from tqdm import tqdm # loading bar
from spacy.training.example import Example
import random

TRAIN_DATA = get_clean_data(train_data,ignore=True)

def train_spacy(data,iterations):
    nlp = spacy.blank('de')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe('ner', last=True)
    # add labels
    for _, annotations in data:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(data)
            losses = {}
            for text, annotations in  tqdm(data):
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example],
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print("losses", losses)
    return nlp


TRAIN_DATA = get_clean_data(train_data,ignore=True)

# Using GPU 
spacy.prefer_gpu()

# Training
prdnlp = train_spacy(TRAIN_DATA, 20)

# Save our trained Model
modelfile = "spacy_prdName"
prdnlp.to_disk(modelfile)

#Test your text
test_text = "what is the price of chair?"
doc = prdnlp(test_text)
print("\n=========Test resultt======\n")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
