import re
import spacy
from tqdm import tqdm # loading bar
from spacy.training.example import Example
import random
from spacy.scorer import Scorer
import argparse



def mapping_label(pattern):
    def remove_suffix(text, suffix):
        return text[:-len(suffix)] if text.endswith(suffix) and len(suffix) != 0 else text

    pattern = remove_suffix(pattern, "deriv")
    pattern = remove_suffix(pattern, "part")
    label_map = {"B-LOC":"LOC", "B-PER":"PER", "B-ORG":"ORG","B-OTH":"MISC", "LOC":"LOC", "PER":"PER", "ORG":"ORG", "MISC":"MISC"}
    
    return label_map.get(pattern)

def remapping_label(pattern):
    label_map = {"LOC":"B-LOC", "PER":"B-PER", "ORG":"B-ORG","MISC":"B-OTH", "B-LOC":"B-LOC", "B-PER":"B-PER", "B-ORG":"B-ORG", "B-OTH":"B-OTH"}
    result = label_map.get(pattern)
    if result == None:
        return pattern
    return result


def get_data(file_name, debug = False, transform = False):
    with open(file_name) as fd:
        # constant
        ITER_DEBUG = debug
        if ITER_DEBUG:
            iter_count = 0
            iter = 1000

        TEXT_EMPTY = False
        START_SENTENCE = False

        sentence = ""
        RE_INT = re.compile(r'^[-+]?([1-9]\d*|0)$')
        entities = []
        result = []
        count = 0
        one_sentence = []
        line_count = 0
        
        for line in fd:
            line_index = line.split('\t')
            if START_SENTENCE:
                one_sentence.append(line)
            if line_index[0] == '#':
                START_SENTENCE = True
                continue
            if line_index[0] == '\n':
                START_SENTENCE = False
               # Processing one line
                for i, elements in enumerate(one_sentence):

                    #remove \n
                    elements = elements.replace('\n','')

                    contents = elements.split("\t")
                    # print(contents)
                    
                    # Start sentence
                    if contents[0] == '#':
                        sentence = ""
                        count = 0
                        
                    # End sentence
                    if contents[0] == '':
                        if len(entities) != 0:
                            result.append((sentence, {'entities':entities}))
                        # print(result)
                        
                        # reset variables
                        sentence = ""
                        entities = []
                        count = 0

                    if RE_INT.match(contents[0]):
                        text = contents[1]
                        pattern = contents[2]
                    
                        if text == '':
                            TEXT_EMPTY = True

                        LAST = (i == len(one_sentence) -1)
                        if text == "," or (LAST and text == "."):
                            sentence = sentence.rstrip(sentence[-1])
                            count = count -1 

                        if text != '' and pattern != 'O':
                            if transform:
                                entities.append((count, count+len(text), mapping_label(pattern) )) #(21, 25, 'PrdName')
                            else:
                                entities.append((count, count+len(text), pattern )) #(21, 25, 'PrdName')

                        sentence += text
                        count += len(text)

                        if not TEXT_EMPTY:
                            sentence += " "
                            count += 1

                            # Reset control variable
                            TEXT_EMPTY = False

                        

            if ITER_DEBUG:
                iter_count += 1
                print(iter_count)
                # print(iter_count, iter)
                if iter_count > iter:
                    return result
            
            if line_count%1000 == 0:
                print(line_count)
            line_count +=1

        return result

def train(data,batch_size = 20, model = None):
    if model != None:
        nlp = model
    else:
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
    loss_result = []
    # print("### Start train loop ###")
    with nlp.disable_pipes(*other_pipes):  # only train NER
        if model == None:
            optimizer = nlp.initialize()
        else:
            optimizer = nlp.resume_training()
        examples = []
        # print("### Start collecting batch ###")
        losses = {}
        for _ in range(batch_size):
            random.shuffle(data)
            for text, annotations in  data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)

        # print("### Start trainning batch ###")
        nlp.update(examples,
        drop=0.2,  # dropout - make it harder to memorise data
        sgd=optimizer,  # callable to update weights
        losses=losses)
        loss_result.append(losses)
        print("losses", losses)
        
        
    return nlp, loss_result

def validate(model, data):
    random.shuffle(data)
    examples = []
    scorer = Scorer()
    for text, annotations in data:
        doc = model.make_doc(text)
        example = Example.from_dict(doc, annotations)
        example.predicted = model(str(example.predicted))
        examples.append(example)
    score = scorer.score(examples)

    return score

    # print("precision", score["ents_p"])
    # print("recall", score["ents_r"])
    # print("f1 score", score["ents_f"])
    
            

def check_ignore(text, entities):
    nlp = spacy.load("de_core_news_sm")
    print("====Checking Ignore ======")
    print(text, entities)
    print(spacy.training.offsets_to_biluo_tags(nlp.make_doc(text),entities))


def train_new_model(train_data, validate_data, epoch = 100 , model_file="de_model_1", batch_size = 20, prdnlp = None):
    loss_accumulate = []
    scores = []
    for i in tqdm(range(epoch)):
        print(f"epoch:{epoch}")
        # Training
        print("### Start training ###")
        if prdnlp == None:
            prdnlp, loss_result = train(train_data, batch_size)
        else:
            prdnlp, loss_result = train(train_data, batch_size, prdnlp)
        
        loss_accumulate.append(loss_result)

        # Validation
        print("### Start validation ###")
        score = validate(prdnlp,validate_data)
        print("f1 score", score["ents_f"])
        scores.append(score)

    # debug
    print("### Writting loss data ###")
    with open(model_file + '_loss.txt', 'a') as fout:
        print(*loss_accumulate, sep="\n", file=fout)

    print("### Writting score data ###")
    with open(model_file + '_score.txt', 'a') as fout:
        print(*scores, sep="\n", file=fout)

    # Save our trained Model
    prdnlp.to_disk(model_file)

    # debug
    print("### Start testing ###")
    #Test your text
    #[('Schartau sagte dem Tagesspiegel vom Freitag, Fischer sei in einer Weise aufgetreten , die alles andere als überzeugend war.',
    #   {'entities': [(0, 7, 'B-PER'), (18, 29, 'B-ORG'), (44, 50, 'B-PER')]})]
    test_text = "Schartau sagte dem Tagesspiegel vom Freitag, Fischer sei in einer Weise aufgetreten , die alles andere als überzeugend war."
    doc = prdnlp(test_text)
    print("\n=========Test result======\n")
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

def search_ents(start_i, ents):
    # print(start_i,ents)
    for ent in ents:
        if ent[0] == start_i:
            return ents[0]
    return None

def write_eval(data, eval_ents, eval_file = "eval.tsv"):
    # init content
    contents = "#\n"
    
    sentences = data[0]
    entities = data[1]["entities"]
    count = 0
    sent_list =  sentences.split()
    for index, sent in enumerate(sent_list):
        contents += str(index)
        contents += '\t'
        contents += sent
        contents += '\t'
        d_ent = search_ents(count,entities)
        if d_ent != None:
            pattern = d_ent[2]
            contents += remapping_label(pattern)
        else:
            contents += 'O'    
        contents += '\t'
        contents += 'O'
        contents += '\t'
        r_ent = search_ents(count,eval_ents)
        if r_ent != None:
            pattern = r_ent[2]
            contents += remapping_label(pattern)
        else:
            contents += 'O'
        contents += '\t'
        contents += 'O'
        contents += '\n'
        count += len(sent)
        count += 1 # space 
    contents += '\n'

    with open(eval_file, 'a') as f:
        f.write(contents)

    return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script')
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    
    DEBUG = args.debug
    RUN_TASK = args.task
    print(f"RUN_TASK:{RUN_TASK}")

    train_data = "NER-de-train.tsv"
    dev_data = "NER-de-dev.tsv"
    evaluate_data = "NER-de-test.tsv"


    if DEBUG:
        data = ('Gleich darauf entwirft er seine Selbstdarstellung " Ecce homo " in enger Auseinandersetzung mit diesem Bild Jesu . ', {'entities': [(52, 56, 'B-OTH'), (57, 61, 'I-OTH'), (108, 112, 'B-PER')]})
        
        # print(get_data(train_data, True))
        
        # # Check alignment
        check_data = ('Erst nach der Zerstörung des Allerheiligenklosters in Freiburg 1678 kehrten die Chorherren nach St. Märgen zurück. ', {'entities': [(29, 50, 'B-LOCpart'), (54, 62, 'B-LOC'), (96, 98, 'B-LOC'), (98, 99, 'I-LOC'), (100, 106, 'I-LOC')]})
        check_ignore(check_data[0],check_data[1]["entities"])
    else:
        if RUN_TASK == 1:
            print("### Start Task 1 ###")
            # Settings
            batch_size = 50
            epoch = 100
            new_model_file = "de_model_3"
            old_model_file = None # None for create new model
            

            print("### Get data ###")
            train_data = get_data(train_data, True)
            validate_data = get_data(dev_data, True)
            
            # debug
            print("### Writting debug data ###")
            with open('debug.txt', 'w') as fout:
                print(*train_data, sep="\n", file=fout)

            # Using GPU 
            spacy.prefer_gpu()

            # Using exist model
            nlp = None
            if old_model_file != None:
                nlp = spacy.load(old_model_file)
            
            train_new_model(train_data, validate_data, epoch = epoch, model_file=new_model_file, batch_size=batch_size, prdnlp=nlp)

        elif RUN_TASK == 2:
            print("### Start Task 2 ###")
            # Settings
            batch_size = 50
            new_model_file = "de_update_1"
            old_model_file = "de_core_news_sm" # Using pretrained model
            epoch = 100

            print("### Get data ###")
            train_data = get_data(train_data, True, transform = True)
            validate_data = get_data(dev_data, True, transform = True)

            # Using GPU 
            spacy.prefer_gpu()

            # Using exist model
            if old_model_file != None:
                nlp = spacy.load(old_model_file)

            train_new_model(train_data,validate_data,epoch=epoch,model_file=new_model_file,batch_size=batch_size, prdnlp=nlp)
        
        elif RUN_TASK == 3:
            print("### Start Task 3 ###")
            evaluate_model_file = "de_update_1"

            # Get Evaluate Model
            nlp = spacy.load(evaluate_model_file)
            
            performance_data = get_data(evaluate_data, True)
            for data in performance_data:
                doc = nlp(data[0])
                eval_ents = []
                for ent in doc.ents:
                    eval_ents.append((ent.start_char, ent.end_char, ent.label_))
                
                write_eval(data,eval_ents, eval_file = "eval.tsv")
            