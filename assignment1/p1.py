import numpy as np
import re
from nltk.corpus import brown
import random
from sklearn import metrics
def remove_comma(match_obj):
    if match_obj.group(2) is not None:
        return match_obj.group(1)+match_obj.group(3)
    
def preprocess(w):
    w=w.lower()
    w=re.sub("n't","",w)
    w=re.sub("ed$","ing",w)
    w= re.sub(r"([a-zA-Z])(,+)([a-zA-Z])", remove_comma, w)
    w= re.sub(r"([0-9]+)([,.]+)([0-9]+)", remove_comma, w)
    w=re.sub("[0-9]+","1",w)
    return w

def tagger(sentence):
    added_begi=False
    added_endi=False
    if(sentence[0]!="begi"):
        sentence.insert(0,"begi")
        added_begi=True
    if sentence[-1]!='.':
        sentence.append('.')
        added_endi=True
    prob = np.zeros((len(tag_dic),1))
    tags_chain = np.full((len(tag_dic), len(sentence) ), 0)
    last_word_tag = tag_dic['end']
    indices=np.zeros(len(tag_dic))
    for i in range(len(sentence)):
        if i == 0:
            prob = transition_num[0][:]
            tags_chain[:,i] = 0
            continue
        if i==1:
            continue
        word = preprocess(sentence[i-1])
        if word in word_dic:
            n=word_dic[word]
        else:
            n=1
        
        lp = emission_num[:,n]
        if np.max(prob)<0.0000001:
            prob=prob/np.max(prob)
        prob=prob*lp
        
       
        if sentence[i] ==".":
            tags_chain[:,i] = indices
            prob=prob*transition_num[:,tag_dic['end']]
            last_word_tag = np.argmax(prob)
            break
        else:
            p_inf=prob.reshape((len(tag_dic),1))*transition_num
            indices=np.argmax(p_inf, axis = 0)
            prob = p_inf[indices,np.arange(len(tag_dic))]
            tags_chain[:,i] = indices
    tagged_sentence = [[word,""] for word in sentence]
    iter = len(sentence) - 1
    cpy=last_word_tag+1
    while iter >=0 :
        if iter == len(sentence) - 1:
            tag_index = tag_dic['end']
        else:
            tag_index = last_word_tag
            last_word_tag = tags_chain[last_word_tag, iter]
        tagged_sentence[iter][1] = rev_dic[tag_index]
        iter -= 1
    tagged_sentence[len(sentence)-2][1] = rev_dic[cpy-1]
    if added_begi:
        tagged_sentence=tagged_sentence[1:]
    else:
        tagged_sentence[0][1]='begi'
    if added_endi:
        tagged_sentence=tagged_sentence[:-1]
    else:
        tagged_sentence[-1][1]='.'
    return tagged_sentence

def test_model(sentences):
    predicted=[]
    true=[]
    for sentence in sentences:
        sentence_words=[]
        for i in range(len(sentence)):
            sentence_words.append(sentence[i][0])
        tagged=tagger(sentence_words)
        for i in range(len(sentence)):
            true.append(sentence[i][1])
            predicted.append(tagged[i][1])
            labels=[]
            # if tagged[i][1]=='VERB' and sentence[i][1]=='NOUN':
            #     print(tagged[i],sentence[i])
    for i in range(len(rev_dic)):
        labels.append(rev_dic[i])
    cm = metrics.confusion_matrix(true,predicted,labels=labels)
    print(labels)
    return cm
    
    
word_tag=brown.tagged_words(tagset='universal')
sentences=brown.tagged_sents(tagset='universal')
words_tags=np.array(word_tag)

p=np.vectorize(preprocess)
words=p(words_tags[:,0])
unique_words, counts_words = np.unique(words, return_counts=True)
bool_words=counts_words>1
unique_words=unique_words[bool_words]
word_dic={}
for A, B in zip(unique_words, np.arange(unique_words.size)):
    word_dic[A] = B+2
word_dic["begi"]=0
word_dic["unk"]=1
no_words=len(word_dic)

unique_tags=np.unique(words_tags[:,1])
tag_dic ={ }
rev_dic={}
for A, B in zip(unique_tags, np.arange(unique_tags.size)):
    tag_dic[A] = B+2
    rev_dic[B+2]=A
tag_dic["begi"]=0
rev_dic[0]="begi"
tag_dic["end"]=1
rev_dic[1]="end"
no_tags=len(tag_dic)

sentences=list(sentences)
random.shuffle(sentences)
for i in sentences:
    i.insert(0,('begi','begi'))
    if i[-1][0]!='.':
        i.append(('.','.'))

transition_num=np.zeros((no_tags,no_tags))
emission_num=np.zeros((no_tags,no_words))

for sentence in sentences:
    n=len(sentence)
    prev_tag=tag_dic[sentence[0][1]]
    prev_word=word_dic[preprocess(sentence[0][0])]
    for x in range(1,n-1):
        curr_tag=tag_dic[sentence[x][1]]
        proc_word=preprocess(sentence[x][0])
        if proc_word in word_dic.keys():
            curr_word=word_dic[proc_word]
        else:
            curr_word=1
        transition_num[prev_tag][curr_tag]+=1
        emission_num[prev_tag][prev_word]+=1
        prev_tag=curr_tag
        prev_word=curr_word
    curr_tag=tag_dic['end']
    curr_word=word_dic['.']
    transition_num[prev_tag][curr_tag]+=1
    emission_num[prev_tag][prev_word]+=1

transition_num[tag_dic['end']][tag_dic['end']]=1
transition_num=np.divide(transition_num,np.sum(transition_num,axis=1))
transition_num[tag_dic['end']][tag_dic['end']]=0
emission_num[tag_dic['end']][word_dic['.']]=1
emission_num=np.divide(emission_num,np.sum(emission_num,axis=1,keepdims=True))
emission_num[tag_dic['end']][word_dic['.']]=0
print(emission_num[0,:])
for i in range(len(rev_dic)):
    print(rev_dic[i],end=" ")
print()

cm=test_model(sentences)
np.set_printoptions(linewidth=5000)
print(cm)
print(np.trace(cm)/np.sum(cm))

# sentencei2=[]
# j=8197
# n=len(sentences[j])
# print(n)
# se=sentences[j]
# for i in range(130):
#     sentencei2.append(se[i][0])
    

# sentence_o=tagger(sentencei2)
# print(sentence_o)
# print()