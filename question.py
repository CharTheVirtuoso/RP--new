# import warnings
# warnings.filterwarnings("ignore")
# import torch
# import random
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# from sense2vec import Sense2Vec
# from sentence_transformers import SentenceTransformer
# from textwrap3 import wrap
# import random
# from nltk.corpus import wordnet as wn
# import numpy as np
# import nltk
# nltk.download('punkt')
# nltk.download('brown')
# nltk.download('wordnet')
# from nltk.corpus import wordnet as wn
# from nltk.tokenize import sent_tokenize
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# import textdistance
# import string
# import pke
# import traceback
# from flashtext import KeywordProcessor
# from collections import OrderedDict
# from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('omw-1.4')
# import pickle
# import time
# import os 

# # Load Sense2Vec model
# s2v = Sense2Vec().from_disk("C:\\Users\\DELL\\Desktop\\segmentation using speech-text\\s2v_reddit_2015_md\\s2v_old")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load or download T5 models and tokenizers
# def load_model_and_tokenizer(model_name, tokenizer_name, model_file, tokenizer_file):
#     if os.path.exists(model_file):
#         with open(model_file, 'rb') as f:
#             model = pickle.load(f)
#         print(f"{model_name} model found on disk, model loaded successfully.")
#     else:
#         print(f"{model_name} model does not exist in the path specified, downloading the model from the web....")
#         start_time = time.time()
#         model = T5ForConditionalGeneration.from_pretrained(model_name)
#         end_time = time.time()
#         print(f"Downloaded the {model_name} model in {(end_time - start_time) / 60} min, now saving it to disk...")
#         with open(model_file, 'wb') as f:
#             pickle.dump(model, f)
#         print("Done. Saved the model to disk.")

#     if os.path.exists(tokenizer_file):
#         with open(tokenizer_file, 'rb') as f:
#             tokenizer = pickle.load(f)
#         print(f"{tokenizer_name} tokenizer found on disk and is loaded successfully.")
#     else:
#         print(f"{tokenizer_name} tokenizer does not exist in the path specified, downloading the tokenizer from the web....")
#         start_time = time.time()
#         tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
#         end_time = time.time()
#         print(f"Downloaded the {tokenizer_name} tokenizer in {(end_time - start_time) / 60} min, now saving it to disk...")
#         with open(tokenizer_file, 'wb') as f:
#             pickle.dump(tokenizer, f)
#         print("Done. Saved the tokenizer to disk.")

#     return model.to(device), tokenizer

# summary_model, summary_tokenizer = load_model_and_tokenizer('t5-base', 't5-base', 't5_summary_model.pkl', 't5_summary_tokenizer.pkl')
# question_model, question_tokenizer = load_model_and_tokenizer('ramsrigouthamg/t5_squad_v1', 'ramsrigouthamg/t5_squad_v1', 't5_question_model.pkl', 't5_question_tokenizer.pkl')

# # Load Sentence Transformer model
# if os.path.exists("sentence_transformer_model.pkl"):
#     with open("sentence_transformer_model.pkl", 'rb') as f:
#         sentence_transformer_model = pickle.load(f)
#     print("Sentence transformer model found on disk, model loaded successfully.")
# else:
#     print("Sentence transformer model does not exist in the path specified, downloading the model from the web....")
#     start_time = time.time()
#     sentence_transformer_model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v2")
#     end_time = time.time()
#     print(f"Downloaded the sentence transformer in {(end_time - start_time) / 60} min, now saving it to disk...")
#     with open("sentence_transformer_model.pkl", 'wb') as f:
#         pickle.dump(sentence_transformer_model, f)
#     print("Done saving to disk.")

# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def postprocesstext(content):
#     final = ""
#     for sent in sent_tokenize(content):
#         sent = sent.capitalize()
#         final = final + " " + sent
#     return final

# def summarizer(text, model, tokenizer):
#     text = text.strip().replace("\n", " ")
#     text = "summarize: " + text
#     max_len = 512
#     encoding = tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
#     input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
#     outs = model.generate(input_ids=input_ids, attention_mask=attention_mask, early_stopping=True, num_beams=3, num_return_sequences=1, no_repeat_ngram_size=2, min_length=75, max_length=300)
#     dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
#     summary = dec[0]
#     summary = postprocesstext(summary)
#     summary = summary.strip()
#     return summary

# def get_nouns_multipartite(content):
#     out = []
#     try:
#         extractor = pke.unsupervised.MultipartiteRank()
#         extractor.load_document(input=content, language='en')
#         pos = {'PROPN', 'NOUN', 'ADJ', 'VERB', 'ADP', 'ADV', 'DET', 'CONJ', 'NUM', 'PRON', 'X'}
#         stoplist = list(string.punctuation)
#         stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
#         stoplist += stopwords.words('english')
#         extractor.candidate_selection(pos=pos)
#         extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
#         keyphrases = extractor.get_n_best(n=15)
#         for val in keyphrases:
#             out.append(val[0])
#     except:
#         out = []
#     return out

# def get_keywords(originaltext):
#     keywords = get_nouns_multipartite(originaltext)
#     return keywords

# def get_question(context, answer, model, tokenizer):
#     text = "context: {} answer: {}".format(context, answer)
#     encoding = tokenizer.encode_plus(text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
#     input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
#     outs = model.generate(input_ids=input_ids, attention_mask=attention_mask, early_stopping=True, num_beams=5, num_return_sequences=1, no_repeat_ngram_size=2, max_length=72)
#     dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
#     Question = dec[0].replace("question:", "").strip()
#     return Question

# def filter_same_sense_words(original, wordlist):
#     filtered_words = []
#     base_sense = original.split('|')[1]
#     for eachword in wordlist:
#         if eachword[0].split('|')[1] == base_sense:
#             filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
#     return filtered_words

# # def get_highest_similarity_score(wordlist, wrd):
# #     score = []
# #     for each in wordlist:
# #         score.append(NormalizedLevenshtein().similarity(each.lower(), wrd.lower()))
# #     return max(score)
# def get_highest_similarity_score(wordlist, wrd):
#     """
#     This function takes the given word along with the wordlist and then gives out the max-score
#     which is the Levenshtein distance for the wrong answers
#     because we need the options which are very different from one another but relating to the same context.
#     """
#     score = []
#     for each in wordlist:
#         score.append(textdistance.levenshtein.similarity(each.lower(), wrd.lower()))
#     return max(score)


# def sense2vec_get_words(word, s2v, topn, question):
#     output = []
#     try:
#         sense = s2v.get_best_sense(word, senses=["NOUN", "PERSON", "PRODUCT", "LOC", "ORG", "EVENT", "NORP", "WORK OF ART", "FAC", "GPE", "NUM", "FACILITY"])
#         if sense is None:
#             print(f"No sense found for '{word}'")
#             return []
        
#         print(f"Sense for '{word}':", sense)  # Debugging output
        
#         most_similar = s2v.most_similar(sense, n=topn)
#         if not most_similar:
#             print(f"No similar words found for sense '{sense}'")
#             return []
        
#         print(f"Most similar words for '{word}':", most_similar)  # Debugging output
        
#         output = filter_same_sense_words(sense, most_similar)
#     except Exception as e:
#         print(f"Error in sense2vec_get_words: {e}")
#         output = []

#     threshold = 0.6
#     final = [word]
#     checklist = question.split()
#     for x in output:
#         if get_highest_similarity_score(final, x) < threshold and x not in final and x not in checklist:
#             final.append(x)

#     return final[1:]


# def get_keywords(originaltext):
#     keywords = get_nouns_multipartite(originaltext)
#     print("Extracted keywords:", keywords)  # Debugging output
#     return keywords



# def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
#     word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
#     word_similarity = cosine_similarity(word_embeddings)
#     keywords_idx = [np.argmax(word_doc_similarity)]
#     candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

#     for _ in range(top_n - 1):
#         candidate_similarities = word_doc_similarity[candidates_idx, :]
#         target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
#         mmr = (lambda_param) * candidate_similarities - (1 - lambda_param) * target_similarities.reshape(-1, 1)
#         mmr_idx = candidates_idx[np.argmax(mmr)]
#         keywords_idx.append(mmr_idx)
#         candidates_idx.remove(mmr_idx)

#     return [words[idx] for idx in keywords_idx]

# def get_similar_words_wordnet(word, topn):
#     output = []
#     synsets = wn.synsets(word)
#     if not synsets:
#         return output
    
#     for syn in synsets:
#         for lemma in syn.lemmas():
#             related_words = lemma.derivationally_related_forms()
#             for related in related_words:
#                 word_form = related.name().replace("_", " ").title()
#                 if word_form not in output:
#                     output.append(word_form)
#                 if len(output) >= topn:
#                     return output
#     return output

# def get_distractors_wordnet(word, topn, question):
#     output = []
#     try:
#         synsets = wn.synsets(word)
#         for syn in synsets:
#             for lemma in syn.lemmas():
#                 related_words = lemma.derivationally_related_forms()
#                 for related_form in related_words:
#                     output.append(related_form.name().replace("_", " ").title())
#     except Exception as e:
#         print(f"Error in get_distractors_wordnet: {e}")
#     return output


# def get_mca_questions(text):
#     try:
#         summary = summarizer(text, summary_model, summary_tokenizer)
#         keywords = get_keywords(summary)
#         context = summary
#         questions = []
#         seen_questions = set()

#         for keyword in keywords:
#             if len(questions) >= 5:
#                 break  # Stop if we already have 5 questions
            
#             try:
#                 keyword_list = sense2vec_get_words(keyword, s2v, topn=10, question=summary)
#                 if not keyword_list:
#                     keyword_list = get_similar_words_wordnet(keyword, topn=10)  # Fallback
                
#                 if keyword_list:
#                     answers = get_distractors_wordnet(keyword, topn=5, question=summary)
#                     if answers:
#                         for answer in answers:
#                             if len(questions) >= 5:
#                                 break  # Stop if we already have 5 questions

#                             try:
#                                 question = get_question(context, answer, question_model, question_tokenizer)
#                                 if question:
#                                     # Generate unique question
#                                     if question not in seen_questions:
#                                         seen_questions.add(question)
#                                         choices = [answer] + keyword_list
#                                         choices = list(set(choices))  # Remove duplicates
                                        
#                                         # Ensure there are exactly 4 options
#                                         if len(choices) < 4:
#                                             additional_choices = get_distractors_wordnet(keyword, topn=4-len(choices), question=summary)
#                                             choices.extend(additional_choices)
#                                             choices = list(set(choices))  # Remove duplicates

#                                         if len(choices) > 4:
#                                             random.shuffle(choices)
#                                             choices = choices[:4]  # Keep only 4 options

#                                         random.shuffle(choices)  # Shuffle options to avoid any bias
#                                         question_text = f"Q: {question}\nOptions:\n" + "\n".join([f"- {choice}" for choice in choices])
#                                         questions.append(question_text)
#                             except Exception as e:
#                                 print(f"Error in generating question for keyword {keyword}: {e}")
#             except Exception as e:
#                 print(f"Error in generating questions: {e}")

#         return questions

#     except Exception as e:
#         print(f"Error in get_mca_questions: {e}")
#         return []




# if __name__ == "__main__":
#     # Example text for testing
#     text = """
#     For all of us, nature is crucial. It’s the reason for the existence of life on this planet. Nature is home to many different creatures. All living organisms benefit from the natural balance maintained by Mother Nature. The study of the natural environment is a separate discipline of science. Every element has its own story to tell. Nature’s beauty is portrayed through the sun and moon, the plants, the flowers, etc. It is a common belief that reacting to something is a natural human characteristic. Naturally drawn characteristics are defined as genetic traits of an organism in sociology. The resources of nature are plentiful. The proper use of resources aids in the conservation of the environment. Natural scavengers include a variety of land and marine animals. Nature has provided us with a variety of ways to utilise it effectively.

# With the increasing population, the threats towards nature are increasing. With the growth in population, the resources are now depleting. Excessive levels of air and environmental pollutants add to the mix. Industrial waste, unrestricted vehicle use, illicit tree cutting, wildlife hunting, nuclear power plants, and a slew of other factors are contributing to the disruption of natural systems. The extinction of species as enormous as dinosaurs and the survival of animals as tiny as ants have been documented in history. It is unavoidable to remember, among other things, that nature can play both a protective and destructive role. Natural disasters, pandemics, and natural crisis scenarios have demonstrated the need for humans to maintain the subtle balance of nature in order to ensure the continuation of life on Earth for the benefit of future generations.
#     """
    
#     questions = get_mca_questions(text)
#     for q in questions:
#         print(q)

import warnings
warnings.filterwarnings("ignore")
import torch
import random
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
from textwrap3 import wrap
import random
from nltk.corpus import wordnet as wn
import numpy as np
import nltk
# nltk.download('punkt')
# nltk.download('brown')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import textdistance
import string
import pke
import traceback
from flashtext import KeywordProcessor
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time
import os

# Load Sense2Vec model
s2v = Sense2Vec().from_disk("C:\\Users\\DELL\\Desktop\\segmentation using speech-text\\s2v_reddit_2015_md\\s2v_old")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load or download T5 models and tokenizers
def load_model_and_tokenizer(model_name, tokenizer_name, model_file, tokenizer_file):
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"{model_name} model found on disk, model loaded successfully.")
    else:
        print(f"{model_name} model does not exist in the path specified, downloading the model from the web....")
        start_time = time.time()
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        end_time = time.time()
        print(f"Downloaded the {model_name} model in {(end_time - start_time) / 60} min, now saving it to disk...")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print("Done. Saved the model to disk.")

    if os.path.exists(tokenizer_file):
        with open(tokenizer_file, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"{tokenizer_name} tokenizer found on disk and is loaded successfully.")
    else:
        print(f"{tokenizer_name} tokenizer does not exist in the path specified, downloading the tokenizer from the web....")
        start_time = time.time()
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        end_time = time.time()
        print(f"Downloaded the {tokenizer_name} tokenizer in {(end_time - start_time) / 60} min, now saving it to disk...")
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(tokenizer, f)
        print("Done. Saved the tokenizer to disk.")

    return model.to(device), tokenizer

summary_model, summary_tokenizer = load_model_and_tokenizer('t5-base', 't5-base', 't5_summary_model.pkl', 't5_summary_tokenizer.pkl')
question_model, question_tokenizer = load_model_and_tokenizer('ramsrigouthamg/t5_squad_v1', 'ramsrigouthamg/t5_squad_v1', 't5_question_model.pkl', 't5_question_tokenizer.pkl')

# Load Sentence Transformer model
if os.path.exists("sentence_transformer_model.pkl"):
    with open("sentence_transformer_model.pkl", 'rb') as f:
        sentence_transformer_model = pickle.load(f)
    print("Sentence transformer model found on disk, model loaded successfully.")
else:
    print("Sentence transformer model does not exist in the path specified, downloading the model from the web....")
    start_time = time.time()
    sentence_transformer_model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v2")
    end_time = time.time()
    print(f"Downloaded the sentence transformer in {(end_time - start_time) / 60} min, now saving it to disk...")
    with open("sentence_transformer_model.pkl", 'wb') as f:
        pickle.dump(sentence_transformer_model, f)
    print("Done saving to disk.")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def postprocesstext(content):
    final = ""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final + " " + sent
    return final

def summarizer(text, model, tokenizer):
    text = text.strip().replace("\n", " ")
    text = "summarize: " + text
    max_len = 512
    encoding = tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outs = model.generate(input_ids=input_ids, attention_mask=attention_mask, early_stopping=True, num_beams=3, num_return_sequences=1, no_repeat_ngram_size=2, min_length=75, max_length=300)
    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    summary = dec[0]
    summary = postprocesstext(summary)
    summary = summary.strip()
    return summary

def get_nouns_multipartite(content):
    out = []
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content, language='en')
        pos = {'PROPN', 'NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = extractor.get_n_best(n=15)
        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
    return out

def get_keywords(originaltext):
    keywords = get_nouns_multipartite(originaltext)
    return keywords

def get_question(context, answer, model, tokenizer):
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outs = model.generate(input_ids=input_ids, attention_mask=attention_mask, early_stopping=True, num_beams=5, num_return_sequences=1, no_repeat_ngram_size=2, max_length=72)
    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    Question = dec[0].replace("question:", "").strip()
    return Question

def filter_same_sense_words(original, wordlist):
    filtered_words = []
    base_sense = original.split('|')[1]
    for eachword in wordlist:
        if eachword[0].split('|')[1] == base_sense:
            filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
    return filtered_words

def get_highest_similarity_score(wordlist, wrd):
    """
    This function takes the given word along with the wordlist and then gives out the max-score
    which is the Levenshtein distance for the wrong answers
    because we need the options which are very different from one another but relating to the same context.
    """
    score = []
    for each in wordlist:
        score.append(textdistance.levenshtein.similarity(each.lower(), wrd.lower()))
    return max(score)

def sense2vec_get_words(word, s2v, topn, question):
    output = []
    try:
        sense = s2v.get_best_sense(word, senses=["NOUN", "PERSON", "PRODUCT", "LOC", "ORG", "EVENT", "NORP", "WORK OF ART", "LAW", "LANGUAGE"])
        most_similar = s2v.most_similar(sense, n=topn)
        output = filter_same_sense_words(sense, most_similar)
        # If distractors are not sufficient, use synonyms
        if len(output) < topn:
            synonyms = [syn.lemmas()[0].name() for syn in wn.synsets(word)]
            if synonyms:
                output.extend(synonyms[:topn - len(output)])
    except Exception as e:
        print(f"Error retrieving distractors for word '{word}':", str(e))
    return output

def get_distractors(word, s2v, sentence_transformer_model, topn=10, max_distractors=4):
    distractors = sense2vec_get_words(word, s2v, topn, None)
    if not distractors:
        return []

    distractors_new = [word.capitalize()]
    distractors_new.extend(distractors)

    embedding_sentence = [word.capitalize()]
    embedding_sentence.extend(distractors)
    
    # Convert to list of numpy arrays
    embedding_sentence = sentence_transformer_model.encode(embedding_sentence, convert_to_tensor=False)
    
    # Calculate cosine similarity
    cosine_vals = cosine_similarity([embedding_sentence[0]], embedding_sentence[1:])
    cosine_vals = cosine_vals[0]
    distractors = [distractors[i] for i in cosine_vals.argsort()[:max_distractors]]
    return distractors

def mcq(question, answer, distractors, max_distractors=4):
    # Ensure the correct answer is in the distractors list
    if answer not in distractors:
        distractors.append(answer)
    
    # If there are more distractors than needed, reduce to max_distractors
    if len(distractors) > max_distractors:
        # Ensure the answer is part of the sample
        if answer in distractors:
            distractors.remove(answer)
        distractors = random.sample(distractors, max_distractors - 1)
        distractors.append(answer)
    
    choices = distractors[:]
    random.shuffle(choices)
    correct_option = chr(choices.index(answer) + 65)  # Convert index to letter (A, B, C, D)
    
    return {
        "question": question,
        "answer": answer,
        "choices": choices,
        "correct_option": correct_option
    }

# Input text for demonstration
input_text = """
Leveraging AI-Driven Video-Based Learning to Enhance Student Engagement and Collaboration has been a significant topic of interest in educational research. AI-driven video-based learning platforms offer personalized learning experiences, adaptive feedback, and interactive content, all of which can significantly boost student engagement and collaboration. By incorporating natural language processing and machine learning algorithms, these platforms can analyze student interactions, predict learning outcomes, and provide real-time support. Furthermore, the integration of AI technologies in video-based learning can facilitate collaborative learning by enabling students to interact with each other and the content in innovative ways. This, in turn, fosters a more engaging and collaborative learning environment that can lead to improved academic performance and a deeper understanding of the subject matter.
"""

# Summarize input text
summarized_text = summarizer(input_text, summary_model, summary_tokenizer)

# Extract keywords
keywords = get_keywords(summarized_text)

# Generate questions and distractors
questions_and_distractors = []
for keyword in keywords:
    question = get_question(summarized_text, keyword, question_model, question_tokenizer)
    distractors = get_distractors(keyword, s2v, sentence_transformer_model)
    if distractors:
        q_and_d = mcq(question, keyword, distractors)
        questions_and_distractors.append(q_and_d)

# Print the output questions and their choices
for idx, q_and_d in enumerate(questions_and_distractors, start=1):
    print(f"Q{idx}: {q_and_d['question']}")
    for option, choice in zip(['A', 'B', 'C', 'D'], q_and_d['choices']):
        print(f"{option}. {choice}")
    print(f"Answer: {q_and_d['correct_option']} ({q_and_d['answer']})\n")

