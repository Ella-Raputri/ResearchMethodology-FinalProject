from symspellpy import SymSpell, Verbosity
from transformers import pipeline
import re

from collections import defaultdict
from wordfreq import top_n_list
english_vocab = set(top_n_list("en", 50000))

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm


sym_spell = SymSpell()
sym_spell.load_pickle("../data/dictionary/symspell_dictionary.pkl")
ner = pipeline("ner", model="cahya/NusaBert-ner-v1.3", grouped_entities=True)
model_name = "indolem/indobert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

ocr_confusions = {
    'rn': ['m'],
    'm': ['rn'],
    'l': ['t', 'i'],
    't': ['l'],
    '0': ['o'],
    '1': ['l', 'i'],
    'o': ['0'],
    'n': ['ri', 'ni'],
    'vv': ['w'],
    'w': ['vv'],
    'e':['c']
}

def find_typos(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    typos = defaultdict(list)
    
    for sentence in sentences:
        words = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
        # print(words)
        
        for word in words:
            if word in english_vocab or len(word) <= 2:
                continue
            
            # print(word)
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
            best = suggestions[0]
            
            if (best.term != word and best.distance > 0) or (best.count < 10):
                if sentence.strip() not in typos[word]:  
                    typos[word].append(sentence.strip())

    return typos

def should_correct(entity_label, suggestion):
    if entity_label in {"PERSON", "GPE", "LOC"}:
        return False

    if entity_label == "ORG":
        if suggestion in sym_spell.words:
            return True
        return False
    
    if entity_label is None:
        return suggestion in sym_spell.words

    return False

def expand_ocr_variants(word):
    variants = set()
    for pattern, subs in ocr_confusions.items():
        if pattern in word:
            for s in subs:
                variants.add(re.sub(pattern, s, word))
    return variants

def suggest_words(word):
    suggest_symspell = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
    
    valid_suggestions = [s for s in suggest_symspell if s.count > 10]
    if valid_suggestions: best = valid_suggestions[0].term
    else: best = suggest_symspell[0].term

    variants = expand_ocr_variants(word)

    final_s = set()
    final_s.add(best)

    for var in variants:
        if var in sym_spell.words:
            final_s.add(var)
    return final_s

def output_highest(model, tokenizer, sentence, cands, word):
    sentence = sentence.lower()
    window = 500

    if (len(sentence) >= window*2): 
        idx = sentence.find(word)
        start = max(0, idx - window)
        end = min(len(sentence), idx + len(word) + window)
        sentence = sentence[start:end]

    sentence = sentence.replace(word,'[MASK]')

    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    mask_token_id = tokenizer.mask_token_id
    mask_token_index = (inputs.input_ids == mask_token_id).nonzero(as_tuple=True)[1]
    mask_logits = logits[0, mask_token_index, :]

    candidate_ids = tokenizer.convert_tokens_to_ids(cands)
    candidate_scores = mask_logits[0, candidate_ids]
    
    best_candidate_index = candidate_scores.argmax().item()
    best_candidate = cands[best_candidate_index]
    return best_candidate

def context_correct(typos):
    corrections = []

    for word, sentences in tqdm(typos.items(), desc='Processing words'):
        for sentence in sentences:
            doc = ner(sentence)
            entity_label = None

            # cari entity label kata typo
            for ent in doc:
                if word.lower() in ent["word"].lower():
                    entity_label = ent["entity_group"]
                    break
            
            suggestions = suggest_words(word)
            if not suggestions:
                continue #kalau gada suggestions

            best_word = output_highest(model, tokenizer, sentence, list(suggestions), word)
            if(not should_correct(entity_label, best_word)): continue
            corrections.append({"word":word, "correction":best_word})
            
    return corrections


def symspell_clean(text):
    typos = find_typos(text)
    res = context_correct(typos)
    corrected_text = text
    for obj in res:
        wrong = re.escape(obj['word'])
        corr = obj['correction']
        corrected_text = re.sub(rf"\b{wrong}\b", corr, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text
