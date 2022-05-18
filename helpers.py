import re
from nltk.corpus import stopwords
from textblob import TextBlob, Word 
def remover_html(texto):
    texto_prep = re.compile('<.*?>')
    texto_prep = re.sub(texto_prep, ' ', str(texto))
    return texto_prep

def remover_caracteres_especiales(texto): 
    texto_prep = re.sub(r'[?|!|\'|"|#]',r'',texto)
    texto_prep = re.sub(r'[.|,|)|(|\|/]',r' ',texto_prep)
    texto_prep = texto_prep.strip()
    texto_prep = texto_prep.replace("\n"," ")
    return texto_prep

def mantener_caracteres_alfabeticos(texto):
    #texto_prep = ""
    #for word in texto.split():
    #    alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
    #    texto_prep += alpha_word
    #    texto_prep += " "
    #texto_prep = texto_prep.strip()
    return ' '.join(re.sub('[^a-z A-Z]+',' ',texto).split())
    #return texto_prep
    

def lemmatizar_con_postag(texto):
    sent = TextBlob(texto)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lista_lematizada = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lista_lematizada)




def remover_stopwords(texto,stop_words):
    texto_prep = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    return texto_prep.sub(" ", texto)