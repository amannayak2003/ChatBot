import re

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        text = text.replace("\t", " ")
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'http\S+', '', text)         
        text = re.sub(r'\[[0-9]+\]', '', text)       
        text = re.sub(r'[^a-zA-Z0-9.,?!;:()\-\s]', '', text)
        return text.strip()