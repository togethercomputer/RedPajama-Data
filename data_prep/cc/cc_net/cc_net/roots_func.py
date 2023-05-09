from cc_net import jsonql

class filter_small_doc(jsonql.Transformer):
    """Filter txt string, delete parts with less than 10 characters"""
    def __init__(self, field, text_min_length=15):
        super().__init__()
        self.field = field
        self.text_min_length = text_min_length

    def do(self, doc: dict):
        text = doc.get(self.field, None)
        if not text:
            return None
        
        def filter_txt(txt, min_length=15):
            """Filter txt string, delete parts with less than specified min_length"""
            options = txt.split('\n')  
            result = []   

            for option in options:    
                if len(option) < min_length:      
                    continue    
                result.append(option)      
                result.append('\n')  
            return ''.join(result) 
        doc[self.field] = filter_txt(doc[self.field], self.text_min_length)
        return doc

class filter_small_docs_by_bytes(jsonql.Transformer):
    def __init__(self, field, text_min_bytes=500):
        """
            a byte of text usually turns into 0.3 tokens. 
            a 256-token sequence would be ~850 bytes of text. 
            I think anywhere from 500 to 1000 min bytes is reasonable.
        """
        super().__init__()
        self.field = field
        self.text_min_bytes = text_min_bytes
    
    def do(self, doc: dict):
        text = doc.get(self.field, None)
        if not text:
            return None
        if len(text.encode('utf-8')) < self.text_min_bytes:
            return None
        return doc