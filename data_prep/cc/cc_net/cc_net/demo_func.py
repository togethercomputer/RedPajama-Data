from cc_net import jsonql

class replace(jsonql.Transformer):
    def __init__(self, field, souce_str="a", target_str="=====>a<======="):
        super().__init__()
        self.field = field
        self.souce_str = souce_str
        self.target_str = target_str

    def do(self, doc: dict):
        text = doc.get(self.field, None)
        if not text:
            return None
        doc[self.field] = doc[self.field].replace(self.souce_str, self.target_str)
        return doc