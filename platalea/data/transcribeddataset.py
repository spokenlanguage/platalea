import torch
from sklearn.preprocessing import LabelEncoder


class TranscribedDataset():
    le = None
    sos = '<sos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'

    @classmethod
    def init_vocabulary(cls, transcriptions):
        cls.le = LabelEncoder()
        tokens = [cls.sos, cls.eos, cls.unk, cls.pad] + \
                 [c for t in transcriptions for c in t]
        cls.le.fit(tokens)

    @classmethod
    def get_label_encoder(cls):
        if cls.le is None:
            raise ValueError('Vocabulary not initialized.')
        return cls.le

    @classmethod
    def get_token_id(cls, token):
        return cls.get_label_encoder().transform([token])[0]

    @classmethod
    def vocabulary_size(cls):
        return len(cls.get_label_encoder().classes_)

    @classmethod
    def caption2tensor(cls, capt):
        le = cls.get_label_encoder()
        capt = [c if c in le.classes_ else cls.unk for c in capt]
        capt = [cls.sos] + capt + [cls.eos]
        return torch.Tensor(le.transform(capt))