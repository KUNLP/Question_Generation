from transformers import (BartForConditionalGeneration,
                          PreTrainedTokenizerFast)
import torch
from tqdm import tqdm

class ChatDataset():
    def __init__(self, filepath, tok_vocab, enc_seq_len=512, dec_seq_len=64):
        self.filepath = filepath
        self.data = open(self.filepath, 'r', encoding='utf8').readlines()
        # self.data = pd.read_csv(filepath)
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_vocab,
            bos_token=self.bos_token, eos_token=self.eos_token, unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

    def make_input_id_mask(self, tokens, max_seq_len):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        if len(input_id) < max_seq_len:
            while len(input_id) < max_seq_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            # logging.warning(f'exceed max_seq_len for given article : {index}')
            input_id = input_id[:max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:max_seq_len]
        return input_id, attention_mask

    def load_dataset(self):
        input_ids = []
        attention_masks = []
        decoder_input_ids = []
        decoder_attention_masks = []
        decoder_labels = []
        for index in tqdm(range(len(self.data))):
            title, answer, context, question = self.data[index].strip().split('\t')
            title_tokens = [self.bos_token] + \
                           self.tokenizer.tokenize(title) + [self.eos_token]
            answer_tokens = [self.bos_token] + \
                            self.tokenizer.tokenize(answer) + [self.eos_token]
            context_tokens = [self.bos_token] + \
                             self.tokenizer.tokenize(context) + [self.eos_token]
            question_tokens = [self.bos_token] + \
                              self.tokenizer.tokenize(question) + [self.eos_token]
            encoder_input_id, encoder_attention_mask = self.make_input_id_mask(
                title_tokens + answer_tokens + context_tokens, self.enc_seq_len)
            decoder_input_id, decoder_attention_mask = self.make_input_id_mask(
                question_tokens, self.dec_seq_len)
            labels = self.tokenizer.convert_tokens_to_ids(
                question_tokens[1:(self.dec_seq_len + 1)])
            if len(labels) < self.dec_seq_len:
                while len(labels) < self.dec_seq_len:
                    # for cross entropy loss masking
                    labels += [-100]
            input_ids.append(encoder_input_id)
            attention_masks.append(encoder_attention_mask)
            decoder_input_ids.append(decoder_input_id)
            decoder_attention_masks.append(decoder_attention_mask)
            decoder_labels.append(labels)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
        decoder_attention_masks = torch.tensor(decoder_attention_masks, dtype=torch.long)
        decoder_labels = torch.tensor(decoder_labels, dtype=torch.long)

        return input_ids, attention_masks, decoder_input_ids, decoder_attention_masks, decoder_labels