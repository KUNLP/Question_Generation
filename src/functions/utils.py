from transformers import (BartForConditionalGeneration,
                          PreTrainedTokenizerFast)
import torch
from tqdm import tqdm
import nltk
from torch.utils.data import TensorDataset
import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from src.functions.rouge import Rouge

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def measure(preds, refs):
    b1, b2, b3, b4, r_l = 0, 0, 0, 0, 0
    r = Rouge()
    count = 0
    for index in range(len(preds)):
        pred = preds[index]
        ref = refs[index]
        b1 += sentence_bleu([ref], pred, weights=(1, 0, 0, 0))
        b2 += sentence_bleu([ref], pred, weights=(0.5, 0.5, 0, 0))
        b3 += sentence_bleu([ref], pred, weights=(0.33, 0.33, 0.33, 0))
        b4 += sentence_bleu([ref], pred, weights=(0.25, 0.25, 0.25, 0.25))
        r_l += r.rouge_l(["".join(pred).replace("▁", " ")], ["".join(ref).replace("▁", " ")])[2]
        # print(decode)
        count += 1
    print("BLEU-1 = ", b1 / count)
    print("BLEU-2 = ", b2 / count)
    print("BLEU-3 = ", b3 / count)
    print("BLEU-4 = ", b4 / count)
    print("ROUGE-L = ", r_l / count)

class ChatDataset():
    def __init__(self, filepath, tokenizer, enc_seq_len=300, dec_seq_len=30):
        self.filepath = filepath
        self.data = open(self.filepath, 'r', encoding='utf8').readlines()
        # self.data = pd.read_csv(filepath)
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.tokenizer = tokenizer
    def make_input_id_mask(self, tokens, max_seq_len, passage_ids=None):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        if len(input_id) < max_seq_len:
            while len(input_id) < max_seq_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
                if passage_ids is not None:
                    passage_ids += [0]
        else:
            # logging.warning(f'exceed max_seq_len for given article : {index}')
            input_id = input_id[:max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:max_seq_len]
            if passage_ids is not None:
                passage_ids = passage_ids[:max_seq_len]
        if passage_ids is not None:
            return input_id, attention_mask, passage_ids
        return input_id, attention_mask
    def gold_passage_mask(self, context):
        split_context = nltk.sent_tokenize(context)
        passage_label = [1 if '<mask>' not in e else 2 for e in split_context]
        tokenized_split_context = [self.tokenizer.tokenize(e) for e in split_context]
        tokenized_context = []
        passage_ids = []
        for e in range(len(passage_label)):
            tokenized_context+=tokenized_split_context[e]
            passage_ids+=[passage_label[e]]*len(tokenized_split_context[e])

        return tokenized_context, passage_ids
    def gold_passage_mask_v2(self, context):
        split_context = nltk.sent_tokenize(context)
        passage_label = [0 if '<mask>' not in e else 1 for e in split_context]
        tokenized_split_context = [self.tokenizer.tokenize(e) for e in split_context]
        tokenized_context = []
        passage_ids = []
        for e in range(len(passage_label)):
            if passage_label[e] == 0:
                continue
            tokenized_context+=tokenized_split_context[e]
            passage_ids+=[passage_label[e]]*len(tokenized_split_context[e])

        return tokenized_context, passage_ids
    def make_dataset(self, title, context):
        input_ids = []
        attention_masks = []


        answer = input("Enter The Answer To The Question : ")
        if '-1' in [title, context, answer]:
            exit(1)
        title_tokens = [self.bos_token] + \
                       self.tokenizer.tokenize(title) + [self.eos_token]
        answer_tokens = [self.bos_token] + \
                        self.tokenizer.tokenize(answer) + [self.eos_token]
        context_tokens = [self.bos_token] + \
                         self.tokenizer.tokenize(context) + [self.eos_token]
        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(
            title_tokens + answer_tokens + context_tokens, self.enc_seq_len)

        input_ids.append(encoder_input_id)
        attention_masks.append(encoder_attention_mask)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)

        return TensorDataset(input_ids, attention_masks)
    def load_dataset_with_passage_ids(self):
        input_ids = []
        attention_masks = []
        gold_passage_ids = []
        decoder_input_ids = []
        decoder_attention_masks = []
        decoder_labels = []
        ids, levels = [], []
        for index in tqdm(range(len(self.data))):
            # title, answer, context, question = self.data[index].strip().split('\t')
            datas = self.data[index].strip().split('\t')
            if len(datas) != 7:
                print(index, "!!!")
                continue
            id, level, title, answer, evidence_sent, context, question = datas
            if level == '하' or level == '중':
                continue
            ids.append(id)
            levels.append(level)
            tokenized_context, passage_ids = self.gold_passage_mask_v2(context)
            title_tokens = [self.bos_token] + \
                           self.tokenizer.tokenize(title) + [self.eos_token]
            answer_tokens = [self.bos_token] + \
                            self.tokenizer.tokenize(answer) + [self.eos_token]
            evidence_tokens = [self.bos_token] + \
                            self.tokenizer.tokenize(evidence_sent) + [self.eos_token]
            context_tokens = [self.bos_token] + \
                             tokenized_context + [self.eos_token]
            passage_ids = [0] + passage_ids + [0]
            question_tokens = [self.bos_token] + \
                              self.tokenizer.tokenize(question) + [self.eos_token]
            encoder_input_id, encoder_attention_mask, encoder_gold_passage_id = self.make_input_id_mask(
                title_tokens + answer_tokens + evidence_tokens + context_tokens, self.enc_seq_len,  [0]*len(title_tokens + answer_tokens + evidence_tokens)+passage_ids)
            # encoder_input_id, encoder_attention_mask, encoder_gold_passage_id = self.make_input_id_mask(
            #     title_tokens + answer_tokens + context_tokens, self.enc_seq_len,
            #     [0] * len(title_tokens + answer_tokens) + passage_ids)
            decoder_input_id, decoder_attention_mask = self.make_input_id_mask(
                question_tokens, self.dec_seq_len)
            labels = self.tokenizer.convert_tokens_to_ids(
                question_tokens[1:(self.dec_seq_len + 1)])
            if len(labels) < self.dec_seq_len:
                while len(labels) < self.dec_seq_len:
                    # for cross entropy loss masking
                    labels += [-100]
            # if len(input_ids) > 100:
            #     break
            # print(context_tokens)
            # print(question_tokens)
            # print(decoder_labels)

            input_ids.append(encoder_input_id)
            attention_masks.append(encoder_attention_mask)
            gold_passage_ids.append(encoder_gold_passage_id)
            decoder_input_ids.append(decoder_input_id)
            decoder_attention_masks.append(decoder_attention_mask)
            decoder_labels.append(labels)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        gold_passage_ids = torch.tensor(gold_passage_ids, dtype=torch.long)
        decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
        decoder_attention_masks = torch.tensor(decoder_attention_masks, dtype=torch.long)
        decoder_labels = torch.tensor(decoder_labels, dtype=torch.long)

        return TensorDataset(input_ids, attention_masks, gold_passage_ids, decoder_input_ids, decoder_attention_masks, decoder_labels), ids, levels
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
            # if len(input_ids) > 100:
            #     break
            # print(context_tokens)
            # print(question_tokens)
            # print(decoder_labels)

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

        return TensorDataset(input_ids, attention_masks, decoder_input_ids, decoder_attention_masks, decoder_labels)