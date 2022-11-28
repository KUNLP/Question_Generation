import os
import torch
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from src.functions.utils import ChatDataset, set_seed
from transformers import AdamW
from src.functions.utils import measure
from src.functions.rouge import Rouge
import timeit

def to_list(tensor):
    return tensor.detach().cpu().tolist()

logger = logging.getLogger(__name__)

def train(args, model, tokenizer):
    """ Train the model """
    dataset = ChatDataset(args.train_file, tokenizer)
    # train_dataset = dataset.load_dataset()
    train_dataset, _, _ = dataset.load_dataset_with_passage_ids()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    global_step = 1
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # Added here for reproductibility
    set_seed(args)

    for epoch in range(args.train_epochs):
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "passage_ids": batch[2],
                "decoder_input_ids": batch[3],
                "decoder_attention_mask": batch[4],
                "decoder_labels":batch[5],
            }
            # batch_size = batch[0].size()[0]
            # ee = batch[0].tolist()
            # d = batch[2].tolist()
            # dd = batch[4].tolist()
            # for e in range(batch_size):
            #     enc_input = dataset.tokenizer.batch_decode(ee[e])
            #     dec_input = dataset.tokenizer.batch_decode(d[e][:d[e].index(3)])
            #     dec_labels = dataset.tokenizer.batch_decode(dd[e][:dd[e].index(-100)])
            #     print(33333)

            outputs = model(input_ids=batch[0],
                              attention_mask=batch[1],
                              passage_ids = batch[2],
                              decoder_input_ids=batch[3],
                              decoder_attention_mask=batch[4],
                              labels=batch[5], return_dict=True)
            loss = outputs["loss"]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            if (global_step+1) % 50 == 0:
                print("{} Processed.. Total Loss : {}".format(global_step+1, loss.item()))
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                model.zero_grad()
                global_step += 1

                # Save model checkpoint
                if global_step % args.save_steps == 0:
                    evaluate(args, model, tokenizer, global_step)
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    print("Model Save in {}".format(output_dir))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)


                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

    return global_step, tr_loss / global_step
from tqdm import tqdm
def evaluate(args, model, tokenizer, global_step=0):
    model.eval()
    dataset = ChatDataset(args.test_file, tokenizer)
    # test_dataset = dataset.load_dataset()
    test_dataset, _, _ = dataset.load_dataset_with_passage_ids()
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    output_file = open(os.path.join(args.output_dir, "result_{}_only_hard.txt".format(global_step)), 'w', encoding='utf8')
    preds = []
    refs = []
    for batch in tqdm(test_dataloader):
        # Skip past any already trained steps if resuming training
        model.train()
        batch = tuple(t.to(args.device) for t in batch)


        dec_outputs = model.generate(input_ids = batch[0],
                                                 attention_mask=batch[1],
                                                 passage_ids=batch[2],
                                                 max_length=32,
                                                num_beams=5,
                                                eos_token_id=1,
                                                bad_words_ids=[[5]])
        batch_size = batch[0].size()[0]

        dec_outputs = dec_outputs.tolist()
        dec_labels = batch[5].tolist()

        for index in range(batch_size):
            if 1 in dec_outputs[index]:
                dec_outputs[index] = dec_outputs[index][:dec_outputs[index].index(1)]
            if -100 in dec_labels[index]:
                dec_labels[index] = dec_labels[index][:dec_labels[index].index(-100)]
            pred = dataset.tokenizer.convert_ids_to_tokens(dec_outputs[index][1:])
            ref = dataset.tokenizer.convert_ids_to_tokens(dec_labels[index][:-1])
            output_file.write("REFERENCE : {}\nDECODED   : {}\n\n".format(''.join(ref), ''.join(pred)))
            preds.append(pred)
            refs.append(ref)
    measure(preds, refs)
import json
def make_file(args, model, tokenizer, global_step=0):
    model.eval()
    dataset = ChatDataset(args.predict_file, tokenizer)
    # test_dataset = dataset.load_dataset()
    test_dataset, ids, levels = dataset.load_dataset_with_passage_ids()
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    # output_file = open(os.path.join(args.output_dir, "all_silver_sent_result_{}.json".format(global_step)), 'w', encoding='utf8')
    output_file = open(os.path.join(args.output_dir, "all_outdomain_aug.json".format(global_step)), 'w',
                       encoding='utf8')
    result_dict = {}
    preds = []
    refs = []
    step = 0
    for batch in tqdm(test_dataloader):
        # Skip past any already trained steps if resuming training
        model.train()
        batch = tuple(t.to(args.device) for t in batch)

        dec_outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     passage_ids=batch[2],
                                     max_length=32,
                                     num_beams=5,
                                     eos_token_id=1,
                                     bad_words_ids=[[5]])
        batch_size = batch[0].size()[0]

        dec_outputs = dec_outputs.tolist()
        dec_labels = batch[5].tolist()

        for index in range(batch_size):
            if 1 in dec_outputs[index]:
                dec_outputs[index] = dec_outputs[index][:dec_outputs[index].index(1)]
            if -100 in dec_labels[index]:
                dec_labels[index] = dec_labels[index][:dec_labels[index].index(-100)]
            pred = dataset.tokenizer.convert_ids_to_tokens(dec_outputs[index][1:])
            ref = dataset.tokenizer.convert_ids_to_tokens(dec_labels[index][:-1])
            # output_file.write("REFERENCE : {}\nDECODED   : {}\n\n".format(''.join(ref), ''.join(pred)))
            preds.append(pred)
            refs.append(ref)
            result_dict[ids[step]] = [''.join(pred).replace("▁", " "), levels[step]]
            step += 1
    json.dump(result_dict, output_file, indent='\t', ensure_ascii=False)
# def predict(args, model, tokenizer):
#     model.eval()
#     dataset = ChatDataset(args.test_file, tokenizer)
#     print("Enter -1 to Quit")
#     title = input("Enter The Document Title : ")
#     context = input("Enter The Document : ")
#     while(1):
#         pred_dataset = dataset.make_dataset(title, context)
#         pred_dataloader = DataLoader(pred_dataset, batch_size=args.batch_size)
#
#         for step, batch in enumerate(pred_dataloader):
#             # Skip past any already trained steps if resuming training
#             model.train()
#             batch = tuple(t.to(args.device) for t in batch)
#
#             inputs = {
#                 "input_ids": batch[0],
#                 "attention_mask" : batch[1]
#             }
#
#             dec_outputs = model.generate(input_ids = batch[0],
#                                                      attention_mask=batch[1],
#                                                      max_length=32,
#                                                     num_beams=5,
#                                                     eos_token_id=1,
#                                                     bad_words_ids=[[5]])
#
#
#             dec_outputs = dec_outputs.tolist()[0]
#
#             if 1 in dec_outputs:
#                 dec_outputs = dec_outputs[:dec_outputs.index(1)]
#
#             pred = dataset.tokenizer.convert_ids_to_tokens(dec_outputs[1:])
#             print("Generated Question : ", ''.join(pred).replace("▁", " "), "\n\n")



