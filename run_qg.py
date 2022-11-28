import argparse
from src.model.model import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
from transformers.models.bart.configuration_bart import BartConfig
from src.model.main_functions import train, evaluate, make_file
from attrdict import AttrDict
import os

from transformers import generation_utils
def create_model(args):
    config = BartConfig.from_pretrained(args.model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)

    init_weight = args.model_path if args.from_init_weight else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint))
    print("Init Weight From {}".format(init_weight))
    model = BartForConditionalGeneration.from_pretrained(init_weight, config=config)

    model.to(args.device)

    return model, tokenizer

def main(cil_args):
    args = AttrDict(vars(cil_args))
    args.device = 'cuda'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(args)
    model, tokenizer = create_model(args)

    if args.do_train:
        train(args, model, tokenizer)
    elif args.do_evaluate:
        evaluate(args, model, tokenizer)
    elif args.do_predict:
        make_file(args, model, tokenizer)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    # Path
    cli_parser.add_argument('--model_path', type=str, default='hyunwoongko/kobart', help='kobart model path')
    cli_parser.add_argument('--train_file', type=str, default='processed_ai_data/indomain_train.txt', help='train file')
    cli_parser.add_argument('--test_file', type=str, default='processed_ai_data/indomain_test.txt', help='test file')
    cli_parser.add_argument('--predict_file', type=str, default='processed_ai_data/all_outdomain_aug.txt', help='test file')
    cli_parser.add_argument('--tokenizer_path', type=str, default='emji_tokenizer', help='tokenizer')
    cli_parser.add_argument('--output_dir', type=str, default='./indomain', help='tokenizer')


    # Training Parameter
    cli_parser.add_argument("--weight_decay", type=float, default=0.0)

    cli_parser.add_argument('--num_workers', type=int, default=5, help='num of worker for dataloader (# of CPU Cores for Pre-processing)')
    cli_parser.add_argument('--learning_rate', type=float, default=5e-5, help='The initial learning rate')
    cli_parser.add_argument("--adam_epsilon", type=int, default=1e-10)
    cli_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    cli_parser.add_argument("--max_grad_norm", type=int, default=1.0)

    cli_parser.add_argument("--seed", type=int, default=42)
    cli_parser.add_argument("--save_steps", type=int, default=2000)
    cli_parser.add_argument("--train_epochs", type=int, default=40)
    cli_parser.add_argument("--checkpoint", type=int, default=20000)
    cli_parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    cli_parser.add_argument('--max_seq_len', type=int, default=128, help='max seq len')
    cli_parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')

    cli_parser.add_argument('--from_init_weight', type=bool, default=False, help='init weight var')
    cli_parser.add_argument('--do_train', type=bool, default=False, help='Train Mode Bool Variable')
    cli_parser.add_argument('--do_evaluate', type=bool, default=False, help='Evaluate Mode Bool Variable')
    cli_parser.add_argument('--do_predict', type=bool, default=True, help='Predict Mode Bool Variable')
    cli_args = cli_parser.parse_args()
    main(cli_args)