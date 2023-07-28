import argparse
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from GENERATOR import Generator

def get_args():
    parser = argparse.ArgumentParser(description='text generation using greedy search and beam search')
    parser.add_argument('--model_name',
                        type=str,
                        default="NlpHUST/gpt2-vietnamese",
                        help='model name')

    parser.add_argument('--prompt',
                        type=str,
                        required=True,
                        help='prompt')

    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=1,
                        help='number of tokens generated')

    parser.add_argument('--no_repeat_ngram_size',
                        type=int,
                        default=0,
                        help='no repeat ngram size')

    parser.add_argument('--generate_type',
                        type=str,
                        default="greedy_search",
                        choices=["greedy_search", "beam_search"],
                        help='generate type')

    parser.add_argument('--num_beam',
                        type=int,
                        default=1,
                        help='num beam when generate_type="beam_search"')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    generator = Generator(args.model_name)
    
    if args.generate_type == "greedy_search":
        generator.generate_greedy_search(prompt=args.prompt, max_new_tokens=args.max_new_tokens,
                                         no_repeat_ngram_size=args.no_repeat_ngram_size)
    else:
        generator.generate_beam_search(prompt=args.prompt, max_new_tokens=args.max_new_tokens, num_beam=args.num_beam,
                                       no_repeat_ngram_size=args.no_repeat_ngram_size)

if __name__ == '__main__':
    main()