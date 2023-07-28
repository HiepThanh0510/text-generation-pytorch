import torch 
from transformers import GPT2LMHeadModel, AutoTokenizer
from BEAM import Beam 

class Generator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    # Split text to token and convert token to id
    def encode(self, text):
        return self.tokenizer.encode(text)

    # Convert id to text
    def decode(self, ids):
        return self.tokenizer.decode(ids)

    # Get the id with the greatest probability
    @staticmethod
    def get_top1(prob):
        score, token_id = torch.max(prob, dim=-1)
        return score, token_id

    # Get the top k id with the greatest probability
    @staticmethod
    def get_topk(prob, k=1):
        scores, token_ids = torch.topk(prob, k=k, dim=-1)
        return scores, token_ids

    # Get next token prob, returns the probability of all tokens
    def get_next_token_prob(self, input_ids: torch.Tensor):
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits
        next_token_prob = logits[-1]
        return next_token_prob

    def generate_greedy_search(self, prompt, max_new_tokens=32, no_repeat_ngram_size=0):
        token_ids = self.encode(prompt)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        # loop until the maximum length is reached
        for i in range(max_new_tokens):
            next_token_prob = self.get_next_token_prob(input_ids=input_ids)
            
            if no_repeat_ngram_size > 0:
                '''
                Idea: 
                - With no_repeat_ngram_size = 3 
                - I have the sentence: 
                    "Hôm nay tôi đi làm ca sáng, ngày mai tôi cũng đi làm"
                - Convert [4253, 767, 554, 585, 471, 1047, 935, 16, 563, 2571, 554, 485, 585, 471] to dictionary below: 
                
                    {(4253, 767): 554,
                    (767, 554): 585,
                    (554, 585): 471,
                    (585, 471): 1047,
                    (471, 1047): 935,
                    (1047, 935): 16,
                    (935, 16): 563,
                    (16, 563): 2571,
                    (563, 2571): 554,
                    (2571, 554): 485,
                    (554, 485): 585,
                    (485, 585): 471}
        
                - Every time i want to generate a new word, pick the 
                last 2 words and check if they match any keys in that 
                dictionary. If they match, retrieve the value of that key.
                - This value should not be used by the model to generate 
                the next word, but rather should be avoided. Simply 
                set next_token_prob[value] = 0.
                '''
                # convert tensor to list 
                my_list = input_ids.tolist()
                # retrieve the desired dictionary from the previous list 
                my_dict = {tuple(my_list[i:i+no_repeat_ngram_size-1]): my_list[i+no_repeat_ngram_size-1] for i in range(len(my_list)-no_repeat_ngram_size+1)}
                # get the last no_repeat_ngram_size - 1 tokens of the sentence
                checked_tokens = tuple(input_ids.tolist()[-(no_repeat_ngram_size-1):])
                # if the checked_tokens match any keys of the desired dictionary, then set next_token_prob[corresponding value] = 0
                if checked_tokens in my_dict.keys():
                    get_value = my_dict[checked_tokens] 
                    next_token_prob[get_value] = 0 
                    
            # get the token with the highest score 
            score, token_id = self.get_top1(next_token_prob)
            '''
            The token_id will have the form of torch.tensor(number): the dimension is equal to 0, 
            the type is similar to scalar, so to concatenate with input_ids, unsqueeze() is needed.
            '''
            input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=-1)

        output = self.decode(input_ids.tolist()[len(token_ids):])
        print(output)
    
    def generate_beam_search(self, prompt, max_new_tokens=32, num_beam=1, no_repeat_ngram_size=0):
        token_ids = self.encode(prompt)
        input_ids = torch.tensor(token_ids, dtype=torch.long)

        # initialize the beam
        beams = [Beam(num_beam, input_ids, 1)]
        # loop until the maximum length is reached
        for i in range(max_new_tokens):
            # create a list called all_next_token_prob to contain the next_token_probs of the beams
            all_next_token_prob = []
            for beam in beams:
                next_token_prob = self.get_next_token_prob(input_ids=beam.get_current_state()) 
                '''
                The idea behind no_repeat_ngram_size is the same as the no_repeat_ngram_size used in greedy search. 
                '''
                if no_repeat_ngram_size > 0: 
                    my_list = beam.get_current_state().tolist() 
                    my_dict = {tuple(my_list[i:i+no_repeat_ngram_size-1]): my_list[i+no_repeat_ngram_size-1] for i in range(len(my_list)-no_repeat_ngram_size+1)}
                    checked_tokens = tuple(beam.get_current_state().tolist()[-(no_repeat_ngram_size-1):])
                    if checked_tokens in my_dict.keys():
                        get_value = my_dict[checked_tokens]
                        next_token_prob[get_value] = 0
                all_next_token_prob.append(next_token_prob) 
                
            all_next_token_prob_tensor = torch.stack(all_next_token_prob, dim=0)
            # Get top k tokens for each beam
            topk_scores, topk_token_ids = self.get_topk(all_next_token_prob_tensor, k=num_beam)
            # Create new beams
            new_beams = []
            for j, beam in enumerate(beams):
                for k in range(num_beam):
                    score = topk_scores[j][k]
                    token_id = topk_token_ids[j][k]
                    new_beam = beam.extend(token_id, score)

                    new_beams.append(new_beam)      
            # If there are no [:num_beam], then after each iteration, the length of beams is num_beam ^ (the i-th iteration).
            # So, i customed this one 
            beams = sorted(new_beams, key=lambda b: b.get_score(), reverse=True)[:num_beam] 
        print(f"no_repeat_ngram_size = {no_repeat_ngram_size}")
        print(f"num_beam = {num_beam}")
        print(self.decode(beams[0].output.tolist()[len(token_ids):]))