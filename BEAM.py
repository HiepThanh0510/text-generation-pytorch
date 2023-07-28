import torch 

class Beam:
    def __init__(self, size, input_ids, score, output=None):
        self.size = size # num_beam 
        self.input_ids = input_ids
        self.score = score
        self.output = output
        
    # get input_ids 
    def get_current_state(self):
        return self.input_ids
    
    # get probability of the sentence         
    def get_score(self):
        return self.score
    
    # create a new instance of Beam class after the top k selection
    def extend(self, token_id, score):
        new_input_ids = torch.cat([self.input_ids, torch.tensor([token_id])], dim=-1)
        new_score = self.score * score
        new_output = torch.cat([self.output, torch.tensor([token_id])], dim=-1) if self.output is not None else new_input_ids
        return Beam(self.size, new_input_ids, new_score, new_output)