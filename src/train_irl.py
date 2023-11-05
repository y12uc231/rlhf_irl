
import os 
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
import datasets
import torch
from torch import nn
import pickle
import numpy as np
import pandas as pd 

class RewardModel(nn.Module):
    def __init__(self, checkpoint_path, eos_token_id):
        super().__init__()
        # model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        model = GPTNeoXForCausalLM.from_pretrained(checkpoint_path)
        self.model = model
        self.v_head = nn.Linear(model.embed_out.out_features, 1, bias=False)  # TODO make not magic number
        # self.eos_token_id = eos_token_id
        self.eos_token_id = eos_token_id
    def forward(self, input_ids):
        returns = torch.mean(self.model(input_ids)[0], 1)
        returns_2 = self.v_head(returns).squeeze(-1)
        print("Returns : ", returns.shape, returns_2)
        return returns_2

    


def get_dataset_train(policy_name = "skrishna/pythia-70m-non-toxic"):
    """
       policy_name =  ["skrishna/pythia-70m-non-toxic", "EleutherAI/pythia-70m", "random"]
    """
    if policy_name == "skrishna/pythia-70m-non-toxic":
        dataset_samples = pickle.load(open("datasets/non_toxic_train.pkl", "rb"))
        return dataset_samples
    elif policy_name == "EleutherAI/pythia-70m":
        dataset_samples = pickle.load(open("datasets/toxic_train.pkl", "rb"))
        return dataset_samples
    pass

def get_initial_model(model_size = "70M"):
    """
    Returns initial reward model. 
    """
    if model_size == "70M":
        reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        reward_model = RewardModel("EleutherAI/pythia-70m", reward_tokenizer.eos_token_id)
        return reward_model.requires_grad_(), reward_tokenizer


def get_irl_loss(loss_type = "max_margin"):
    """
    Returns loss function for optimization. 
    """
    if loss_type == "max_margin":
        def loss(reward_n_t, reward_t):
            def custom_operator(input_vector):
                # Create masks for positive and negative values
                positive_mask = input_vector > 0
                negative_mask = input_vector < 0
                # Multiply positive values by -1 and negative values by -2
                input_vector[positive_mask] *= -1
                input_vector[negative_mask] *= -2
                return input_vector
            reward_diff = reward_n_t - reward_t
            reward_diff_dir = custom_operator(reward_diff)
            return reward_diff_dir
        return loss

def get_optimizer(model, optim_type = "Adam", lr = 0.00001, momentum = 0.9):
    """
    Select and return suitable optimizer with training hyper-params. 
    """
    if optim_type == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        pass

def get_evaluation(model):
    pass

def get_policy_outputs(model, sample):
    """
     -- NTBU -- 
    """
    pass

def get_reward_score(reward_model, input_text, tokenizer):
    """
    Takes reward model and input and returns reward score. 
    """
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = reward_model(input_ids)
    return output

def data_loader(list1, list2, batch_size=2):
    assert len(list1) == len(list2), "Both lists should have the same length"
    
    for i in range(0, len(list1), batch_size):
        batch1 = [row[-1] for row in list1[i:min(i+batch_size, len(list1))]]
        batch2 = [row[-1] for row in list2[i:min(i+batch_size, len(list2))]]
        yield batch1, batch2


def irl_function(num_epochs = 100, current_directory = "./models", sub_model_name = "reward_model" ):
    """
    Step 1 : Generate dataset with random policy 
    Step 2 : Initiate model with K million parameters
    Step 3 : Implement the loss function
    Step 4 : Evaluate reward model on toxicity.  
    """
    data_set_ntoxic = get_dataset_train(policy_name="skrishna/pythia-70m-non-toxic")
    data_set_toxic = get_dataset_train(policy_name="EleutherAI/pythia-70m")

    # Obtain training functions with hyper-params
    init_model, tokenizer = get_initial_model()
    loss = get_irl_loss()
    optimizer = get_optimizer(init_model)
    
    # Training loop
    for num_epoch in range(num_epochs):
        print(f"Processing Epoch {num_epoch}..")
        eval_metric = get_evaluation(init_model)
        print(f" Toxic Model Accuracy : {eval_metric}")
        for sample_n_toxic, sample_toxic in data_loader(data_set_ntoxic, data_set_toxic):
            policy_outputs_nt = get_reward_score(init_model, sample_n_toxic, tokenizer)
            policy_outputs_t = get_reward_score(init_model, sample_toxic, tokenizer)
            loss_value = loss(policy_outputs_nt, policy_outputs_t)
            print("Loss Value : ", torch.mean(loss_value))
            loss_value.backward()
            optimizer.step()
     
    # save trained reward model
    current_directory_repo = current_directory + f"/{sub_model_name}"
    model.save_pretrained(current_directory_repo)








if __name__ == "__main__":
    irl_function()

