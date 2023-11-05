import torch
import pickle
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer,GPTNeoXForCausalLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_generation(input_text, model, tokenizer):
    print(f"SAMPLE::\n {input_text}")
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, temperature=0)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"SAMPLE::\n {input_text} \n Output : {output_text}")
    return output_text


def main(dataset_prompts, debug = True):
    num_samples_train = int(dataset_prompts["train"].num_rows * 0.6)
    num_samples = dataset_prompts["train"].num_rows
    if debug:
        num_samples_train = 100
        num_samples = 150
    model_checkpoints = ["skrishna/pythia-70m-non-toxic", "EleutherAI/pythia-70m", "random"]
    dataset_destinations = ["datasets/non_toxic", "datasets/toxic", "datasets/random"]
    for ind, model_chkp in enumerate(model_checkpoints):
        if model_chkp != "random":
            model = GPTNeoXForCausalLM.from_pretrained(model_chkp)
            tokenizer = AutoTokenizer.from_pretrained(model_chkp)
            out_samples = []
            for sample_id in range(num_samples_train):
                input_sample = dataset_prompts["train"][sample_id]["prompt"]["text"]
                output_sample = get_generation(input_sample, model, tokenizer)
                out_samples.append([input_sample, output_sample])
            pickle.dump(out_samples, open(dataset_destinations[ind] +"_train.pkl", "wb"))

            for sample_id in range(num_samples_train, num_samples):
                input_sample = dataset_prompts["train"][sample_id]["prompt"]["text"]
                output_sample = get_generation(input_sample, model, tokenizer)
                out_samples.append([input_sample, output_sample])
            pickle.dump(out_samples, open(dataset_destinations[ind] +"_test.pkl", "wb"))
    pass

if __name__ == "__main__":
    dataset_toxicity = datasets.load_dataset("allenai/real-toxicity-prompts")
    main(dataset_toxicity)
    








