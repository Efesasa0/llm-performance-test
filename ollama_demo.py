from langchain_community.llms.ollama import Ollama
import json
import random
import time 


class Tester():
    def __init__(self, model_name, prompts, n_repeat):
        self.model_name = model_name
        self.prompts = prompts
        self.prompt_texts = []
        self.prompt_ids = []
        self.n_repeat = n_repeat
        self.prompt_responses = []
        self.prompt_times = []
            
    def warmup(self, llm, n_warmup=1):
        for _ in range(n_warmup):
            prompt = random.choice(self.prompts)
            _ = llm.invoke(prompt)
        return
    
    def run_tests(self):
        llm = Ollama(model=self.model_name)
        _ = self.warmup(llm)
        for idx, prompt in enumerate(self.prompts):
            self.prompt_texts += [prompt for _ in range(self.n_repeat)]
            self.prompt_ids += [f"prompt_{idx}" for _ in range(self.n_repeat)]
            for _ in range(self.n_repeat):
                start = time.time()
                response = llm.invoke(prompt)
                end = time.time()
                self.prompt_responses.append(response)
                self.prompt_times.append(end - start)
        return self.prompt_texts, self.prompt_ids, self.prompt_responses, self.prompt_times

def main(model_names):

    with open("prompts.json", "r") as file:
        data = json.load(file)
    prompts_ls = data['prompts']
    n_repeat = 5
    stats = {"model_names":[],
             "prompt_ids":[],
             "prompt_texts":[],
             "prompt_responses":[],
             "prompt_times":[]}
    for model in model_names:
        print(f"looking at model: {model}")
        tester = Tester(model_name=model, prompts=prompts_ls, n_repeat=n_repeat)
        prompts, prompt_ids, responses, times = tester.run_tests()
        stats["model_names"] += [model] * n_repeat * len(prompts_ls)
        stats["prompt_ids"] += prompt_ids
        stats["prompt_texts"] += prompts
        stats["prompt_responses"] += responses
        stats["prompt_times"] += times
        print(stats)
        print("~"*10)
    return stats

if __name__ == "__main__":

    model_names = [
        "hal0000:latest",
        "hal1000:latest",
        "hal2000:latest",
        "hal3000:latest",
        "hal4000:latest",
        "hal5000:latest",
        "hal6000:latest",
        "hal7000:latest",
        "hal8000:latest",
        "hal9000:latest",
        "hal10000:latest"
        ]
    
    stats = main(model_names)

    with open("ollama_out.json", 'w') as file:
        json.dump(stats, file, indent=4)
    print("#"*10)
    print(stats)