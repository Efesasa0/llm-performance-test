from langchain_community.llms.ollama import Ollama
import json
import random
import time 


class Tester():
    def __init__(self, model_name, prompts, n_repeat):
        self.model_name = model_name
        self.prompts = prompts
        self.n_repeat = n_repeat

        self.results = {self.model_name:{}}
        for idx, prompt in enumerate(self.prompts):
            self.results[self.model_name][f"prompt_{idx}"]={"response_ls":[],
                                                              "time_ls":[],
                                                              "prompt_text":prompt}
            
    def warmup(self, llm, n_warmup=2):
        for _ in range(n_warmup):
            prompt = random.choice(self.prompts)
            _ = llm.invoke(prompt)
        return
    
    def run_tests(self):
        llm = Ollama(model=self.model_name)
        _ = self.warmup(llm)
        for idx, prompt in enumerate(self.prompts):
            for _ in range(self.n_repeat):
                start = time.time()
                response = llm.invoke(prompt)
                end = time.time()
        
                path = self.results[self.model_name][f"prompt_{idx}"]
                path["response_ls"].append(response)
                path["time_ls"].append(end - start)
            path['mean_time'] = sum(path['time_ls']) / self.n_repeat
            path['max_time'] = max(path['time_ls'])
            path['min_time'] = min(path['time_ls'])
        return self.results

def main():
    model_names = ["hal0000:latest",
"hal1000:latest",
"hal2000:latest",
"hal3000:latest",
"hal4000:latest",
"hal5000:latest",
"hal6000:latest",
"hal7000:latest",
"hal8000:latest",
"hal9000:latest",
"hal10000:latest",
                   ]
    with open("prompts.json", "r") as file:
        data = json.load(file)
    prompts = data['prompts']
    n_repeat = 2
    stats = {}
    for model in model_names:
        print(f"looking at model: {model}")
        tester = Tester(model_name=model, prompts=prompts, n_repeat=n_repeat)
        results = tester.run_tests()
        print(results)
        print("~"*10)
        for key, val in results.items():
            stats[model]=results
    return stats
    

if __name__ == "__main__":
    stats = main()

    with open("ollama_out.json", 'w') as file:
        json.dump(stats, file, indent=4)
    print("#"*10)
    print(stats)