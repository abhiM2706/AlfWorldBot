import pandas as pd
import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import yaml
import re
from openai import AzureOpenAI
from thefuzz import fuzz
from thefuzz import process


with open('/home/azureuser/cloudfiles/code/Users/abhinav/alfworld_config.yaml') as reader:
    config = yaml.safe_load(reader)

env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

print(env_type)


# setup environment
env = getattr(environment, env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

obs, info = env.reset()

setting = obs

tries = 0

client = AzureOpenAI(
    api_key=""***APIKey***"",
    api_version="2024-02-01",
    azure_endpoint="https://nsg-gpt4-vision-instance.openai.azure.com/"
)

i = 0
cnt=0
COLUMN_NAMES=['Trial Number','Win/Lose','Tries', 'Prompt']
df = pd.DataFrame(columns = COLUMN_NAMES)
while(i<10):
    print("\n\n\n"+str(i)+"\n\n\n")
    
    obs, infos = env.reset()
    setting = obs
    inventory=""
    admissible_commands = list(infos['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
    rules = str(setting[0])
    print (rules)
    promptInd = rules.find("Your task")
    prompt = rules[promptInd:]
    commands=""
    lastAction=False
    #print(rules)
    tries = 0

    messageList = [
        {"role": "system", "content":"""You are completing a task provided by the user. Follow the user's instructions to find and place an object in a specified area. Use only the valid commands listed. When you find the correct item, pick it up first before placing it. Read the prompt: """ + prompt + """ after each move. Use only commands from the list. Double-check the command is valid. Items that need to be cooled, heated, or cleaned are not yet in that state. Clean items in the sink before placing them if instructed.

    Think logically about the object's location and what to do with it. Use only the provided commands. Follow prompt clues, focusing on your task. Ignore other items in the area. Only put one command at a time. explain your thoughts before making a calculated move in brackets. If you get stuck in a loop, try different things

Explain your reasoning step by step in regualar text no brackets before issuing a command in brackets (e.g., [go to cabinet 1]). This command in brackets should be the last thing you output. Ensure you know how many objects are needed and retrieve them all. The 'look' and 'examine' commands are useless.

Before you issue one command, look at the list of commands: """ + str(admissible_commands) + """. Specify the final action clearly (e.g., put the candle in/on the toilet). Use trial and error and common sense. Before heating cooling or cleaning make sure u use inventory to see if you even have the object. Sometimes you use commands that are not the most efficent in the context of the prompt so make sure u read through all the allowed commands before making a choice. Learn to ignore certain things if they cause you to go in a loop and try things that may be considered unusual in real life

Use 'inventory' after picking up an item to make sure u have it check your items and avoid repetition. Commands should only be from the list of commands above shown after "Commands:" ."""},
        {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": "This is the current state of the game: "+setting[0]+"You can open cabinets, drawers, etc., even if they are locked.Don't assume something is clean or heated until you have cleaned in the sink or microwaved it"
                    }
        ]}

    ]



    while(tries<50):
        admissible_commands = list(infos['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
        admissible_commands = str(admissible_commands)

        #print(admissible_commands)

        #print("\n\n"+str(messageList)+"\n\n")
        response = client.chat.completions.create(
            model = 'gpt-4o',
            messages = messageList
        )
        

        input = response.choices[0].message.content
        print("AI: " + str(input))
        x = re.findall("\[(.*)\]",input)
        if(len(x)==0):
            input = " "
        else:
            input = x[-1]

        #print(input)
        compare = list(infos['admissible_commands'])
        wordRatio = process.extractOne(input,compare[0])
        #print("\n\n"+str(wordRatio[0])+"\n\n")


        messageList.append({"role": "assistant", "content": response.choices[0].message.content})




        action = [str(wordRatio[0])]
        obs, scores, dones, infos = env.step(action)

        messageList.append({"role": "user", "content":f"step {tries+1}: {action}, {obs}, {admissible_commands}"})
        #print(action)
        setting = obs
        print("Action: {}, Obs: {}".format(str(wordRatio[0]), obs[0]))
        tries = tries+1
        #print(dones[0])
        #print(scores)
        loopEnd = dones[0]
        if(infos["won"][0]==True):
            print(f"Tries {tries}")
            print(infos["won"][0])
            break
        commands = commands+input+", "
        
    print("\n\n\nTask has been completed:\n\n")
        #print(commands)

    num = scores[0]
    x = False
    if(num==1):
        cnt=cnt+1
        x = True
    df2 = {'Trial Number': str(i), 'Win/Lose': str(x), 'Tries': str(tries), 'Prompt': prompt} 
    df2 = pd.DataFrame([df2])
    #print("\n\n"+df.to_string()+"\n\n"+df2.to_string()+"\n\n")
    df = df.append(df2, ignore_index=True)
    i=i+1
    
print("\n\nTesting Complete:\n\n"+df.to_string()+"\n\n")



print("\n\nAccuracy: "+ str(float(cnt)/(i)*100)+"%")
