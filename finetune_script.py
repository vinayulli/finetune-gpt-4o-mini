from openai import OpenAI
from datasets import load_dataset
from dotenv import load_dotenv
import random
import json

load_dotenv()

# load the dataset
dataset = load_dataset("Amod/mental_health_counseling_conversations", split = 'train')

# preparing dataset
system_prompt = "You serve as a supportive and honest psychology and psychotherapy assistant. Your main duty is to offer compassionate, understanding, and non-judgmental responses to users seeking emotional and psychological assistance. Respond with empathy and exhibit active listening skills. Your replies should convey that you comprehend the user’s emotions and worries. In cases where a user mentions thoughts of self-harm, suicide, or harm to others, prioritize their safety. Encourage them to seek immediate professional help and provide emergency contact details as needed. It’s important to note that you are not a licensed medical professional. Refrain from diagnosing or prescribing treatments. Instead, guide users to consult with a licensed therapist or medical expert for tailored advice. Never store or disclose any personal information shared by users. Uphold their privacy at all times. Avoid taking sides or expressing personal viewpoints. Your responsibility is to create a secure space for users to express themselves and reflect. Always aim to foster a supportive and understanding environment for users to share their emotions and concerns. Above all, prioritize their well-being and safety."
sampled_dataset = random.choices(dataset, k=100)
train_dataset = []

for row in sampled_dataset:
    user_content = row['Context']
    system_content = system_prompt
    assistant_content = row['Response']
    message =  {'messages':[{'role':'system','content': system_content}, {'role':'user','content':user_content}, {'role':'assistant','content':assistant_content}]}
    
    train_dataset.append(message)

# Save data in JSONl format 
def save_to_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for row in data:
            line = json.dumps(row)
            file.write(line + '\n')


# Store the data in JSONL format
training_data_path = 'train.jsonl'
save_to_jsonl(train_dataset[:-10], training_data_path)

validation_data_path = 'validation.jsonl'
save_to_jsonl(train_dataset[-10:], validation_data_path)

# load the data from jsonl files 
training_data = open(training_data_path,"rb")
validation_data = open(validation_data_path,"rb")


# Upload the training and validation files
client = OpenAI()
training_response = client.files.create(file=training_data,purpose="fine-tune")
training_file_id = training_response.id
validation_response = client.files.create(file=validation_data,purpose="fine-tune")
validation_file_id = validation_response.id

# Create a fine-tuning job
response = client.fine_tuning.jobs.create(
    training_file = training_file_id,
    model = "gpt-4o-mini-2024-07-18",
    suffix = "mental-health-chatbot",
    validation_file = validation_file_id
)

job_id = response.id
print(response)