import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# Example input data
batch_size = 2
chunk_size = 3
seq_length = 5

# Simulating input_ids and attention_mask
input_ids = torch.randint(0, 100, (batch_size, chunk_size, seq_length))  # Replace with actual tokenized input
attention_mask = (input_ids != 0).long()  # Example attention mask

# Reshape input_ids and attention_mask
input_ids = input_ids.view(-1, seq_length)  # Shape: (batch * chunk, seq_length)
attention_mask = attention_mask.view(-1, seq_length)  # Shape: (batch * chunk, seq_length)

# Forward pass
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# Obtain logits
logits = outputs.logits  # Shape: (batch * chunk, num_classes)
print(logits.shape)
# Reshape logits back to (batch, chunk, num_classes) if needed
num_classes = logits.size(-1)
logits = logits.view(batch_size, chunk_size, num_classes)  # Shape: (batch, chunk, num_classes)

# Now you can process logits as needed
