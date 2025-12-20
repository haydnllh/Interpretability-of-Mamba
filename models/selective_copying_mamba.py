import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from models.configs.selective_copying.config import training_config, dataset_config, MambaConfig
from models.dataloaders.selective_copying import generate_dataset

# Training function
def train(save_path):
    """
    Train the model.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Define model
    mambaconfig = MambaConfig()
    model = MambaLMHeadModel(mambaconfig, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])
    
    model.train()
    start_time = time.time()
    for step in range(training_config["num_steps"]):
        step_loss = 0
        correct = 0
        total = 0
        inputs, targets = generate_dataset(dataset_config, training_config)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, num_last_tokens=dataset_config['l_memorize']).logits
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss += loss.item()
        total += targets.size(0) * targets.size(1)
        correct += (outputs.argmax(1) == targets).sum().item()
        accuracy = 100 * correct / total
        logger.info(f'Step [{step+1}/{training_config["num_steps"]}], Loss: {step_loss/training_config["batch_size"]:.4f}, Accuracy: {accuracy:.2f}%')

    end_time = time.time()
    logger.info(f'Training completed in: {(end_time - start_time)/60:.2f} minutes')
    
    torch.save(model.state_dict(), save_path)
    
    return model

# Validation function
def validate(model):
    model.eval()
    logger = logging.getLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        correct = 0
        total = 0
        inputs, targets = generate_dataset(dataset_config, training_config)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, num_last_tokens=dataset_config['l_memorize']).logits
        total += targets.size(0) * targets.size(1)
        correct += (outputs.argmax(1) == targets).sum().item()
        accuracy = 100 * correct / total
        logger.info(f'Validation Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    model = train("checkpoints/selective_copying")
    validate(model)
