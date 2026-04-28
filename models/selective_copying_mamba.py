import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import csv
import os
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from models.configs.selective_copying.config import training_config, dataset_config, MambaConfig
from models.dataloaders.selective_copying import generate_dataset

# Training function
def train(save_path, log_dir="/scratch/lhl1g23/lightning_logs/selective_copying/"):
    """
    Train the model.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    os.makedirs(log_dir, exist_ok=True)
    csv_file = open(log_dir + 'training_log.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'train_acc', 'val_acc'])

    # Define model
    mambaconfig = MambaConfig()
    model = MambaLMHeadModel(mambaconfig, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])
    
    model.train()
    start_time = time.time()
    
    train_inputs, train_targets = generate_dataset(dataset_config, training_config)
    test_inputs, test_targets = generate_dataset(dataset_config, training_config)
    train_inputs, train_targets, test_inputs, test_targets = train_inputs.to(device), train_targets.to(device), test_inputs.to(device), test_targets.to(device)

    for step in range(training_config["num_steps"]):
        step_loss = 0
        correct = 0
        total = 0
        
        outputs = model(train_inputs, num_last_tokens=dataset_config['l_memorize']).logits
        loss = criterion(outputs, train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss += loss.item()
        
        if step % 100 == 0:
            total += train_targets.size(0) * train_targets.size(1)
            correct += (outputs.argmax(1) == train_targets).sum().item()
            train_acc = 100 * correct / total
            
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_inputs, num_last_tokens=dataset_config['l_memorize']).logits
                test_acc = (test_outputs.argmax(1) == test_targets).sum() / test_targets.numel()
            model.train()
            
            logger.info(f'Step [{step+1}/{training_config["num_steps"]}], Loss: {step_loss/training_config["batch_size"]:.4f}, Accuracy: {train_acc:.2f}%')
            writer.writerow([step, train_acc, test_acc.item()])
            csv_file.flush()
        
    csv_file.close()

    end_time = time.time()
    logger.info(f'Training completed in: {(end_time - start_time)/60:.2f} minutes')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    model = train("/scratch/lhl1g23/mamba/checkpoints/selective_copying_new")