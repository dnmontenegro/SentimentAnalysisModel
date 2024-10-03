import torch
from torch.utils.tensorboard import SummaryWriter
from data_loader import ReviewDataset
from torch.utils.data import DataLoader
import json
from lstm import LSTMModel
import torch.optim as optim
from torch import nn
import numpy as np
import time

def _save_checkpoint(ckp_path, model, epoches, global_step, optimizer):
    checkpoint = {'epoch': epoches,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, ckp_path)

def main():
    writer = SummaryWriter()
    gpu_id = 0
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda", gpu_id)
    else:
        device = torch.device('cpu')
    print('Device: ', device)
    
    # Parameter settings
    mode = 'train'
    Batch_size = 16
    n_layers = 1
    input_len = 150
    embedding_dim = 200
    hidden_dim = 50
    output_size = 1
    num_epoches = 1
    learning_rate = 0.005
    clip = 5
    load_cpt = False
    ckp_path = 'cpt/name.pt'
    embedding_matrix = None
    pretrain = False

    # Loading training and test data
    training_set = ReviewDataset('training_data.csv')
    training_generator = DataLoader(training_set, batch_size=Batch_size, shuffle=True, num_workers=1)
    test_set = ReviewDataset('testing_data.csv')
    test_generator = DataLoader(test_set, batch_size=Batch_size, shuffle=False, num_workers=1)

    # Read tokens and load pre-train embedding
    with open('tokens_dict.json', 'r') as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)

    # Import model and load model to device
    model = LSTMModel(vocab_size, output_size, embedding_dim, embedding_matrix, hidden_dim, n_layers, input_len, pretrain)
    model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=learning_rate)  
    loss_fun = nn.BCELoss()
    
    if load_cpt:
        print("**** Loading checkpoint ****")
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoches = checkpoint['epoch']
        
    print("---- Model training ----")
    global_step = 0
    if mode == 'train':
        model.train()
        for epoches in range(num_epoches):
            for x_batch, y_labels in training_generator:
                global_step += 1
                x_batch, y_labels = x_batch.to(device), y_labels.to(device)
                y_out = model(x_batch)
                loss = loss_fun(y_out, y_labels)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                if global_step%10==0:
                    writer.add_scalar("Loss/train", loss, global_step)

            print("**** Saving checkpoint ****")
            ckp_path = 'checkpoint/step_{}.pt'.format(global_step)
            _save_checkpoint(ckp_path, model, epoches, global_step, optimizer)
            writer.flush()	
    writer.close()	
            
    print("---- Model testing ----")
    total = 0
    accuracy_count = 0
    matrix = np.zeros((2,2))
    model.eval()
    with torch.no_grad():
        for x_batch, y_labels in test_generator:
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            y_out = model(x_batch)
            y_pred = torch.round(y_out)

            total += len(y_labels)
            accuracy_count += ((y_pred==y_labels).sum().item())

            for t, p in zip(y_labels.view(-1), y_pred.view(-1)):
                matrix[t.long(), p.long()] += 1
    
    accuracy = accuracy_count/total
    print("Testing accuracy: ", accuracy)
    print("Confusion matrix:")
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix]))

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("Running time: ", (time_end - time_start)/60.0, "mins")
    

