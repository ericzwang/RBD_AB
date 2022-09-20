import gc
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from processing_functions import *
from copy import deepcopy


class NeuralNetwork(nn.Module):
    """
    Inputs:
    maxlen - maximum length of protein sequence, used for position embedding
    vocab_size - number of amino acids, used for token embedding
    embed_dim - dimension of embedding vectors
    num_heads - number of attention heads
    num_blocks - number of transformer blocks
    ff_dim - number of neurons in dense layers inside transformer block
    ff_dim? - number of neurons in dense layers 
    """
    def __init__(self,
                maxlen,
                vocab_size,
                embed_dim,
                num_heads,
                num_blocks,
                ff_dim,
                ff_dim2,
                ff_dim3,
                ff_dim4):
        super(NeuralNetwork, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ff_dim = ff_dim
        self.ff_dim2 = ff_dim2
        self.ff_dim3 = ff_dim3
        self.ff_dim4 = ff_dim4
        
        self.token_embedding = nn.Embedding(self.vocab_size,
                                            self.embed_dim,
                                            padding_idx=0)
        self.position_embedding = nn.Embedding(self.maxlen+1,
                                               self.embed_dim,
                                               padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(self.embed_dim,
                                                        self.num_heads,
                                                        dim_feedforward=self.ff_dim,
                                                        batch_first=True,
                                                        dropout=0.,
                                                        activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=self.num_blocks)
        
        self.linear1 = nn.Linear(self.embed_dim, self.ff_dim2)
        self.linear2 = nn.Linear(self.ff_dim2, 20)
        self.linear3 = nn.Linear(20*3, self.ff_dim3)
        self.linear4 = nn.Linear(self.ff_dim3, self.ff_dim4)
        self.outlinear = nn.Linear(self.ff_dim4, 1)
        
    def forward(self, heavys, heavyp, lights, lightp, rbds, rbdp, outtype='final'):
        """
        Execute neural network
        
        Inputs:
        heavys - ordinal encoding of heavy chain sequence (masked residues are 0)
        heavyp - position indices of heavy chain sequence (masked residues are 0)
        lights - ordinal encoding of light chain sequence (masked residues are 0)
        lightp - position indices of light chain sequence (masked residues are 0)
        rbds - ordinal encoding of rbd chain sequence (masked residues are 0)
        rbdp - position indices of rbd chain sequence (masked residues are 0)
        outtype - type of output. For example, if heavy then heavy chain embeddings are returned.
        
        Outputs:
        embeddings vectors or predicted log escape fraction
        """
        token_emb = self.token_embedding(heavys)
        pos_emb = self.position_embedding(heavyp)
        x = token_emb + pos_emb
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1) # global average pooling
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        if outtype == 'heavy':
            return x

        token_emb2 = self.token_embedding(lights)
        pos_emb2 = self.position_embedding(lightp)
        x2 = token_emb2 + pos_emb2
        x2 = self.transformer_encoder(x2)
        x2 = torch.mean(x2, dim=1)
        x2 = self.linear1(x2)
        x2 = F.relu(x2)
        x2 = self.linear2(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.3)
        if outtype == 'light':
            return x2

        token_emb3 = self.token_embedding(rbds)
        pos_emb3 = self.position_embedding(rbdp)
        x3 = token_emb3 + pos_emb3
        x3 = self.transformer_encoder(x3)
        x3 = torch.mean(x3, dim=1)
        x3 = self.linear1(x3)
        x3 = F.relu(x3)
        x3 = self.linear2(x3)
        x3 = F.leaky_relu(x3, negative_slope=0.3)
        if outtype == 'rbd':
            return x3

        x = torch.cat([x, x2, x3], dim=1)
        x = self.linear3(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        x = self.linear4(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        x = self.outlinear(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer, device, batch_print_interval=100):
    """
    Run one epoch for training neural network. Print out metrics
    
    Inputs:
    dataloader - dataloader for training data
    model - neural network model
    loss_fn - loss function
    optimizer - optimizer for gradient descent
    device - cuda for gpu, otherwise cpu
    batch_print_interval - interval for printing information about batch
    
    Outputs:
    model - neural network model after training
    """

    size = len(dataloader) * dataloader[0][0].shape[0]
    num_batches = len(dataloader)

    for batch, (heavys, lights, rbds, abclass, escapes) in enumerate(dataloader):
        escapes = torch.reshape(escapes.float(), (-1,1))
        heavyp, lightp, rbdp = [get_positions(seq) for seq in [heavys, lights, rbds]]
        heavys = heavys.to(device); heavyp = heavyp.to(device)
        lights = lights.to(device); lightp = lightp.to(device)
        rbds = rbds.to(device); rbdp = rbdp.to(device)
        escapes = escapes.to(device)

        pred = model(heavys, heavyp, lights, lightp, rbds, rbdp)

        loss = loss_fn(pred, escapes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % batch_print_interval == 0:
            loss, current = loss.item(), batch * len(heavys)
            pred_numpy = pred.flatten().cpu().detach().numpy()
            escapes_numpy = escapes.flatten().cpu().detach().numpy()
            corr = np.corrcoef(pred_numpy, escapes_numpy)[0,1]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] corr:{corr:>7f}")
            
    return model


def val_loop(dataloader, model, loss_fn, device):
    """
    Run model on validation data. Calculate loss and metrics.
    
    Inputs:
    dataloader - dataloader for validation data
    model - neural network model
    loss_fn - loss function
    device - cuda for gpu, otherwise cpu
    
    Outputs:
    val_loss - loss on validation data
    """
 
    num_batches = len(dataloader)
    size = num_batches * dataloader[0][0].shape[0]
    val_loss, corr = 0, 0

    with torch.no_grad():
        for batch, (heavys, lights, rbds, abclass, escapes) in enumerate(dataloader):
            escapes = torch.reshape(escapes.float(), (-1,1))
            heavyp, lightp, rbdp = [get_positions(seq) for seq in [heavys, lights, rbds]]
            heavys = heavys.to(device); heavyp = heavyp.to(device)
            lights = lights.to(device); lightp = lightp.to(device)
            rbds = rbds.to(device); rbdp = rbdp.to(device)
            escapes = escapes.to(device)

            pred = model(heavys, heavyp, lights, lightp, rbds, rbdp)
            val_loss += loss_fn(pred, escapes).item()
            pred_numpy = pred.flatten().cpu().detach().numpy()
            escapes_numpy = escapes.flatten().cpu().detach().numpy()
            corr += abs(np.corrcoef(pred_numpy, escapes_numpy)[0,1])

    val_loss /= num_batches
    corr /= num_batches
    print(f"Val. Error: \n Avg Corr: {(100*corr):>0.1f}%, Avg loss: {val_loss:>8f}")
    return val_loss


def savemodel(model, file):
    print("Saving model")
    torch.save(model.state_dict(), file)

    
def training(train_dataloader, val_dataloader, model, loss_fn, optimizer, patience, epochs,
             device, file, p1, p2, chains, batch_size, 
             keep_best_weights = True, batch_print_interval=100):
    
    """
    Run neural network training. On each new epoch, masked residues are applied. Training loop is run
    then validation loop is run, and results of validation loop are used to determine early stopping.
    
    Inputs:
    train_dataloader - dataloader for training data, already batched
    val_dataloader - dataloader for validation data, already batched
    model - neural network model
    loss_fn - loss function
    optimizer - optimizer for gradient descent
    patience - patience for early stopping
    epochs - max number of epochs
    device - cuda for gpu, otherwise cpu
    file - path to file for saving the model parameters
    p1 - probability of masking rbd residues
    p2 - probability of masking antibody residues
    chains - list of chain names, used to access dataframe columns
    batch_size - batch size
    keep_best_weights - whether to save best weights from early stopping
    batch_print_interval - interval for printing information about batch
    
    Outputs:
    best_model - neural network model with best weights
    best_loss - loss from neural network model with best weights
    """
    
    train_data = deepcopy(train_dataloader)
    trigger = 0
    prev_val_loss, best_loss, best_model = np.inf, np.inf, model
    
    seqs, abclass, escapes = get_tensors_noshuffle(train_data, chains)
    sites = train_data['site']
    mask_rbd(seqs, abclass, sites, p=p1)
    seqs_old = deepcopy(seqs)

    for epoch in range(epochs):
        gc.collect()
        torch.cuda.empty_cache()

        seqs = deepcopy(seqs_old)
        mask_ab(seqs, p=p2)
        train_dataloader = shuffle_batch(seqs, abclass, escapes, batch_size)

        print(f"Epoch {epoch+1}\n-------------------------------")
        model = train_loop(train_dataloader, model, loss_fn, optimizer, device,
                           batch_print_interval=batch_print_interval).to(device)

        #Early stopping
        val_loss = val_loop(val_dataloader, model, loss_fn, device)

        if (val_loss >= prev_val_loss) or (np.isnan(val_loss) is True):
            trigger += 1
            if trigger >= patience:
                break
        else:
            trigger = 0
        prev_val_loss = val_loss

        if keep_best_weights:
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
                savemodel(best_model, file)
        else:
            best_model = model

    print("Done!")
    return best_model, best_loss