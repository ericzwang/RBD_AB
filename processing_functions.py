import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from copy import deepcopy

aa_map = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,
          'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,
          'X':0, '-':0}

class_residues = {1:[405,417,420,421,452,455,456,460,472,473,475,476,484,486,487,489,493,504],
                  2:[452,455,456,472,473,483,484,485,486,489,490,493,494],
                  3:[346,439,440,443,444,445,446,447,448,449,450,452,490,494,496,498,499,500],
                  4:[369,374,376,378,384,396,408,417,462,504,514,516,518]}

factor_dict = {0.00:1.00, 0.01:1.01, 0.02:1.02, 0.03:1.02, 0.04:1.02,
               0.05:1.03, 0.06:1.04, 0.07:1.04, 0.08:1.04, 0.09:1.05,
               0.10:1.06, 0.11:1.06, 0.12:1.06, 0.13:1.07, 0.14:1.08,
               0.15:1.08, 0.16:1.08, 0.17:1.09, 0.18:1.10, 0.19:1.11,
               0.20:1.12, 0.21:1.13, 0.22:1.14, 0.23:1.14, 0.24:1.14,
               0.25:1.15, 0.26:1.16, 0.27:1.17, 0.28:1.18, 0.29:1.18,
               0.30:1.18, 0.31:1.19, 0.32:1.20, 0.33:1.21, 0.34:1.22,
               0.35:1.23, 0.36:1.24, 0.37:1.25, 0.38:1.26, 0.39:1.27,
               0.40:1.28, 0.41:1.29, 0.42:1.30, 0.43:1.31, 0.44:1.32,
               0.45:1.33, 0.46:1.34, 0.47:1.35, 0.48:1.36, 0.49:1.37,
               0.50:1.38, 0.51:1.40, 0.52:1.42, 0.53:1.43, 0.54:1.44,
               0.55:1.45, 0.56:1.46, 0.57:1.48, 0.58:1.50, 0.59:1.51,
               0.60:1.52, 0.61:1.54, 0.62:1.56, 0.63:1.58, 0.64:1.60,
               0.65:1.62, 0.66:1.64, 0.67:1.66, 0.68:1.68, 0.69:1.70,
               0.70:1.72, 0.71:1.74, 0.72:1.76, 0.73:1.79, 0.74:1.82,
               0.75:1.85, 0.76:1.88, 0.77:1.91, 0.78:1.94, 0.79:1.98,
               0.80:2.02, 0.81:2.06, 0.82:2.10, 0.83:2.14, 0.84:2.18,
               0.85:2.23, 0.86:2.28, 0.87:2.34, 0.88:2.40, 0.89:2.48,
               0.90:2.56, 0.91:2.65, 0.92:2.74, 0.93:2.87, 0.94:3.00,
               0.95:3.20, 0.96:3.40, 0.97:3.70, 0.98:4.00, 0.99:6.00,
               1.00:8.00}

rbd = ('NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDL'
       'CFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYN'
       'YLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRV'
       'VVLSFELLHAPATVCGPKKST')

def get_abclass(df):
    """ Get one hot encoding of antibody class from dataframe """
    return F.one_hot(torch.tensor(df['subtype'].values.astype(int)).long()-1, num_classes=4)

def get_escapes(df):
    """ Get log transformed escape fractions from dataframe """
    return torch.log(torch.tensor(df['mut_escape'].to_numpy()))

def padded_seq(chain, df):
    """
    Pad sequence with zeroes up to max length
    
    Inputs:
    chain - list of chain names, used to access dataframe columns
    df - dataframe
    
    Outputs:
    padded - padded sequence
    """
    maxlens = {'heavy_variable':172, 'light_variable':125, 'rbd':200}
    temp = [torch.tensor([aa_map[aa] for i, aa in enumerate(seq)]) for seq in df[chain]]
    padded = nn.utils.rnn.pad_sequence(temp, batch_first=True, padding_value=0.0)[:,:maxlens[chain]]
    return padded

def shuffle_tensors(seqs_unshuffled, abclass_unshuffled, escapes_unshuffled):
    """
    Shuffle sequence tensors, abclass tensors, and escape fraction tensors
    
    Inputs:
    seqs_unshuffled - list of unshuffled protein sequences
    abclass_unshuffled - unshuffled antibody class tensors
    escapes_unshuffled - unshuffled escape fraction tensors
    
    Outputs:
    seqs - shuffled protein sequences
    abclass - shuffled antibody class tensors
    escapes - shuffled escape fraction tensors
    """
    seqs = deepcopy(seqs_unshuffled)
    abclass = deepcopy(abclass_unshuffled)
    escapes = deepcopy(escapes_unshuffled)
    shuffled = np.arange(abclass.shape[0]); np.random.shuffle(shuffled)
    for i in range(len(seqs)): seqs[i] = seqs[i][shuffled]
    abclass, escapes = abclass[shuffled], escapes[shuffled]
    return seqs, abclass, escapes

def get_tensors(df, chains):
    """
    Pad sequences, get antibody class and escape fraction tensors from dataframe, and shuffle.
    
    Inputs:
    df - dataframe
    chain - list of chain names, used to access dataframe columns
    
    Outputs:
    list of protein sequence, antibody class, and escape fraction tensors
    """
    seqs = [padded_seq(chain, df) for chain in chains]
    abclass, escapes = get_abclass(df), get_escapes(df)
    seqs, abclass, escapes = shuffle_tensors(seqs, abclass, escapes)
    return [seqs, abclass, escapes]

def get_tensors_batched(df, chains, batchsize):
    """
    Pad sequences, get antibody class and escape fraction tensors from dataframe,
    shuffle, and split into batches.
    
    Inputs:
    df - dataframe
    chain - list of chain names, used to access dataframe columns
    batchsize - batch size
    
    Outputs:
    list of protein sequences (heavy, light, rbd), antibody class, and escape fraction tensors in batches
    """
    seqs = [padded_seq(chain, df) for chain in chains]
    abclass, escapes = get_abclass(df), get_escapes(df)
    seqs, abclass, escapes = shuffle_tensors(seqs, abclass, escapes)
    for i in range(len(seqs)):
        seqs[i] = torch.split(seqs[i], batchsize)
    abclass, escapes = torch.split(abclass, batchsize), torch.split(escapes, batchsize)
    return [(x1,x2,x3,x4,x5) for x1,x2,x3,x4,x5 in zip(seqs[0],seqs[1],seqs[2],abclass,escapes)]

def get_tensors_noshuffle(df, chains):
    """ Pad sequences and get antibody class and escape fraction tensors from dataframe
    Inputs:
    df - dataframe
    chain - list of chain names, used to access dataframe columns
    
    Outputs:
    list of protein sequence, antibody class, and escape fraction tensors
    """
    seqs = [padded_seq(chain, df) for chain in chains]
    abclass, escapes = get_abclass(df), get_escapes(df)
    return [seqs, abclass, escapes]

def shuffle_batch(seqs_unshuffled, abclass_unshuffled, escapes_unshuffled, batchsize):
    """ Shuffle tensors and split into batches
    
    Inputs:
    seqs_unshuffled - list of unshuffled protein sequences
    abclass_unshuffled - unshuffled antibody class tensors
    escapes_unshuffled - unshuffled escape fraction tensors
    
    Outputs:
    list of protein sequences (heavy, light, rbd), antibody class, and escape fraction tensors in batches
    """
    seqs, abclass, escapes = shuffle_tensors(seqs_unshuffled, abclass_unshuffled, escapes_unshuffled)
    for i in range(len(seqs)):
        seqs[i] = torch.split(seqs[i], batchsize)
    abclass, escapes = torch.split(abclass, batchsize), torch.split(escapes, batchsize)
    return [(x1,x2,x3,x4,x5) for x1,x2,x3,x4,x5 in zip(seqs[0],seqs[1],seqs[2],abclass,escapes)]

def get_positions(x):
    """ For a vector x, fill entries with index position. 0s are filled with 0. """
    return torch.where(x>0, torch.arange(x.shape[1])+1, 0)

def mask_rbd(seqs, abclass, sites, p=0.5):
    """
    Apply mask on rbd sequence tensors. Residues randomly become 0.
    
    Inputs:
    seqs - list of protein sequence tensors, this gets mutated
    abclass - antibody class tensor
    sites - pandas series containing mutated residues
    p - probability of being 0 
    
    Outputs:
    None
    """
    for i, seq in enumerate(seqs[2]):
        abclass_i = int(np.argmax(abclass[i])) + 1
        cls_lst = class_residues[abclass_i]
        tmp = sites.iloc[i]
        if isinstance(tmp, (int, np.integer)):
            tmp = [tmp]
            
        tmp_tensor = []
        for j, aa in enumerate(seq):
            cond1 = ((j+331) in cls_lst)
            cond2 = ((j+331) not in cls_lst and np.random.uniform() > p)
            cond3 = ((j+331) in tmp)
            if (cond1 or cond2 or cond3):
                tmp_tensor.append(aa)
            else:
                tmp_tensor.append(0)
        seqs[2][i] = torch.tensor(tmp_tensor)

def mask_ab(seqs, p=0.7, factor_dict=factor_dict):
    """ 
    Apply mask on antibody sequence tensors. Residues randomly become 0.
    
    Inputs:
    seqs - list of protein sequence tensors, this gets mutated
    p - probability of being 0
    factor_dict - dictionary of correspondence between p and factor that increases
        the number of 0s. This is because of overlap when randomly selecting entries.
    
    Outputs:
    None
    """
    for k in range(2):
        arr = np.array(seqs[k])
        num_replaced = int(arr.size*p*factor_dict[p])
        indx = np.random.randint(0, arr.shape[0], num_replaced)
        indy = np.random.randint(0, arr.shape[1], num_replaced)
        seqs[k][indx, indy] = 0

def getprediction(df, chains, p1, device, model):
    """
    Get tensors, apply mask on rbd sequences, and evaluate model
    
    Inputs:
    df - dataframe
    chain - list of chain names, used to access dataframe columns
    p1 - probability of rbd residue being masked
    device - cuda for gpu, otherwise cpu
    model - neural network model
    
    Outputs:
    prediction - output of model
    escapes - log escape fractions
    """
    seqs, abclass, escapes = get_tensors_noshuffle(df, chains)
    sites = df['site']
    mask_rbd(seqs, abclass, sites, p=p1)

    heavyp, lightp, rbdp = [get_positions(seq) for seq in seqs]
    heavys, lights, rbds = seqs
    heavys = heavys.to(device); heavyp = heavyp.to(device)
    lights = lights.to(device); lightp = lightp.to(device)
    rbds = rbds.to(device); rbdp = rbdp.to(device)
    escapes = escapes.to(device).cpu().detach().numpy()

    prediction = model(heavys, heavyp, lights, lightp, rbds, rbdp).flatten().cpu().detach().numpy()
    tmp = pd.DataFrame([prediction, escapes]).transpose().drop_duplicates(subset=1).to_numpy()
    prediction, escapes = tmp.T
    return prediction, escapes

def makerbdvar(variant, rbd=rbd):
    """ Create a variant RBD sequence using mutations and WT sequence """
    mutdict = {'wildtype':[],
               'Alpha':['N501Y'],
               'Beta':['K417N','E484K','N501Y'],
               'Gamma':['K417T','E484K','N501Y'],
               'Delta':['L452R','T478K'],
               'Omicron':['G339D', 'S371L', 'S373P', 'S375F', 'K417N', 'N440K', 'G446S',
                          'S477N', 'T478K', 'E484A', 'Q493R', 'G496S', 'Q498R', 'N501Y',
                          'Y505H'],
               'NKY':['K417N', 'E484K', 'N501Y'],
               'KKY':['K417K', 'E484K', 'N501Y'],
               'NEY':['K417N', 'E484E', 'N501Y'],
               'NKN':['K417N', 'E484K', 'N501N']}

    rbdnew = deepcopy(rbd)
    residues = []
    for mutation in mutdict[variant]:
        residue = int(mutation[1:-1])
        site = residue - 331
        mutant = mutation[-1]
        rbdnew = rbdnew[:site] + mutant + rbdnew[site+1:]
        residues.append(residue)
    return rbdnew, residues    

def getreinckedf(chains_original, df):
    """ Get dataframe for Kd measurements from Reincke et al.
    Inputs:
    chain - list of chain names, used to access dataframe columns
    df - dataframe
    
    Outputs:
    dataframe of Kd measurements with protein sequences
    """
    chains = deepcopy(chains_original); chains.remove('rbd')
    reincke_ab = pd.read_csv('reincke_2022/tables3_withseqs.csv')
    kds = pd.read_csv('reincke_2022/fig4j.csv')
    kds_ab = kds.drop(columns='strain')
    
    chains_list = ['antibody'] + chains
    tmp = df[df['antibody'].isin(kds_ab.columns)][chains_list].drop_duplicates()
    tmp2 = reincke_ab[reincke_ab['antibody'].isin(kds_ab.columns)][chains_list]
    abseqs = pd.concat([tmp, tmp2])

    reincke_df = []
    for i, row in kds.iterrows():
        strain = row['strain']
        rbdvar, residues = makerbdvar(strain)
        for ab in kds_ab.columns:
            heavys, lights = abseqs[abseqs['antibody']==ab].iloc[0][chains]
            mut_escape = 1 / kds[kds['strain']==strain][ab].item()
            subtype = 4 if ab == 'CR3022' else 1
            reincke_df.append([ab, strain, rbdvar, heavys,
                               lights, mut_escape, residues, subtype])
            
    columns_dict = {0:'antibody', 1:'strain', 2:'rbd', 3:chains[0], 4:chains[1],
                    5:'mut_escape', 6:'site', 7:'subtype'}
    reincke_df = pd.DataFrame(reincke_df).rename(columns=columns_dict)
    
    return reincke_df[reincke_df['strain']!='wildtype']

def bootstrap_se(x, func, access):
    """
    Bootstrap standard error from a matrix x.
    
    Inputs:
    x - Nx2 matrix
    func - function to calculate desired quantity
    access - lambda function to access field of output. Ex: lambda x: x.correlation
    
    Outputs:
    se - standard error
    """
    sampled = [x[np.random.choice(x.shape[0], x.shape[0], replace=True)] for _ in range(10000)]
    se = np.std([access(func(sample[:,0], sample[:,1])) for sample in sampled])
    return se


