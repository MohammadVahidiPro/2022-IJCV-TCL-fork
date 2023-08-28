import numpy as np
import torch
import argparse
import network
import loss
# from utils.save_model import save_model, save_model_10
from torch.utils import data
from sentence_transformers import SentenceTransformer
from EDA.augment import gen_eda
from evaluation import inference2
from utils import cluster_utils
import os
import itertools
import torch.nn as nn
import nlpaug.augmenter.word as naw #14, 15, 18, 23, 24, and 27
import functools
import time
from pathlib import Path
# import wandb as wb

# global_step = 0
# timer_log = {}
# def timer(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         results = func(*args, **kwargs)
#         dif = time.time() - start
#         dif_min = round(dif / 60, 2)
#         name = func.__name__
        
#         if global_step < 20:
#             print(f"## timer step {global_step} ## {name}: {dif_min} MIN")
#         # if global_step < 20 or global_step % 20:
#             # wb.log({f"t(m)/{name}": dif_min, f"t(s)/{name}": dif})
#         return results
#     return wrapper


def get_args_parser():
    parser = argparse.ArgumentParser("TCL for clustering", add_help=False)
    parser.add_argument(
        "--batch_size", default=256, type=int, help="Batch size per GPU"
    ) # 128
    parser.add_argument("--epochs", default=510, type=int)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--wb_mode", default="offline", type=str)

    # Model parameters
    parser.add_argument("--feature_dim", default=128, type=int, help="dimension of ICH")
    parser.add_argument(
        "--instance_temperature",
        default=0.5,
        type=float,
        help="temperature of instance-level contrastive loss",
    )
    parser.add_argument(
        "--cluster_temperature",
        default=1.0,
        type=float,
        help="temperature of cluster-level contrastive loss",
    )

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=5e-6,
        help="learning rate of backbone",
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=5e-4,
        help="learning rate of head",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_dir", default="./datasets/", type=str, help="dataset path",
    )
    parser.add_argument(
        "--dataset",
        default="StackOverflow",
        type=str,
        help="dataset",
        choices=["StackOverflow", "Biomedical", "SearchSnippets"],
    )
    parser.add_argument(
        "--class_num", default=20, type=int, help="number of the clusters",
    )
    parser.add_argument(
        "--model_path",
        default="save/StackOverflow/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--resume",
        default="False",
        help="resume from checkpoint",
    )
    parser.add_argument("--check_id", type=str, default="this")
    parser.add_argument(
        "--start_epoch", default=0, type=int, help="start epoch"
    )
    parser.add_argument("--save_freq", default=50, type=int, help="saving frequency")
    parser.add_argument("--num_workers", default=8, type=int) # 10

    return parser

def update_args(args):
    if args.dataset == "Biomedical":
        args.class_num = 20
        args.model_path = "save/Biomedical/"
    elif args.dataset == "SearchSnippets":
        args.class_num = 8
        args.model_path = "save/SearchSnippets/"
    elif args.dataset == "StackOverflow":
        args.class_num = 20
        args.model_path = "save/Stackoverflow/"
    else:
        raise NotImplementedError
    
    
    embed_path = Path(args.model_path).resolve()  / 'embed'
    label_path = Path(args.model_path).resolve()  / 'embed'
    checkpoint_path = Path(args.model_path).resolve() / 'checkpoints'
    best_check_path = Path(args.model_path).resolve() / "checkpoints" / "best_model_avg.tar"
    embed_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    return best_check_path, checkpoint_path ,embed_path, label_path
    
class DatasetIterater(data.Dataset):
    def __init__(self, texta, textb):
        self.texta = texta
        self.textb = textb
        # print(type(texta), type(textb))
        # print(len(texta), len(textb))
        assert len(texta) == len(textb)

    def __getitem__(self, item):
        return self.texta[item], self.textb[item]

    def __len__(self):
        return len(self.texta)
    
class EvalDatasetIterater(data.Dataset):
    def __init__(self, texta, label):
        self.texta = texta
        self.label = label
        assert len(texta) == len(label)

        

    def __getitem__(self, item):
        return self.texta[item], self.label[item]

    def __len__(self):
        return len(self.texta)

def evaluation(dataset, model, device, epoch, args):

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    # print("### Creating features from model ###")
    X, Y, reprs = inference2(data_loader, model, device)
    Y = Y - 1
    # print(np.min(X), np.min(Y))

    score, _ = cluster_utils.clustering_metric(Y, X, args.class_num)
    score["avg"] = (score["acc"] + score["nmi"]) / 2
    score["epoch"] = epoch


    print(                          '### epoch {}### [f:{:.2f}....ari:{:.2f}....nmi:{:.2f}....acc:{:.2f}....avg:{:.2f}]'
          .format(epoch, score['f_measure'] * 100, score['ari'] * 100, score['nmi'] * 100, score['acc'] * 100, score['avg'] * 100))
    return X, Y, reprs, score

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # run = wb.init(project="tcl-text", mode=args.wb_mode)
    # run.tags = [args.dataset.lower()[:4], str(args.epochs), str(args.start_epoch)]
    # run.name = "|".join(run.tags)
    best_check_path, checkpoint_path ,embed_path, label_path = update_args(args)

    # run.config.update(args)
    import pprint
    pprint.pprint(args)
    print(best_check_path.__str__(), checkpoint_path.__str__(), embed_path.__str__(), label_path.__str__(), sep='\n')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MODEL_DIR"] = '../model'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    for itr in range []:

        # model and optimizer
        text_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')
        class_num = args.class_num
        model = network.Network(text_model, args.feature_dim, class_num)
        model = model.to('cuda')

            # model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        p = f""
        if best_check_path.exists():
            print("loading checkpoint...")
            checkpoint = torch.load(best_check_path.__str__())
            model.load_state_dict(checkpoint['net'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            # run.config.update(args)
            print("resuming from epoch ", args.start_epoch)
        else:
            print("checkpoint path doesn't exist")
        
    
        data, label = [], []
        with open(os.path.join(args.dataset_dir, args.dataset + '.txt'), 'r', encoding='utf8') as f1:
            for line in f1:
                data.append(line.strip('\n'))
        with open(os.path.join(args.dataset_dir, args.dataset + '_gnd.txt'), 'r', encoding='utf8') as f2:
            for line in f2:
                label.append(line.strip('\n'))
        evaldataset = EvalDatasetIterater(data, label)

        print("### initial eval")
        _, labels, reprs, first_score = evaluation(dataset=evaldataset, model=model, device='cuda', epoch=-1, args=args)
        print("reprs.shape=========== ", reprs.shape)
        t = time.time()
        
        


    
        embeds, labels, reprs, score = evaluation(dataset=evaldataset, model=model, device='cuda', epoch=epoch + 1, args=args)
        
        e_path = embed_path  / f"iter_{itr}_embeds.npy"
        l_path = label_path  / f"iter_{itr}_labels.npy"
        np.save(file=e_path.__str__(), arr=reprs) 
        np.save(file=l_path.__str__(), arr=labels)

            
            # print(f"Epoch [{epoch+1}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
        dif = (time.time() - t) / 60
        print(f"training time: {dif:.2f} MIN")
        # run.summary.update({"t(m)/training-total": dif})
        print("first scores:", first_score)
        

