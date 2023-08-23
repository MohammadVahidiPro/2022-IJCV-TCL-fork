import numpy as np
import torch
import argparse
import network
import loss
from utils import cluster_utils
from utils.save_model import save_model, save_model_10
from evaluation import inference2
from torch.utils import data
from sentence_transformers import SentenceTransformer
from EDA import augment
import os
import itertools
import nlpaug.augmenter.word as naw
import functools
from pathlib import Path
import time
import wandb as wb
global_step = 0

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        dif = time.time() - start
        dif_min = round(dif / 60, 2)
        name = func.__name__
        
        if global_step < 20:
            print(f"## timer step {global_step} ## {name}: {dif_min} MIN")
        if global_step < 20 or global_step % 20:
            wb.log({f"t(m)/{name}": dif_min, f"t(s)/{name}": dif})
        return results
    return wrapper

def get_args_parser():
    parser = argparse.ArgumentParser("TCL boosting for clustering", add_help=False)
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Batch size per GPU"
    )
    parser.add_argument(
        "--start_epoch", default=1, type=int, help="start epoch"
    )
    parser.add_argument("--epochs", default=500, type=int) #10
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
    parser.add_argument("--weight_decay", type=float, default=0., help="weight decay")
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
        default="Biomedical",
        type=str,
        help="dataset",
        choices=["StackOverflow", "Biomedical"],
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
        default=True,
        help="resume from checkpoint",
    )
    parser.add_argument("--save_freq", default=25, type=int, help="saving frequency")
    parser.add_argument("--num_workers", default=8, type=int)

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
    
    if args.check_id == "this":
        args.check_id = run.id
    embed_path = Path(args.model_path).resolve() / args.train_id / f"{run.id}-boost" / 'embed'
    label_path = Path(args.model_path).resolve() / args.train_id / f"{run.id}-boost" / 'label'
    checkpoint_path = Path(args.model_path).resolve() / f"{run.id}-boost" / 'checkpoints'
    best_check_path = Path(args.model_path).resolve() / args.check_id / "checkpoints" / "best_model_avg.tar"
    embed_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    return best_check_path, checkpoint_path ,embed_path, label_path

class DatasetIterater(data.Dataset):
    def __init__(self, text, texta, textb):
        self.text = text
        self.texta = texta
        self.textb = textb

    def __getitem__(self, item):
        return self.texta[item], self.textb[item], self.text[item], item

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

def perform_augmentation(args):
    # prepare data
    data_dir = args.dataset_dir
    aug1, aug2 = [], []

    # EDA augmentation
    augment.gen_eda(os.path.join(data_dir, args.dataset + 'WithGnd.txt'),
                    os.path.join(data_dir, args.dataset + 'EDA_aug_boost.txt'),
                    0.2, 0.2, 0.2, 0.2, 1)
    with open(os.path.join(data_dir, args.dataset + 'EDA_aug_boost.txt'), "r") as f1:
        for line in f1:
            aug1.append(line.strip('\n'))
        f1.close()

    # Roberta augmentation
    data = []
    with open(os.path.join(data_dir, args.dataset + '.txt'), "r") as f2:
        lines = f2.readlines()
        lines = [line.strip('\n') for line in lines]
        # for line in f1:
        #     data.append(line.strip('\n'))
        data = lines
        f2.close()
        
    aug_robert = naw.ContextualWordEmbsAug(
        model_path='roberta-base', action="substitute", device='cuda', aug_p=0.2)
    tmp = []
    internal = 400  # the larger the faster, as long as not overflow the memory
    with torch.no_grad():
        for i in range(len(data)):
            tmp.append(data[i])
            if (i + 1) % internal == 0 or (i + 1) == len(data):
                # if (i + 1) % 2000 == 0:
                    # print("roberta aug: the iter {} / {}".format(i // 2000, np.ceil(len(data) / 2000)))
                aug2.extend(aug_robert.augment(tmp))
                tmp.clear()
    return aug1, aug2

@timer
def boost(model, optimizer, optimizer_head, criterion, pseudo_labels, data_loader, class_num):
    loss_epoch = 0
    for step, (x_i, x_j, x, index) in enumerate(data_loader):
        optimizer.zero_grad()
        optimizer_head.zero_grad()
        model.eval()
        with torch.no_grad():
            c = model.forward_c(x)
            confidence = c.max(dim=1).values
            unconfident_pred_index = (confidence < 0.9)
            cur_pseudo_label = criterion.generate_pseudo_labels(c, class_num)
            tmp_pseudo_label = pseudo_labels[index]
            todo_index = (tmp_pseudo_label == -1)
            tmp_pseudo_label[todo_index] = cur_pseudo_label[todo_index]
            tmp_pseudo_label[unconfident_pred_index] = -1
            pseudo_labels[index] = tmp_pseudo_label
        model.train()
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        instance_loss = criterion.forward_instance_elim(z_i, z_j, pseudo_labels[index])
        ce_loss = torch.tensor(0., requires_grad=True).to('cuda')
        if torch.unique(pseudo_labels[index]).shape[0] > class_num:
            pseudo_index = (pseudo_labels[index] != -1)
            c_ = model.forward_c_psd(x_j, pseudo_index)
            ce_loss += criterion.forward_weighted_ce(c_, pseudo_labels[index][pseudo_index], class_num)
        loss = instance_loss + ce_loss
        loss.backward()
        optimizer.step()
        optimizer_head.step()
        # if step % 50 == 0:
        #     print(f"Step [{step}/{len(data_loader)}]\t "
        #           f"loss_instance: {instance_loss.item()}\t "
        #           f"loss_ce: {ce_loss.item()}")
        wb.log({"bch_boost/instance": instance_loss.item(),
                "bch_boost/cross-entropy": ce_loss.item(),
                "bch_boost/loss": loss.item()})
        loss_epoch += loss.item()
        idx, counts = torch.unique(pseudo_labels, return_counts=True)
    return loss_epoch

@timer
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
    wb.log(score)
    return X, Y, reprs, score


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    run = wb.init(project="tcl-text", mode=args.wb_mode)
    run.tags = [args.dataset.lower()[:4], str(args.epochs), str(args.start_epoch), 'boost']
    run.name = "|".join(run.tags)
    best_check_path, checkpoint_path ,embed_path, label_path = update_args(args)

    run.config.update(args)
    import pprint
    pprint.pprint(args)
    print("wandb run id: ", run.id)
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

    # model and optimizer
    text_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')
    class_num = args.class_num
    model = network.Network(text_model, args.feature_dim, class_num)
    model = model.to('cuda')

    optimizer = torch.optim.SGD(model.backbone.parameters(),
                                lr=args.lr_backbone,
                                weight_decay=args.weight_decay)
    optimizer_head = torch.optim.Adam(itertools.chain(model.instance_projector.parameters(),
                                                      model.cluster_projector.parameters()),
                                      lr=args.lr_head,
                                      weight_decay=args.weight_decay)

    if args.resume == 'True':
        # model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        if best_check_path.exists():
            print("loading checkpoint...")
            checkpoint = torch.load(best_check_path.__str__())
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            run.config.update(args)
            print("resuming from epoch ", args.start_epoch)
        else:
            print("checkpoint path doesn't exist")
            print("training from scratch")
    else:
        print("training from scratch")


    # loss
    loss_device = torch.device("cuda")
    criterion = loss.ContrastiveLoss(args.batch_size, args.batch_size, class_num,
                                     args.instance_temperature, args.cluster_temperature,
                                     loss_device).to(loss_device)

    # pipeline
    labels = []
    with open(os.path.join(args.dataset_dir, args.dataset + '_gnd.txt'), "r") as f1:
        for line in f1:
            labels.append(int(line.strip('\n')))
        f1.close()

    labels = np.array(labels)
    data_size = len(labels)
    pseudo_labels = -torch.ones(data_size, dtype=torch.long).to('cuda')
    last_pseudo_num = 0


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
    best_score = first_score.copy()
    e_path = embed_path  / f"iter_{-1}_embeds.npy"
    l_path = label_path  / f"iter_{-1}_labels.npy"
    np.save(file=e_path.__str__(), arr=reprs) 
    np.save(file=l_path.__str__(), arr=labels)
    print("### start training...")
    best_conf_score = None
    wb.watch(model)
    t = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # prepare data
        data_dir = args.dataset_dir

        aug1, aug2 = perform_augmentation(args)
        
        dataset = DatasetIterater(data, aug2, aug1)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )
        loss_epoch = boost(model, optimizer, optimizer_head, criterion, pseudo_labels, data_loader, class_num)
        wb.log({"epoch": epoch + 1, "loss": loss_epoch / len(data_loader)})

        # if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
        #     save_model(args, model, optimizer, optimizer_head, epoch + 1)

        # print(f"Epoch [{epoch+1}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
        pseudo_index = (pseudo_labels != -1).cpu()
        pseudo_num = pseudo_index.sum()
        if pseudo_num > 0:
            conf_scores, _ = cluster_utils.clustering_metric(labels[pseudo_index],
                                                       pseudo_labels[pseudo_index].cpu().numpy(),
                                                       args.class_num)
            conf_scores["avg"] = (conf_scores["acc"] + conf_scores["nmi"]) / 2
            print('pseudo CONFIDENCE scores F = {:.4f} ARI = {:.4f} NMI = {:.4f} ACC = {:.4f} AVG = {:.4f}'.format(conf_scores['f_measure'],
                                                                                                                    conf_scores['ari'],
                                                                                                                    conf_scores['nmi'],
                                                                                                                    conf_scores['acc'],
                                                                                                                    conf_scores['avg']))
            
            conf_scores["epoch"] = conf_scores
            if best_conf_score is None or best_conf_score['avg'] < conf_scores['avg']:
                best_conf_score = conf_scores.copy()
                print("////////////////// best conf \ \ \ \ \ \ \ ")
                save_model(args, model, optimizer, optimizer_head, epoch + 1, path=checkpoint_path, id=run.id, best=True, conf=True)
                wb.log({f'best-conf/{k}': v for k, v in conf_scores.items()})
                
            wb.log({f"conf/{k}": v for k, v in conf_scores.items()})
        last_pseudo_num = pseudo_num

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            best = False
            embeds, labels, reprs, score = evaluation(dataset=evaldataset, model=model, device='cuda', epoch=epoch + 1, args=args)
            if score["avg"] > best_score["avg"]:
                best = True
                best_score = score.copy()
                wb.log({f"best/{k}": v for k, v in best_score.items()})
                print(f"/\/\/\ new best at epoch {epoch + 1} /\/\/\ ")

            e_path = embed_path  / f"iter_{epoch + 1}_embeds.npy"
            l_path = label_path  / f"iter_{epoch + 1}_labels.npy"
            np.save(file=e_path.__str__(), arr=reprs) 
            np.save(file=l_path.__str__(), arr=labels)
            save_model(args, model, optimizer, optimizer_head, epoch + 1, path=checkpoint_path, id=run.id, best=best)

        if (epoch + 1) % 10 == 0:
            save_model_10(args, model, optimizer, optimizer_head, epoch + 1, path=checkpoint_path, id=run.id)
    dif = time.time() - t
    print(f"training time: {dif/60:.2f} MIN")
    run.summary.update({"t(m)/training-total": dif/60})
    run.summary.update({"t(s)/training-total": dif})

    print("first eval scores:", first_score)
    print("best eval scores:",  best_score)
    print("best confidence score: ", best_conf_score)
    run.summary.update({f"first/{k}": v for k, v in first_score.items()})
    run.summary.update({f"top/{k}": v for k, v in best_score.items()})    
    
    
    run.finish()
