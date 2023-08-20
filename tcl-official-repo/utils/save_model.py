import os
import torch


def save_model(args, model, optimizer, optimizer_head, current_epoch, path, id, best=False):
    check_path = path / f"checkpoint_{current_epoch}.tar"
    # out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'optimizer_head': optimizer_head.state_dict(), 'epoch': current_epoch}
    torch.save(obj=state, f=check_path)

    if best:
        best_path = path / f"best_model_avg.tar"
        # out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch))
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'optimizer_head': optimizer_head.state_dict(), 'epoch': current_epoch}
        torch.save(obj=state, f=best_path)