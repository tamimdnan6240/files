from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #Tamim: off visdom


def test(test_data_path="data/test_example.txt",
         save_path='deepcrack_results/',
         pretrained_model='C:/Users/tamim/DeepCrack_Master/codes/checkpoints/DeepCrack_CT260_FT1/checkpoints/DeepCrack_CT260_FT1_epoch(1)_0000003_2023-03-03-00-12-59.pth', ):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    test_pipline = dataReadPip(transforms=None)

    test_list = readIndex(test_data_path)

    test_dataset = loadedDataset(test_list, preprocess=test_pipline)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=1, drop_last=False)

    # -------------------- build trainer --------------------- #

    device = torch.device("cpu")
    #num_gpu = torch.cpu.device_count()

    model = DeepCrack()

    #model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)

    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=False)) ## Tamim: multiple gpu false

    model.eval()

    with torch.no_grad():
        for names, (img, lab) in tqdm(zip(test_list, test_loader)):
            test_data, test_target = img.type(torch.FloatTensor).to(device), lab.type(torch.FloatTensor).to(device)
            test_pred = trainer.val_op(test_data, test_target)
            test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
            save_pred = torch.zeros((512 * 2, 512))
            save_pred[:512, :] = test_pred
            save_pred[512:, :] = lab.cpu().squeeze()
            save_name = os.path.join(save_path, os.path.split(names[1])[1])
            save_pred = save_pred.numpy() * 255
            cv2.imwrite(save_name, save_pred.astype(np.uint8))


if __name__ == '__main__':
    test()
