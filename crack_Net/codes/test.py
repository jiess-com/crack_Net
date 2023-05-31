import PIL

from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def test(test_data_path="F:/crack_test/0001.jpg",
         save_path='deepcrack_results/',
         pretrained_model="C:/Users/User/Desktop/DeepCrack_CT260_FT1.pth", ):
    if not os.path.exists(save_path):
        os.mkdir(save_path)



    # test_pipline = dataReadPip(transforms=None)

    # test_list = readIndex(test_data_path)
    # # test_dataset = loadedDataset(readIndex("C:/Users/User/Desktop/DCrack/crack_test/", shuffle=True))
    #
    # test_dataset = loadedDataset(test_list, preprocess=test_pipline)
    #
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
    #                                           shuffle=False, num_workers=1, drop_last=False)

    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()

    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)

    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=True))

    model.eval()
    data_transform = transforms.Compose([transforms.ToTensor()])
    img=cv2.imread(test_data_path)
    img = data_transform(img)
    # img=PIL.Image(test_data_path)
    with torch.no_grad():
        # for names, (img, lab) in tqdm(zip( test_list,test_loader)):
        # test_data = img.type(torch.cuda.FloatTensor).to(device)
        # test_data=torch.from_numpy(img).permute(1,2,0)
        test_data = torch.unsqueeze(img, dim=0)

        # save_pred = torch.zeros((1,3,512, 512))

        out=model(test_data)

        prediction=torch.sigmoid(out[0].contiguous().cpu())
        # prediction=prediction.to("cpu").numpy().astype(np.uint8)
        prediction=prediction.squeeze(0)
        prediction=prediction.squeeze(0)
        prediction[prediction>=0.5]=255
        prediction[prediction<0.5]=0
        prediction=prediction.numpy()
        cv2.imwrite("re.png",prediction)


if __name__ == '__main__':
    test()
