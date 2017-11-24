import os
used_gpu = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

from utils import *
from se_inception_v3 import *


cfg = {"image_size": (3,150,150),
       "num_classes": 5270,
       "num_test": 1768182,
       "saved_model": "epoch-14-acc-0.63.pkl",
       "test_bson_path": '/data/lixiang/test.bson',
       "batch_size": 256,
       "data_worker": 4}


def submission(cfg):
    net = SEInception3(in_shape=cfg["image_size"], num_classes=cfg["num_classes"])
    if torch.cuda.is_available():
        net = net.cuda()

    print("*-------Begin Loading Saved Models!------*")
    net.load_pretrained_model('saved_models/' + cfg["saved_model"])
    net.eval()

    print("*----------Begin Loading Data!-----------*")
    data_frame = get_data_frame(cfg['test_bson_path'], cfg['num_test'], False)
    test_dataset = CdiscountTestDataset(cfg['test_bson_path'], data_frame)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg['batch_size'],
                             num_workers=cfg['data_worker'],
                             shuffle=False)

    save_pd = pd.DataFrame()
    save_pd["category_id"] = net.predict(test_loader)
    save_pd["_id"] = data_frame.index
    save_pd.to_csv("submission2.csv", columns=["_id","category_id"], index=False, sep=",")

    #save_pd = pd.DataFrame()
    #save_pd["category_id"] = net.predict(test_loader)
    #save_pd = data_frame.index
    #save_pd.index.name = "_id"
    print("*------------End Submission!------------*")


if __name__ == "__main__":
    submission(cfg)