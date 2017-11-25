import os
used_gpu = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

from utils import *
from se_inception_v3 import *


cfg = {"image_size": (3, 180, 180),
       "num_classes": 5270,
       "num_test": 1768182,
       "saved_model": "epoch-5-acc-0.67.pth",
       "test_bson_path": '/data/lixiang/test.bson',
       "batch_size": 256,
       "data_worker": 4}


def submission(cfg):
    net = SEInception3(in_shape=cfg["image_size"], num_classes=cfg["num_classes"])
    if torch.cuda.is_available():
        net = net.cuda()

    print("*-------Begin Loading Saved Models!------*")
    pretrain_state_dict = get_state_dict('saved_models/' + cfg["saved_model"])
    state_dict = net.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        try:
            state_dict[key] = pretrain_state_dict["module."+key]
        except KeyError:
            print("KeyError: {} dosen't lie in pretrain state dict".format(key))
            continue
    net.load_state_dict(state_dict)
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
    save_pd.to_csv("submission3.csv", columns=["_id","category_id"], index=False, sep=",")
    print("*------------End Submission!------------*")


if __name__ == "__main__":
    submission(cfg)