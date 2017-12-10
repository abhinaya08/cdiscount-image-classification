import os
used_gpu = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

from utils import *
from se_inception_v3 import *
from data_transform import *


cfg = {"image_size": (3, 180, 180),
       "num_classes": 5270,
       "num_test": 1768182,
       "saved_model": "ep-14acc0.6891-model.pth",
       "test_bson_path": '/data/lixiang/test.bson',
       "batch_size": 256,
       "data_worker": 4}


def submission(cfg):
    net = SEInception3(num_classes=cfg["num_classes"])
    if torch.cuda.is_available():
        net = net.cuda()

    print("*-------Begin Loading Saved Models!------*")
    print("loaded model:", cfg["saved_model"])
    net.load_pretrained_model('saved_models/' + cfg["saved_model"])
    net.eval()

    print("*----------Begin Loading Data!-----------*")
    data_frame = extract_categories_df(cfg['test_bson_path'], is_test=True)
    test_dataset = CdiscountTrain(cfg['test_bson_path'], data_frame,
                                  transform=valid_augment)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg['batch_size'],
                             num_workers=cfg['data_worker'],
                             shuffle=False)

    # data_frame = get_data_frame(cfg['test_bson_path'], cfg['num_test'], False)
    # test_dataset = CdiscountTestDataset(cfg['test_bson_path'], data_frame, valid_augment)
    # test_loader = DataLoader(test_dataset,
    #                          batch_size=cfg['batch_size'],
    #                          num_workers=cfg['data_worker'],
    #                          shuffle=False)

    save_pd = pd.DataFrame()
    results_dict = net.predict(test_loader, True)

    # Compute major vote for products with multiple images. If tied, use the 1st one.
    for _id in results_dict:
        voting = results_dict[_id]
        results_dict[_id] = sorted(zip(voting, [voting.count(vote) for vote in voting]), key=lambda x: -x[1])[0][0]
    save_pd["_id"], save_pd["category_id"] = zip(*results_dict)

    # save_pd["category_id"] = net.predict(test_loader)
    # save_pd["_id"] = data_frame["item_id"]
    save_pd.to_csv("submission9.csv", columns=["_id", "category_id"], index=False, sep=",")
    print("*------------End Submission!------------*")


if __name__ == "__main__":
    submission(cfg)
