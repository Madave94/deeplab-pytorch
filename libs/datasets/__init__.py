from .voc import VOC, VOCAug
from .cocostuff import CocoStuff10k, CocoStuff164k
from .s2ds import S2DS, S2DSAug


def get_dataset(name):
    return {
        "cocostuff10k": CocoStuff10k,
        "cocostuff164k": CocoStuff164k,
        "voc": VOC,
        "vocaug": VOCAug,
        "S2DS": S2DS,
        "S2DSAug": S2DSAug,
    }[name]
