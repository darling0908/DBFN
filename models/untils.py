from models.DBFN import mainNet


def get_net(config):
    name = config.net
    assert name in ['DB-FN'], (
        "Network Name not valid "
        "or not"
        "supported!")
    if name == 'DB-FN':
        return mainNet(num_classes=config.num_outs)