from thirdparty.hed_impl.python.run import predict


def hed_predict_api(img):
    """
    input RGB raw img
    :param img:
    :return: the line sketched img of raw
    """
    return predict(img)
