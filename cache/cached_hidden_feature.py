
class CachedHiddenFeature:
    def __init__(self, origin_sample_uid, layer_id, np_hidden_feature):
        self.origin_sample_uid = origin_sample_uid
        self.layer_id = layer_id

        # np means numpy format
        self.np_hidden_feature = np_hidden_feature

    def get_origin_sample_uid(self):
        return self.origin_sample_uid

    def get_layer_id(self):
        return self.layer_id

    def get_np_hidden_feature(self):
        return self.np_hidden_feature
