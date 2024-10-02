import json
from collections import defaultdict

class GeneralLogger(object):
    def __init__(self):
        self.logits_across_time = defaultdict(list)

    def update(self, idxs, measures):
        for idx, measure in zip(idxs, measures):
            self.logits_across_time[idx].append(measure.numpy().tolist())

    def get_measures(self, idxs):
        return [self.logits_across_time[idx] for idx in idxs]

    def reset(self):
        super().reset()
        self.var = None

    def on_epoch_end(self):
        pass

    def to_json(self, fp):
        json.dump(self.logits_across_time, fp)
