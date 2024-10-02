import models
MODELS = {
    'transformer': models.TransformerModel,
}

import datasetss
DATASETS = {
    'nli': datasetss.NLIDataset,
    'our_nli': datasetss.OurNLIDataset,
    'stress-test': datasetss.StressTestNLIDataset,
    'our-stress-test': datasetss.OurStressTestNLIDataset
}

TRAINERS = {}

import evaluators
EVALUATORS = {
    'classification': evaluators.ClassificationEvaluator,
    'classification-stress': evaluators.ClassificationStressEvaluator,
    'classification-our-stress': evaluators.ClassificationOurStressEvaluator,
}
