amigos_labels = {
    "Extroversion": 0,
    "Agreeableness": 1,
    "Conscientiousness": 2,
    "Emotional Stability": 3,
    "Creativity (openness)": 4
}

ascertain_labels = {
    "Extroversion": 0,
    "Agreeableness": 1,
    "Conscientiousness": 2,	
    "Emotional Stability": 3,
    "Openness (Creativity)": 4
}

amigos_labels_reverse = {
    0: 'Extroversion',
    1: 'Agreeableness',
    2: 'Conscientiousness',
    3: 'Emotional Stability',
    4: 'Creativity (openness)'
}

ascertain_labels_reverse = {
    0: 'Extraversion',
    1: 'Agreeableness',
    2: 'Conscientiousness',
    3: 'Emotional Stability',
    4: 'Openness'
}

validation_schemes = [
    "K-FOLDCV",
    "LOOCV",
    "SPLIT"
]

supported_datasets = [
    "AMIGOS",
    "ASCERTAIN"
]

optimizers = [
    "Adam",
    "AdamW",
    "SGD"
]

schedulers = [
    "StepLR",
    "MultiStepLR",
    "ReduceLROnPlateau",
    "CosineAnnealingLR"
]

criterions = [
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "TripletMarginLoss"
]

positional_encodings = [
    "sinusoidal",
    "learnable"
]

merge_mels_typologies = [
    "channels",
    "samples"
]

discretization_methods = [
    "personality_mean",
    "fixed_mean"
]

augmentation_methods = [
    "spec_augment",
    "additive_noise",
    "flipping"
]