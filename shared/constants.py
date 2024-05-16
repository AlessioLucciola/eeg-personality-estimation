amigos_labels = {
    "Extroversion": 0,
    "Agreeableness": 1,
    "Conscientiousness": 2,
    "Emotional Stability": 3,
    "Creativity (openness)": 4
}

amigos_labels_reverse = {
    0: 'Extroversion',
    1: 'Agreeableness',
    2: 'Conscientiousness',
    3: 'Emotional Stability',
    4: 'Creativity (openness)'
}

validation_schemes = [
    "K-FOLDCV",
    "LOOCV",
    "SPLIT"
]

supported_datasets = [
    "AMIGOS"
]

optimizers = [
    "Adam",
    "AdamW",
    "SGD"
]

schedulers = [
    "StepLR",
    "MultiStepLR",
    "ReduceLROnPlateau"
]

positional_encodings = [
    "sinusoidal"
]