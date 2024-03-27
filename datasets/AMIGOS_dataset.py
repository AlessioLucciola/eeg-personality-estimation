from config import AMIGOS_NUM_CLASSES, AMIGOS_DATASET_DIR
from shared.constants import amigos_labels
from EEG_classification_dataset import EEGClassificationDataset

class AMIGOSDataset(EEGClassificationDataset):
    def __init__(self,
                 data_path: str
                ):
        super(AMIGOSDataset, self).__init__(
            data_path=data_path,
            dataset_name="AMIGOS",
            labels = amigos_labels,
            labels_classes=AMIGOS_NUM_CLASSES,
            subject_ids=self.get_subject_ids_static(data_path)
        )

if __name__ == "__main__":
    dataset = AMIGOSDataset(path=AMIGOS_DATASET_DIR)