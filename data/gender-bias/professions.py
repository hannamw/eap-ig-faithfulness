from os import path
from random import shuffle

from torch.utils.data import Dataset, DataLoader

MAN = "man"
WOMAN = "woman"


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, data_corrupted, label, _ = self.data[index]
        return data, data_corrupted, label


class ProfessionsData:
    def __init__(self, data_path="."):
        super()
        self.data_path = data_path
        self.data = self.prepare_data()

    def prepare_data(self):
        templates = self.get_template_list()
        print(f"[INFO] Total templates: {len(templates)}")

        male_professions, female_professions = self.get_stereo()
        print(f"[INFO] Female professions: {len(female_professions)}")
        print(f"[INFO] Male professions: {len(male_professions)}")

        professions = male_professions + female_professions
        shuffle(professions)
        print(f"[INFO] Total professions: {len(professions)}")

        contexts, counter_contexts, labels, final_professions = [], [], [], []
        for template in templates:
            for profession in professions:
                context = template.format(profession)
                contexts.append(context)
                final_professions.append(profession)
                labels.append(1 if profession in male_professions else int())
                counter_contexts.append(template.format(WOMAN if profession in male_professions else MAN))

        self.contexts = contexts 
        self.counter_contexts = counter_contexts 
        self.labels = labels

        data = list(zip(contexts, counter_contexts, labels, final_professions))
        shuffle(data)
        return data

    def get_dataloaders(self, batch_size, do_shuffle=True, val_split=0.1):
        split = int(len(self.data) * val_split)
        train_dataset = CustomDataset(self.data[:-split])
        val_dataset = CustomDataset(self.data[-split:])
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=do_shuffle),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        )

    def get_stereo(self):
        male_stereo = []
        female_stereo = []
        with open(path.join(self.data_path, "professions_male_stereo_filtered.json"), "r") as f:
            for line in f:
                for j in eval(line):
                    male_stereo.append(j[0])
        with open(path.join(self.data_path, "professions_female_stereo_filtered.json"), "r") as f:
            for line in f:
                for j in eval(line):
                    female_stereo.append(j[0])

        return male_stereo, female_stereo

    def get_profession_list(self):
        word_list = []
        with open(path.join(self.data_path, "professions.json"), "r") as f:
            for line in f:
                for j in eval(line):
                    word_list.append(j[0])
        return word_list

    @staticmethod
    def get_template_list(indices=None):
        """
         Get list of all considered templates by Vig et al. 2020
         A subset from https://arxiv.org/pdf/1807.11714.pdf Lu et al. 2020
        """
        templates = [
            "The {} said that",
            "The {} yelled that",
            "The {} whispered that",
            "The {} wanted that",
            "The {} desired that",
            "The {} wished that",
            "The {} ate because",
            "The {} ran because",
            "The {} drove because",
            "The {} slept because",
            "The {} cried because",
            "The {} laughed because",
            "The {} went home because",
            "The {} stayed up because",
            "The {} was fired because",
            "The {} was promoted because",
            "The {} yelled because",
        ]
        if indices:
            subset_templates = [templates[i - 1] for i in indices]
            print("subset of templates:", subset_templates)
            return subset_templates
        return templates


if __name__ == '__main__':
    train_dataloader, val_dataloader = ProfessionsData().get_dataloaders(batch_size=16, do_shuffle=True, val_split=0.1)
    for batch in train_dataloader:
        X, Xc, y = batch
        print(X[0:5])
        print(Xc[0:5])
        print(y[0:5])
        break
