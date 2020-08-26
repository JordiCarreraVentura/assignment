from collections import Counter

from FeatureExtraction import FeatureExtractor

from tools import from_json

from tqdm import tqdm


POSITIVE_TEST = 'data/fake_test.json'
POSITIVE_TRAIN = 'data/fake_train.json'
NEGATIVE_TEST = 'data/real_test.json'
NEGATIVE_TRAIN = 'data/real_train.json'
    
    
def has_sentence_end(title):
    if (
        title.strip().endswith('.')
        or title.strip().endswith('?')
        or title.strip().endswith('!')
    ):
        return True
    return False


class Dataset:

    def __init__(self):
        self._X = []
        self._Y = []
        self._is_train = []
    
    def __len__(self):
        return len(self._X) + len(self._Y)
    
    def __str__(self):
        return '<%s <%s>>' % (
            self.__class__.__name__,
            ' '.join([
                '%d=%s' % (freq, label)
                for label, freq in Counter(self._Y).most_common()
            ])
        )

    def __iter__(self):
        for x, y in zip(self._X, self._Y):
            yield x, y
    
    def train(self):
        for x, y, is_train in zip(
            self._X, self._Y, self._is_train
        ):
            if is_train:
                yield x, y
    
    def test(self):
        for x, y, is_train in zip(
            self._X, self._Y, self._is_train
        ):
            if not is_train:
                yield x, y
    
    def labels(self):
        return sorted(list(set(self._Y)))



class FakenewsDataset(Dataset):
    
    def __init__(self):
        super().__init__()
        self._positive_test = self.__read_from_json(POSITIVE_TEST)
        self._positive_train = self.__read_from_json(POSITIVE_TRAIN)
        self._negative_test = self.__read_from_json(NEGATIVE_TEST)
        self._negative_train = self.__read_from_json(NEGATIVE_TRAIN)
    
    def __read_from_json(self, path):
        return from_json(path)
    


class TitleTextDataset(FakenewsDataset):

    def __init__(self):
        super().__init__()

        for json_record in tqdm(self._positive_train):
            x = self.__make_record(json_record)
            self._X.append(x)
            self._Y.append(1)
            self._is_train.append(True)

        for json_record in tqdm(self._negative_train):
            x = self.__make_record(json_record)
            self._X.append(x)
            self._Y.append(0)
            self._is_train.append(True)

        for json_record in tqdm(self._positive_test):
            x = self.__make_record(json_record)
            self._X.append(x)
            self._Y.append(1)
            self._is_train.append(False)

        for json_record in tqdm(self._negative_test):
            x = self.__make_record(json_record)
            self._X.append(x)
            self._Y.append(0)
            self._is_train.append(False)

    def __make_record(self, json_record):
        x = '%s%s%s' % (
            json_record['title'].strip(),
            ' ' if has_sentence_end(json_record['title']) else '. ',
            json_record['text']
        )
        x = ' '.join(x.split('\n'))
        return x




class TextDataset(FakenewsDataset):

    def __init__(self):
        super().__init__()

        for json_record in tqdm(self._positive_train):
            x = json_record['text']
            self._X.append(x)
            self._Y.append(1)
            self._is_train.append(True)

        for json_record in tqdm(self._negative_train):
            x = json_record['text']
            self._X.append(x)
            self._Y.append(0)
            self._is_train.append(True)

        for json_record in tqdm(self._positive_test):
            x = json_record['text']
            self._X.append(x)
            self._Y.append(1)
            self._is_train.append(False)

        for json_record in tqdm(self._negative_test):
            x = json_record['text']
            self._X.append(x)
            self._Y.append(0)
            self._is_train.append(False)



class TitleDataset(FakenewsDataset):

    def __init__(self):
        super().__init__()

        for json_record in tqdm(self._positive_train):
            x = json_record['title']
            self._X.append(x)
            self._Y.append(1)
            self._is_train.append(True)

        for json_record in tqdm(self._negative_train):
            x = json_record['title']
            self._X.append(x)
            self._Y.append(0)
            self._is_train.append(True)

        for json_record in tqdm(self._positive_test):
            x = json_record['title']
            self._X.append(x)
            self._Y.append(1)
            self._is_train.append(False)

        for json_record in tqdm(self._negative_test):
            x = json_record['title']
            self._X.append(x)
            self._Y.append(0)
            self._is_train.append(False)




class NormalizedTitleTextDataset(TitleTextDataset):

    remove_entities=True
    remove_nonalpha=True
    remove_stopwords=True
    lowercase=False

    def __init__(self):
    
        super().__init__()
        fext = FeatureExtractor(
            remove_entities = self.remove_entities,
            remove_nonalpha = self.remove_nonalpha,
            remove_stopwords = self.remove_stopwords
        )
        self._X = fext.transform(self._X)
        


if __name__ == '__main__':
    d = TitleTextDataset()
    
    from tools import to_csv
    XY = [('text', 'label')]
    for x, y in d.train():
        XY.append((x, y))
    to_csv(XY, 'train.csv')
    
    XY = [('text', 'label')]
    for x, y in d.test():
        XY.append((x, y))
    to_csv(XY, 'test.csv')
    exit()
    
    print(d)
    
    print(len(list(d.train())))
    print(len(list(d.test())))
    
    for x, y in d:
        print(x)
        print(y)
        print()
