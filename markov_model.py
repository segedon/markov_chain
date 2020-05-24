import random
import numpy


class MarkovModel:
    def __init__(self, n=1):
        self.n = n
        self._dict = dict()

    def _update_weight(self, window, word):
        if word in self._dict[window].keys():
            self._dict[window][word] += 1
        else:
            self._dict[window][word] = 1

    def fit(self, words):
        for i in range(len(words) - self.n):
            # Создаем окно
            window = tuple(words[i: i + self.n])
            word_for_window = None
            if i == len(words) - self.n - 1:
                word_for_window = 'END'
            else:
                word_for_window = words[i + self.n]
            # Добавляем в словарь
            if window in self._dict.keys():
                self._update_weight(window, word_for_window)
            else:
                self._dict[window] = {word_for_window: 1}

    def _get_next_word(self, current_word):
        # собираем все слова, которые могут идти после данного слова
        words_list = [x[0] for x in self._dict[current_word].items()]
        # рассчитываем для них распределение вероятностей
        prb_list = [x[1] for x in self._dict[current_word].items()]
        prb_list = list(map(lambda x: x / sum(prb_list), prb_list))
        # выбираем следующее слова в соответствии с распределением
        return numpy.random.choice(words_list, size=1, p=prb_list)[0]

    def generate_random_word(self):
        return random.choice(list(self._dict.keys()))

    def generate_sentence(self, start_word=None, lenght=10):
        if start_word is None:
            start_word = self.generate_random_word()
        if start_word not in self._dict.keys():
            raise ValueError("Данного слова нет в нашем распределении")
        sentence = ' '.join(start_word)
        current_word = start_word
        for i in range(lenght - self.n):
            next_word = self._get_next_word(current_word)
            sentence += ' ' + next_word
            # если конец, то дальше не сможешь генерировать
            if next_word == 'END':
                return sentence
            # иначе генерируем следующее слово
            else:
                current_word = (*current_word[1:], next_word)
        return sentence


if __name__ == "__main__":
    with open("test.txt") as f:
        text = f.read()
    words = [x.strip() for x in text.split(' ')]
    # параметр n - это размер окна
    model = MarkovModel(n=2)
    model.fit(words)
    print(model.generate_sentence(lenght=20))
