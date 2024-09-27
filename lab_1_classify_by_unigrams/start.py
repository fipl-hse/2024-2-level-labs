"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

def tokenize(text):

    text = text.lower()
    split_text = ''.join(text)
    token_list = []
    for symbol in split_text:
        if symbol.isalpha():
            token_list.append(symbol)
    return token_list

def calculate_frequencies(token_text):

    freq_dic = {}
    for letter in token_text:
        freq_dic[letter] = token_text.count(letter) / len(token_text)
    return freq_dic

def create_language_profile(language,text):

    language_profile = {"name":language,"freq":calculate_frequencies(tokenize(text))}
    return language_profile

def calculate_mse(predicted, actual):

    result = 0
    for index in range(len(actual)):
        step = (((predicted[index] - actual[index]) ** 2))
        result += step
    return result/len(actual)

def compare_profiles(unknown_profile,profile_to_compare):

    unknown_freq, compare_freq = unknown_profile["freq"], profile_to_compare["freq"]
    unknown_keys, compare_keys = list(unknown_freq.keys()), list(compare_freq.keys())
    all_keys = (unknown_keys + list(set(compare_keys) - set(unknown_keys)))
    all_unknown_profile, all_profile_to_compare = dict.fromkeys(all_keys, 0), dict.fromkeys(all_keys, 0)
    for (key, value) in unknown_freq.items():
        for (key_, value_) in all_unknown_profile.items():
            if key == key_:
                all_unknown_profile[key_] = value
    for (key, value) in compare_freq.items():
        for (key_, value_) in all_profile_to_compare.items():
            if key == key_:
                all_profile_to_compare[key_] = value

    return calculate_mse(list(all_unknown_profile.values()), list(all_profile_to_compare.values()))

def detect_language(unknown_profile,profile_1,profile_2,):

    if compare_profiles(unknown_profile,profile_1) < compare_profiles(unknown_profile,profile_2):
        return profile_1["name"]
    elif compare_profiles(unknown_profile,profile_2) < compare_profiles(unknown_profile,profile_1):
        return profile_2["name"]


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
    result = print(create_language_profile("english",en_text),(create_language_profile("german",de_text)))
    assert result, "Detection result is None"




if __name__ == "__main__":
    main()
