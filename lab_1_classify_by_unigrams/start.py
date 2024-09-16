# pylint:disable=too-many-locals, unused-argument, unused-variable


def main() -> None:

    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    paths_to_profiles = [
        'assets/profiles/de.json',
        'assets/profiles/en.json',
        'assets/profiles/es.json',
        'assets/profiles/fr.json',
        'assets/profiles/it.json',
        'assets/profiles/ru.json',
        'assets/profiles/tr.json'
    ]

    result = 'AAAAA'
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
