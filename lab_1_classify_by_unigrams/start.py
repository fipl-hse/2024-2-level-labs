"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

from main import (
    tokenize,
    calculate_frequencies,
    create_language_profile,
    calculate_mse,
    compare_profiles,
    detect_language,
)


def main() -> None:
    """
    Main function to execute the language detection process using all functions.
    """
    # 1. Читаем тексты из файлов (английский, немецкий, неизвестный)
    with open("lab_1_classify_by_unigrams/assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("lab_1_classify_by_unigrams/assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("lab_1_classify_by_unigrams/assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    # 2. Токенизация текста
    en_tokens = tokenize(en_text)
    de_tokens = tokenize(de_text)
    unknown_tokens = tokenize(unknown_text)

    # 3. Вычисление частот токенов
    en_frequencies = calculate_frequencies(en_tokens)
    de_frequencies = calculate_frequencies(de_tokens)
    unknown_frequencies = calculate_frequencies(unknown_tokens)

    # 4. Создание языковых профилей
    en_profile = create_language_profile("English", en_text)
    de_profile = create_language_profile("German", de_text)
    unknown_profile = create_language_profile("Unknown", unknown_text)

    # Проверяем, что профили были успешно созданы
    if not all([en_profile, de_profile, unknown_profile]):
        print("Error: One or more profiles could not be created.")
        return

    # 5. Сравнение профилей с помощью MSE
    distance_en = compare_profiles(unknown_profile, en_profile)
    distance_de = compare_profiles(unknown_profile, de_profile)

    if distance_en is None or distance_de is None:
        print("Error: Could not compare profiles.")
        return

    # Выводим расстояния (MSE)
    print(f"Distance to English profile: {distance_en}")
    print(f"Distance to German profile: {distance_de}")

    # 6. Определение языка с помощью detect_language
    result = detect_language(unknown_profile, en_profile, de_profile)

    # Выводим результат
    if result:
        print(f"Detected language: {result}")
    else:
        print("Error: Could not detect the language.")


if __name__ == "__main__":
    main()
