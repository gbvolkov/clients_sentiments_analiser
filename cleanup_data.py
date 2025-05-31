#!/usr/bin/env python
# coding: utf-8
"""
Конвертирует выгрузку опросов -> датасет DeepPavlov (text,label)
Автор: ChatGPT, май-2025
"""

import re
import json
from pathlib import Path
from typing import Callable, Any
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


# ────────────────────────────────────────────────────────────────
# 0.  Справочник категорий (номер → текст)
#    Положительные 1-18, 38;  Отрицательные 19-36, 37
# ────────────────────────────────────────────────────────────────
CAT_MAP = {
     1: "Без замечаний (все понравилось).",
     2: "Процесс рассмотрения и согласования сделки. Скорость.",
     3: "Процесс рассмотрения и согласования сделки. Качество.",
     4: "Передача ПЛ. Скорость.",
     5: "Передача ПЛ. Качество процесса.",
     6: "Документооборот. Скорость подготовки и подписания договоров.",
     7: "Документооборот. Скорость подготовки бухгалтерских документов.",
     8: "Документооборот. Качество бухгалтерских документов.",
     9: "Обработка обращений клиента. Скорость.",
    10: "Обработка обращений клиента. Качество.",
    11: "Взаимодействие с сотрудниками. Скорость.",
    12: "Взаимодействие с сотрудниками. Качество.",
    13: "Условия лизинга. Стоимость.",
    14: "Условия лизинга. Штрафы. Пени.",
    15: "Поставщик. Скорость.",
    16: "Поставщик. Качество ПЛ. Качество сервисного обслуживания.",
    17: "Страхование. Перечень. Условия. Страховые случаи.",
    18: "Личный кабинет. Удобство. Функциональность.",
    19: "Все ужасно (нет положительного опыта).",
    20: "Процесс рассмотрения и согласования сделки. Скорость.",
    21: "Процесс рассмотрения и согласования сделки. Качество.",
    22: "Передача ПЛ. Скорость.",
    23: "Передача ПЛ. Качество процесса.",
    24: "Документооборот. Скорость подготовки и подписания договоров.",
    25: "Документооборот. Скорость подготовки бухгалтерских документов.",
    26: "Документооборот. Качество бухгалтерских документов.",
    27: "Обработка обращений клиента. Скорость.",
    28: "Обработка обращений клиента. Качество.",
    29: "Взаимодействие с сотрудниками. Скорость.",
    30: "Взаимодействие с сотрудниками. Качество.",
    31: "Условия лизинга. Стоимость.",
    32: "Условия лизинга. Штрафы. Пени.",
    33: "Поставщик. Скорость.",
    34: "Поставщик. Качество ПЛ. Качество сервисного обслуживания.",
    35: "Страхование. Перечень. Условия. Страховые случаи.",
    36: "Личный кабинет. Удобство. Функциональность.",
    37: "Прочее. Минусы.",
    38: "Прочее. Плюсы."
}

def normalise_category(raw_val: Any) -> str:
    """Вернёт текстовую категорию или pd.NA."""
    if pd.isna(raw_val):
        return pd.NA
    raw = str(raw_val).strip()
    if raw.isdigit():
        num = int(raw)
        return CAT_MAP.get(num, f"UNK_{num}")
    return raw

# ───────────────────────────────────────────────────────────────────────────
# ❶  Формулировки вопросов  --------------------------------------------------
#    Если они меняются — поправьте словарь.
# ───────────────────────────────────────────────────────────────────────────
QUESTION_TEXTS = {
    1: ("Какова вероятность того, что Вы порекомендуете нас своим "
        "партнёрам по шкале от 0 до 10, где 0 — «Ни в коем случае…», "
        "а 10 — «Обязательно порекомендую»?"),
    2: ("Расскажите, пожалуйста, что вам понравилось в работе с нами, "
        "а что необходимо улучшить?")
}

# ───────────────────────────────────────────────────────────────────────────
# ❷  Функция, извлекающая конкретный ответ из столбца «Комментарии»
#    pattern = "Ответ на <N> вопрос - ..."
# ───────────────────────────────────────────────────────────────────────────
def extract_answer(comments: str, qnum: int) -> str:
    """Вернёт чистый текст ответа на вопрос qnum или 'NO ANSWER'."""
    if pd.isna(comments):
        return "NO ANSWER"
    pat = rf"Ответ на\W*{qnum}\W*вопрос\s*-\s*(.*?)(?:;|$)"
    m = re.search(pat, comments, flags=re.IGNORECASE | re.DOTALL)
    answer = m.group(1).strip() if m else ""
    return answer or "NO ANSWER"


# ───────────────────────────────────────────────────────────────────────────
# ❸  Основное преобразование выгрузки -> two-column DataFrame
# ───────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
# 2. Основное преобразование
# ────────────────────────────────────────────────────────────────
def transform_df(df: pd.DataFrame) -> pd.DataFrame:

    df["orig_row"] = df.index
    df["ID"] = (
        df["Компания (Клиент)"].astype(str).str.lower()
        + "_" 
        + df["Дата обзвона"].astype(str)
    )

    df = df.rename(columns={
        "Номер Вопроса": "qnum",
        "Комментарии": "comments",
        "Номер ответа": "category",
        "Категория ответа": "answer_cat",
        "НастроениеОтвета 1": "sentiment"
    })

    # 2.1  Составляем text
    #answers = [extract_answer(c, int(q)) for c, q in zip(df["comments"], df["qnum"])]
    #df["text"] = [
    #    f"Q: {QUESTION_TEXTS.get(int(q), '')} A: {ans}".strip()
    #    for q, ans in zip(df["qnum"], answers)
    #]
    df["text"] = [
        f"Q: {QUESTION_TEXTS.get(int(q), '')} A: {c}".strip()
        for q, c in zip(df["qnum"], df["comments"].astype(str))
    ]

    # 2.2  Маппим тональность
    sent_map = {
        "Положительный отзыв": "POS",
        "Отрицательный отзыв": "NEG",
        "Негативный отзыв": "NEG"
    }
    df["sent"] = df["sentiment"].map(sent_map)
    
    # 2.3  Нормализуем категории
    df["cat_norm"] = df["category"].apply(normalise_category)

    # 7. Заполняем пустые cat_norm на основе answer_cat
    #    Преобразуем answer_cat в число (NaN там, где не удалось)
    answer_num = pd.to_numeric(df["answer_cat"], errors="coerce")
    #    Маркер пустых cat_norm: либо NaN, либо пустая строка после strip()
    #cat_empty = df["cat_norm"].isna() | (df["cat_norm"].astype(str).str.strip() == "")

    #    Для тех, у кого cat_norm пуст, присваиваем новую строку:
    #    если answer_num >= 6 → POS_Без замечаний (все понравилось).
    #    иначе (включая NaN или <6) → NEG_Все ужасно (нет положительного опыта).
    #df.loc[cat_empty & (answer_num >= 6), "cat_norm"] = "Без замечаний (все понравилось)."
    #df.loc[cat_empty & (answer_num < 6) | (cat_empty & answer_num.isna()), "cat_norm"] = (
    #    "Все ужасно (нет положительного опыта)."
    #)
    df.loc[(answer_num >= 5), "cat_norm"] = "Без замечаний (все понравилось)."
    df.loc[(answer_num < 5), "cat_norm"] = (
        "Все ужасно (нет положительного опыта)."
    )


    sent_empty = df["sent"].isna()
    df.loc[sent_empty & (answer_num >= 6), "sent"] = "POS"
    df.loc[sent_empty & ((answer_num < 6) | answer_num.isna()), "sent"] = "NEG"

    # 2.3  Итоговая метка 
    #df["label_txt"] = [
    #    f"{s}_{str(cat).strip()}"
    #    if pd.notna(s) and pd.notna(cat) and str(cat).strip()
    #    else pd.NA
    #    for s, cat in zip(df["sent"], df["cat_norm"])
    #]
    # 2.3  Итоговая метка 
    df["label_txt"] = [
        f"{str(cat).strip()}"
        if pd.notna(cat) and str(cat).strip()
        else pd.NA
        for cat in df["cat_norm"]
    ]


    # 2.4  Отбрасываем пустые
    #before, after = len(df), df["label"].notna().sum()
    #df = df[df["label"].notna()].reset_index(drop=True)
    #dropped = before - after
    #print(f"[info] отфильтровано пустых label: {dropped}")

    
    #df[["ID", "orig_row", "qnum", "comments", "text", "sent", "sentiment", "label_txt", "cat_norm", "category", "answer_cat"]].to_excel("./data/dataset_full.xlsx", index=False)


    # 2.4  фильтрация
    before = len(df)
    df = df[
        df["label_txt"].notna()                           # метка существует
        & ~df["label_txt"].str.contains("_UNK_", na=False)  # нет неизвестной категории
        & ~df["text"].str.contains("NO ANSWER", case=False, na=False)  # клиент что-то написал
    ].reset_index(drop=True)
    dropped = before - len(df)
    print(f"[info] исключено строк (UNK / NO ANSWER / NaN): {dropped}")

    #df[["ID", "orig_row", "qnum", "comments", "text", "sent", "sentiment", "label_txt", "cat_norm", "category", "answer_cat"]].to_excel("./data/dataset_clean.xlsx", index=False)


    if df.empty:
        raise ValueError("После фильтрации не осталось строк с валидной меткой")

    return df[["ID", "orig_row", "qnum", "comments", "text", "sent", "sentiment", "label_txt", "cat_norm", "category", "answer_cat"]]


# ───────────────────────────────────────────────────────────────────────────
# ❹  Универсальный пайплайн: чтение файла → train/valid/test + конфиг
# ───────────────────────────────────────────────────────────────────────────
def build_dataset(
    src: Path,
    out_dir: Path,
    transformer: Callable[[pd.DataFrame], pd.DataFrame] = transform_df,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Чтение Excel или CSV
    raw = pd.read_excel(src, sheet_name="dataset", header=0) if src.suffix.lower() in {".xlsx", ".xls"} \
        else pd.read_csv(src)

    data = transformer(raw)

    data.to_csv(out_dir / "dataset.csv", encoding="utf-8", header=True, index=False)

    print("Датасет сохранён")
    print("Распределение категорий:")
    label_counts = data["label_txt"].value_counts()
    print(label_counts)

    result = (
        data
        .groupby(['ID', 'qnum', 'text'])['sent']
        .agg(lambda x: 'NEG' if (x == 'NEG').any() else 'POS')
        .reset_index(name='label')
    )
    #print(result[:10])
    result.to_csv(out_dir / "sentiments_dataset.csv", encoding="utf-8", header=True, index=False)


def make_dp_config(out_dir: Path, cfg_name: str = "dp_bert_cls.json"):
    """Минимальный конфиг для fine-tune RuBERT-классификатора."""
    cfg = {
        "dataset_reader": {
            "class_name": "csv_dataset_reader",
            "x": "text",
            "y": "label",
            "delimiter": ","
        },
        "dataset_iterator": {"class_name": "split_valid_iterator"},
        "train": {
            "batch_size": 16,
            "epochs": 3,
            "learning_rate": 3e-5
        },
        "chainer": {
            "in": ["x"],
            "in_y": ["y"],
            "pipe": [
                {
                    "class_name": "bert_embedder",
                    "pretrained_bert": "DeepPavlov/rubert-base-cased",
                    "do_lower_case": False
                },
                {"class_name": "dense", "out_features": 768},
                {
                    "class_name": "sigmoid_cross_entropy",
                    "in": ["dense"],
                    "id2label": f"{out_dir/'label2id.json'}"
                }
            ],
            "out": ["sigmoid_cross_entropy"]
        }
    }
    with (out_dir / cfg_name).open("w", encoding="utf-8") as fp:
        json.dump(cfg, fp, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert survey table → DeepPavlov dataset")
    parser.add_argument("-s", "--src", type=Path, default=Path("data/202401-202412.xlsx"), help="Путь к исходному CSV/XLSX")
    parser.add_argument("-o", "--out", type=Path, default=Path("data"),
                        help="Каталог, куда писать train/valid/test")
    args = parser.parse_args()

    build_dataset(args.src, args.out)

    print("\nДанные готовы. "
          "Для обучения запустите:\n"
          f"  python -m deeppavlov train {args.out / 'dp_bert_cls.json'}")
