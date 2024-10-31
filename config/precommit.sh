set -ex

echo $1
if [[ "$1" == "smoke" ]]; then
  DIRS_TO_CHECK=(
    "config"
    "seminars"
    "lab_5_scrapper"
    "lab_3_ann_retriever"
  )
else
  DIRS_TO_CHECK=(
    "config"
    "seminars"
    "lab_3_ann_retriever"
  )
fi

python -m pylint "${DIRS_TO_CHECK[@]}"

mypy "${DIRS_TO_CHECK[@]}"

python -m flake8 "${DIRS_TO_CHECK[@]}"

if [[ "$1" != "smoke" ]]; then
  python -m pytest -m "mark10 and lab_1_classify_by_unigrams"
  python -m pytest -m "mark10 and lab_2_retrieval_w_bm25"
  python -m pytest -m "mark10 and lab_3_ann_retriever"
fi

