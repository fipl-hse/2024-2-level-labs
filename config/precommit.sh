set -ex

echo $1
if [[ "$1" == "smoke" ]]; then
  DIRS_TO_CHECK=(
    "config"
    "seminars"
    "lab_4_retrieval_w_clustering"
  )
else
  DIRS_TO_CHECK=(
    "config"
    "seminars"
    "lab_1_classify_by_unigrams"
    "lab_2_retrieval_w_bm25"
    "lab_3_ann_retriever"
    "lab_4_retrieval_w_clustering"
  )
fi

export PYTHONPATH=$(pwd)

python -m pylint "${DIRS_TO_CHECK[@]}"

python -m black --check "${DIRS_TO_CHECK[@]}"

mypy "${DIRS_TO_CHECK[@]}"

python -m flake8 "${DIRS_TO_CHECK[@]}"

python -m pytest -m "mark10 and lab_4_retrieval_w_clustering"

if [[ "$1" != "smoke" ]]; then
  python -m pytest -m "mark10 and lab_1_classify_by_unigrams"
  python -m pytest -m "mark10 and lab_2_retrieval_w_bm25"
  python -m pytest -m "mark10 and lab_3_ann_retriever"
  python -m pytest -m "mark10 and lab_4_retrieval_w_clustering"
fi

