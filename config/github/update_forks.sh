#!/bin/bash

set -x

while (( "$#" )); do
  case "$1" in
    --URL)
      FORKED_URL=$2
      shift 2
      ;;
    --USER)
      USER=$2
      shift 2
      ;;
    --STRATEGY)
      STRATEGY=$2
      shift 2
      ;;
    --AUTH)
      AUTH_TOKEN=$2
      shift 2
      ;;
    *)
      echo Unsupport argument $1
      exit 1
      ;;
  esac
done

source config/common.sh

configure_script

LABS=$(get_labs)

git config --global user.name "${USER}"
git config --global user.email "${USER}@users.noreply.github.com"
# Use HTTPS instead of SSH
git config --global url.https://github.com/.insteadOf git://github.com/

# Need to remove 'https://'
FORKED_URL="${FORKED_URL:8}"

echo $FORKED_URL
FORKED_URL="https://x-access-token:${AUTH_TOKEN}@${FORKED_URL}"

git checkout origin/main
git reset --hard origin/main

git remote add fork "${FORKED_URL}"
git fetch fork

git remote -v

git checkout fork/main

if [[ "$STRATEGY" == 'update_winner' ]]; then
  # Merge in favour of the original repository

  git merge --strategy-option theirs --no-edit origin/main

  git checkout origin/main -- lab_1_classify_by_unigrams
  git checkout origin/main -- lab_2_tokenize_by_bpe

  git checkout fork/main -- lab_2_tokenize_by_bpe/start.py
  git checkout fork/main -- lab_2_tokenize_by_bpe/main.py
  git checkout fork/main -- lab_2_tokenize_by_bpe/target_score.txt

  git commit -m "checkout labs from the origin repository"

elif [[ "$STRATEGY" == 'update_loser' ]]; then
  # Merge in favour of the forked repository

  git merge --strategy-option ours --no-edit origin/main

  git checkout origin/main -- lab_1_classify_by_unigrams
  git checkout origin/main -- lab_2_tokenize_by_bpe

  git checkout fork/main -- lab_1_classify_by_unigrams/start.py
  git checkout fork/main -- lab_1_classify_by_unigrams/main.py
  git checkout fork/main -- lab_1_classify_by_unigrams/target_score.txt

  git checkout fork/main -- lab_2_tokenize_by_bpe/start.py
  git checkout fork/main -- lab_2_tokenize_by_bpe/main.py
  git checkout fork/main -- lab_2_tokenize_by_bpe/target_score.txt

  git commit -m "checkout labs from the origin repository"

else
  # Just get the latest changes from the original repository
  git merge --no-edit origin/main
fi

DIFF=$(git diff --name-only HEAD@{0} HEAD@{1})

echo "files_diff=${DIFF}"

git push fork HEAD:main

git remote remove fork

# Instructions:
# WINNERS=(
#   fork1
#   fork2
# )
# for FORK in "${WINNERS[@]}"
# do
#     bash update_forks.sh \
#     --URL ${FORK} \
#     --USER artyomtugaryov \
#     --STRATEGY update_winner \
#     --AUTH TOKEN
# done
