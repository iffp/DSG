#!/bin/bash

# Define root directory, dataset, and other variables
ROOT_DIR="/research/projects/zp128/RangeIndexWithRandomInsertion/"
N=10000
index_k_arr=(4 8 16 32)
ef_construction_arr=(100)
ef_max_arr=(400 600 800 1000)
alpha=1.0
# Define dataset paths
# List of datasets #
DATASETS=(
   "wiki-image" "deep" "yt8m-video")

# List of dataset paths with ROOT_DIR appended
DATASET_PATHS=(
  "${ROOT_DIR}data/wiki_image_embedding.fvecs" 
  "${ROOT_DIR}data/deep_sorted_10M.fvecs" 
  "${ROOT_DIR}data/exp2_used_data/yt8m_video_1_2m.fvecs") #

QUERY_PATHS=("${ROOT_DIR}data/wiki_image_querys.fvecs" "${ROOT_DIR}data/deep1B_queries.fvecs" "${ROOT_DIR}data/yt8m_video_querys_10k.fvecs") # Example: wiki_image_querys

# always check the number
GROUNDTRUTH_PATHs=("../groundtruth/wiki-image_benchmark-groundtruth-deep-10k-num1000-k10.arbitrary.cvs"
  "../groundtruth/deep_benchmark-groundtruth-deep-10k-num1000-k10.arbitrary.cvs"
  "../groundtruth/yt8m-video_benchmark-groundtruth-deep-10k-num1000-k10.arbitrary.cvs")

INDEX_PATH="${ROOT_DIR}index/tune_m_z"
# Iterate over methods and versions
for i in $(seq 0 $((${#DATASETS[@]} - 1))); do
  DATASET="${DATASETS[$i]}"

  # only use deep dataset
  if [ "$DATASET" != "deep" ]; then
    continue
  fi

  if [ "$DATASET" == "yt8m-video" ]; then
    alpha=1.2
  fi

  DATASET_PATH="${DATASET_PATHS[$i]}"
  QUERY_PATH="${QUERY_PATHS[$i]}"
  GROUNDTRUTH_PATH="${GROUNDTRUTH_PATHs[$i]}"

  # Determine index size suffix
  if [ "$N" -ge 1000000 ]; then
    INDEX_SIZE="$((N / 1000000))m"
  else
    INDEX_SIZE="$((N / 1000))k"
  fi

  for index_k in "${index_k_arr[@]}"; do
    for ef_max in "${ef_max_arr[@]}"; do
      for ef_construction in "${ef_construction_arr[@]}"; do
        # Define log path
        LOG_PATH="${ROOT_DIR}log/tune_m_z/${DATASET}_${index_k}_${ef_max}_${ef_construction}.log"

        # Display the command being executed
        echo "Running query for:"
        echo "  Dataset      : $DATASET"
        echo "  index_k      : $index_k"
        echo "  ef_max       : $ef_max"
        echo "  alpha        : $alpha"
        echo "  ef_construction: $ef_construction"
        echo "  Log File     : $LOG_PATH"

        # First build the index
        ./benchmark/build_index -alpha $alpha -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max \
          -dataset $DATASET -dataset_path $DATASET_PATH \
          -index_path "${INDEX_PATH}/${DATASET}_${index_k}_${ef_max}_${ef_construction}.bin" >>"$LOG_PATH"

        # Execute the command and log the output
        ./benchmark/query_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max \
          -dataset $DATASET -dataset_path $DATASET_PATH -query_path $QUERY_PATH \
          -groundtruth_path $GROUNDTRUTH_PATH -index_path "${INDEX_PATH}/${DATASET}_${index_k}_${ef_max}_${ef_construction}.bin" >>"$LOG_PATH"

        echo "Finished query with index_k: $index_k, ef_max: $ef_max, ef_construction: $ef_construction"
        echo "------------------------------------------------------"
      done
    done
  done
done


exit 0
