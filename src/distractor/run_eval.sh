python preprocess.py
python ../contriever/generate_passage_embeddings.py \
    --model_name_or_path facebook/contriever \
    --output_dir grounding_embeddings  \
    --passages corpus_for_contriever.tsv \
    --shard_id 0 --num_shards 1 \

python ../contriever/passage_retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages corpus_for_contriever.tsv \
    --passages_embeddings "grounding_embeddings/*" \
    --data query_for_contriever.json \
    --output_dir output \
    --n_docs 200 \

python file_process.py
python postprocess.py
