## 1. clone official Contriever repository
`git clone https://github.com/facebookresearch/contriever.git`

## 2. preprocess
`python preprocess.py`

## 3. generate passage embeddings
```
python ../contriever/generate_passage_embeddings.py \
    --model_name_or_path facebook/contriever \
    --output_dir grounding_embeddings  \
    --passages corpus_for_contriever.tsv \
    --shard_id 0 --num_shards 1 \
```

## 4. retrieve relevant passages
```
python ../contriever/passage_retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages corpus_for_contriever.tsv \
    --passages_embeddings "grounding_embeddings/*" \
    --data query_for_contriever.json \
    --output_dir output \
    --n_docs 200 \
```

## 5. postprocess
```
python file_process.py
python postprocess.py
```

## One Step
If you want to run all in one step, clone contriever repository and run `run_eval.sh`