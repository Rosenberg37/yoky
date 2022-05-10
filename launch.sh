# Weibo large
# python main.py --corpus_name=weiboNER --batch_size=8 --pretrain_select=yechen/bert-large-chinese --word2vec_select=chinese --epochs=20 --cache_in_cur --evaluate=test

# Weibo base
# python main.py --corpus_name=weiboNER --batch_size=4 --pretrain_select=bert-base-chinese --word2vec_select=chinese --epochs=5 --evaluate=test

# python main.py --corpus_name=weiboNER --batch_size=4 --pretrain_select=bert-base-chinese --word2vec_select=chinese --epochs=5 --evaluate=test --kernels_size
# python main.py --corpus_name=weiboNER --batch_size=4 --pretrain_select=bert-base-chinese --word2vec_select=chinese --epochs=5 --evaluate=test --kernels_size 2
# python main.py --corpus_name=weiboNER --batch_size=4 --pretrain_select=bert-base-chinese --word2vec_select=chinese --epochs=5 --evaluate=test --kernels_size 2 2
# python main.py --corpus_name=weiboNER --batch_size=4 --pretrain_select=bert-base-chinese --word2vec_select=chinese --epochs=5 --evaluate=test --kernels_size 2 2 2
# python main.py --corpus_name=weiboNER --batch_size=4 --pretrain_select=bert-base-chinese --word2vec_select=chinese --epochs=5 --evaluate=test --kernels_size 2 2 2 2
# python main.py --corpus_name=weiboNER --batch_size=4 --pretrain_select=bert-base-chinese --word2vec_select=chinese --epochs=5 --evaluate=test --kernels_size 2 3
# python main.py --corpus_name=weiboNER --batch_size=4 --pretrain_select=bert-base-chinese --word2vec_select=chinese --epochs=5 --evaluate=test --kernels_size 2 3 2

# Conll2003 large
# python main.py --corpus_name=conll2003 --batch_size=28 --pretrain_select=bert-large-cased --word2vec_select=glove --word_dim=300 --epochs=20 --cache_in_cur --evaluate=test --checkpoint_name=conll03_large

# Conll2003 base
# python main.py --corpus_name=conll2003 --batch_size=4 --pretrain_select=bert-base-cased --word2vec_select=glove --epochs=5 --evaluate=test --concat

# kernels_size
# python main.py --corpus_name=conll2003 --batch_size=16 --pretrain_select=bert-base-cased --word2vec_select=glove --epochs=5 --evaluate=test --cache_in_cur --concat --kernels_size
# python main.py --corpus_name=conll2003 --batch_size=16 --pretrain_select=bert-base-cased --word2vec_select=glove --epochs=5 --evaluate=test --cache_in_cur --concat --kernels_size 2
# python main.py --corpus_name=conll2003 --batch_size=16 --pretrain_select=bert-base-cased --word2vec_select=glove --epochs=5 --evaluate=test --cache_in_cur --concat --kernels_size 2 2
# python main.py --corpus_name=conll2003 --batch_size=16 --pretrain_select=bert-base-cased --word2vec_select=glove --epochs=5 --evaluate=test --cache_in_cur --concat --kernels_size 2 2 2
# python main.py --corpus_name=conll2003 --batch_size=16 --pretrain_select=bert-base-cased --word2vec_select=glove --epochs=5 --evaluate=test --cache_in_cur --concat --kernels_size 2 2 2 2
# python main.py --corpus_name=conll2003 --batch_size=16 --pretrain_select=bert-base-cased --word2vec_select=glove --epochs=5 --evaluate=test --cache_in_cur --concat --kernels_size 2 3
# python main.py --corpus_name=conll2003 --batch_size=16 --pretrain_select=bert-base-cased --word2vec_select=glove --epochs=5 --evaluate=test --cache_in_cur --concat --kernels_size 2 3 2

# Genia base
# python main.py --corpus_name=genia --batch_size=4 --pretrain_select=dmis-lab/biobert-base-cased-v1.2 --word2vec_select=bio --epochs=5 --evaluate=test --concat


# mini
# python main.py --corpus_name=genia --batch_size=2 --pretrain_select=dmis-lab/biobert-base-cased-v1.2 --word2vec_select=bio --mini_size=100 --epochs=5 --evaluate=train
