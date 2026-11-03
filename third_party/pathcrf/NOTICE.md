# PathCRF (third-party, unmodified)

Source: https://github.com/hyunsungkim-ds/pathcrf, commit 1b7c6f8 (2026-06-08), by Hyunsung Kim
et al. Licence: Mozilla Public License 2.0 (`LICENSE` in this folder); these files are
distributed under it, unmodified. Only the files needed for inference are copied, plus the
pretrained Dynamic Masked CRF of the paper (`saved/140`: trained on five Sportec Open DFL
matches, validation J03WQQ, test J03WR9).

Paper: H. Kim, K. Lee, S. Seo, S.-K. Ko, J. Yoon, C. Park. PathCRF: Ball-Free Soccer Event
Detection via Possession Path Inference from Player Trajectories. KDD 2026.
doi:10.1145/3770855.3818152.

Used by `possession/pathcrf.py` as ball-free team evidence (config `possession_v6.json`).

SHA-256 of the copied files (as in the source repository):

```
3f3d9e0024b1921b067d6f7f88deb4a60cbe7a78e76c64e3f1d7fc3b779b9d04  LICENSE
a4815e8b4a77edb163ac99ad5785390597d3bbd63b5b906bb51375d3e403816d  inference.py
4179ecaf6459a831f83676e433d4319cbaad0cfcf1e726eb0423526b9ad60db9  datatools/config.py
5b346aa4ea277b73874772e3ba09bdddf4b74414a18f5f6ac4cec1f9e33a0ec0  datatools/utils.py
8503a89fb93f1843ee37d56730ff86122e7d7bfd06ad5176d4a32324e6750f9a  datatools/postprocess.py
5f3d3a045154e5480de24bbf46b0bf8daf63b2d4640e5441d7a3b31315a25cda  datatools/matplotsoccer.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  models/__init__.py
511e69f52dc833941c922ebab84f981995a77a78d61d3cac26e555bc5f46f7b2  models/utils.py
ad1690e14f7693d26e19f11fe549367dba45fb4f470bf70972259665b378e963  models/set_lstm.py
8eb08e643886c3d986c110d46ebe5e5d4307a8553fa9ee1bc3bc93c1a9f85869  models/dynamic_sparse_crf.py
acd217ef153c60fc9a6c84f0b1ca4b72bef3a2ed132127e571b076539ba93ab1  models/dynamic_dense_crf.py
0ba5b56d0a7caed8135a8d73303a7361332907cece3d872bc4ea589ef2e62385  models/static_dense_crf.py
bdad038a32a98335c5b8f8c89f697944e00b6e3e43a9fb49dd9b746614904639  models/static_sparse_crf.py
0f96a7c887bcf9040f91731479c6367e7abed60297f2d8224725009c6ebe25c3  set_transformer/attention.py
9f94aa29dec3d3f7431f82d75d9399799583964e8633325e7bc9cef7865b0611  set_transformer/blocks.py
b072091111475b8928f5cedb482d712367613b30116551c3ef23505c49cc6060  set_transformer/model.py
b83d9c47fd5817c816a163b709b93d51b873bfc0e4f59a6d0b2254cf486c2489  saved/140/args.json
c82235a17416354b5544f85721ca850304765737a2d9d5435dcdefd52a0d3700  saved/140/model/state_dict_best.pt
```
