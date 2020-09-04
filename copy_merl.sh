srcdir="tessera:../../vinai/khanhpd4/MERL-RAV_dataset/merl_rav_organized/frontal/trainset"
dstdir="MERL-RAV_dataset/frontal/trainset"
mkdir -p "$dstdir"

find "$srcdir" -maxdepth 1 -type f |head -10|xargs scp -t "$dstdir"
