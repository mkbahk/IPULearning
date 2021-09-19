## 아래의 내용을 .profile 에 넣어주세요.
# set Graphcore IPU Poplar SDK 2.0
export TF_POPLAR_FLAGS=--executable_cache_path=~/tmp/
export TMPDIR=~/tmp/
source ~/poplar_sdk-ubuntu_18_04-2.0.0+481-79b41f85d1/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64/enable.sh
source ~/poplar_sdk-ubuntu_18_04-2.0.0+481-79b41f85d1/popart-ubuntu_18_04-2.0.0+108156-165bbd8a64/enable.sh

# set Graphcore IPU Poplar SDK 2.1
export TF_POPLAR_FLAGS=--executable_cache_path=~/tmp/
export TMPDIR=~/tmp/
source ~/poplar_sdk-ubuntu_18_04-2.1.0+617-6bb5f5b742/poplar-ubuntu_18_04-2.1.0+145366-ce995e299d/enable.sh
source ~/poplar_sdk-ubuntu_18_04-2.1.0+617-6bb5f5b742/popart-ubuntu_18_04-2.1.0+145366-ce995e299d/enable.sh

# set Graphcore IPU Poplar SDK 2.2
export TF_POPLAR_FLAGS=--executable_cache_path=~/tmp/
export TMPDIR=~/tmp/
source ~/poplar_sdk-ubuntu_18_04-2.2.0+688-7a4ab80373/poplar-ubuntu_18_04-2.2.0+166889-feb7f3f2bb/enable.sh
source ~/poplar_sdk-ubuntu_18_04-2.2.0+688-7a4ab80373/popart-ubuntu_18_04-2.2.0+166889-feb7f3f2bb/enable.sh
