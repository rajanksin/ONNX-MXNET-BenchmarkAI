#!/bin/bash

install_packages()
{
    echo "Installing mxnet ......."
    pip install mxnet --pre
    echo "Installing protobuf ......."
    apt-get install protobuf-compiler libprotoc-dev
    echo "Installing ONNX version 1.1.1 ........"
    export MACOSX_DEPLOYMENT_TARGET=10.9 
    export CC=clang 
    export CXX=clang++
    pip install onnx==1.1.1
}

get_models()
{
    if [ ! -d "models" ]; then
        mkdir models
    fi
    if [ ! -f "models/bvlc_alexnet.tar.gz" ]; then
        curl -o models/bvlc_alexnet.tar.gz   https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_alexnet.tar.gz
    fi
    if [ ! -f "models/bvlc_googlenet.tar.gz" ]; then
        curl -o models/bvlc_googlenet.tar.gz https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_googlenet.tar.gz
    fi
    
    for f in models/*.tar.gz; do tar xzf  "$f" -C models/; done
}

main() {
    install_packages
    get_models
}

main
