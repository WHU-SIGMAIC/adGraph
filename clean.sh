#!/bin/bash

rm 9*.loop

rm -rf cpp/build/*

mkdir -p cpp/build/googletest
unzip ../googletest-release-1.8.0.zip -d cpp/build/googletest
