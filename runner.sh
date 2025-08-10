#!/bin/bash
set -e

cd backend
python optimizer.py \
  --func func.c \
  --sig "size_t count_above_threshold(const uint8_t* data, size_t len)" \
  --test test.c \
  --bench bench.c \
  --iterations 5 \
  --output optimized.c
