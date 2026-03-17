#!/usr/bin/env bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -xe
cd /home/pc2160/yangxiaochun/PaddleCppAPI
#source paddle_env/bin/activate
python3 /home/pc2160/yangxiaochun/PaddleCppAPI/PaddleCppAPITest/coverage/analyze_api_coverage.py \
    --header-dir /home/pc2160/yangxiaochun/PaddleCppAPI/Paddle/paddle/phi/api/include/compat \
    -I /home/pc2160/yangxiaochun/PaddleCppAPI/Paddle/paddle/phi/api/include \
    -I /home/pc2160/yangxiaochun/PaddleCppAPI/Paddle/paddle/phi/api/include/compat \
    --test-dir /home/pc2160/yangxiaochun/PaddleCppAPI/PaddleCppAPITest/test     \
    --output api_coverage_report.txt     \
    --json api_coverage_report.json
