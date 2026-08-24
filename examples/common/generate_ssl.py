# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
# et Automatique)
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

"""Script to generate self-signed SSL certificates for example experiments.

Usage Notes
-----------
- By default, generates the certificates in the current directory (where you
have run this script).

- `alt_dns` and `alt_ips` arguments expect a list of strings, but as they are
CLI arguments, you must wrap them into a string (using quotes).
Ex:
```bash
python generate_ssl.py \
    --alt-dns='["my.domain.com", "localhost"]' \
    --alt-ips='["127.0.0.1"]'
```

"""

import fire

from declearn.utils.examples import generate_ssl_certificates

if __name__ == "__main__":
    fire.Fire(generate_ssl_certificates)
