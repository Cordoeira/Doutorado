#!/bin/bash
set -e
V=~/.venvs/a0bench
SRC=${1:?uso: bash ca_setup.sh <certificado_raiz_do_proxy.crt>}
cat /etc/ssl/certs/ca-certificates.crt "$SRC" > "$V/ca-bundle.crt"
openssl x509 -in "$SRC" -noout -subject
grep -q "A0BENCH_CA" "$V/bin/activate" || cat >> "$V/bin/activate" <<'EOT'

# A0BENCH_CA: rede com proxy Zscaler; pacote de CAs do sistema + CA raiz do Zscaler (so neste venv)
export A0BENCH_CA="$VIRTUAL_ENV/ca-bundle.crt"
export SSL_CERT_FILE="$A0BENCH_CA" REQUESTS_CA_BUNDLE="$A0BENCH_CA" CURL_CA_BUNDLE="$A0BENCH_CA" GIT_SSL_CAINFO="$A0BENCH_CA" PIP_CERT="$A0BENCH_CA"
EOT
source "$V/bin/activate"
echo "-- git test"; cd /tmp; rm -rf gt; timeout 120 git clone --filter=blob:none --quiet https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab.git gt && echo GIT_OK; rm -rf gt
echo "-- curl hf"; curl -sS -o /dev/null -w "%{http_code}\n" https://huggingface.co
echo "-- python ssl"; python -c "import urllib.request;print(urllib.request.urlopen('https://github.com',timeout=20).status)"
