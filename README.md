# OpenCode Zen AI Gateway

Gateway FastAPI dengan endpoint kompatibel OpenAI yang meneruskan chat completions ke OpenCode Zen.

## Fitur

- `GET /health` untuk status gateway.
- `GET /v1/models` untuk daftar model dari `zen_models.json` atau daftar fallback.
- `POST /v1/chat/completions` untuk request streaming maupun non-streaming.
- Model yang tidak dikenal diarahkan ke model default `nemotron-3.5-super-vl-free`.
- Proxy dipilih secara acak dari daftar lokal. Jika upstream mengembalikan `FreeTierError` yang menyebut jaringan atau abusive traffic, proxy tersebut di-blacklist dan gateway mencoba proxy lain.

## Menjalankan

Gunakan Python dan pasang dependensi:

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi httpx uvicorn
```

Jalankan server:

```bash
uvicorn ai_gateway:app --host 0.0.0.0 --port 8000
```

Gateway tersedia di `http://localhost:8000`.

## Proxy

Simpan URL proxy HTTP atau HTTPS, satu per baris, di `proxyscrape_premium_http_proxies.txt` yang berada di direktori yang sama dengan `ai_gateway.py`. Contoh format:

```text
http://proxy.example:8080
http://username:password@proxy.example:8080
```

Jangan commit file daftar proxy karena URL dapat berisi kredensial. File ini sudah dikecualikan melalui `.gitignore`. Proxy yang terkena blokir disimpan sebagai hash di `proxy_blacklist.txt`, bukan sebagai URL. Bila daftar lokal tidak memiliki proxy valid, gateway menggunakan `ZEN_PROXY` jika disetel, atau konfigurasi proxy standar HTTPX.

## Penggunaan

Daftar model:

```bash
curl http://localhost:8000/v1/models
```

Chat non-streaming:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron-3.5-super-vl-free","messages":[{"role":"user","content":"Halo"}],"stream":false}'
```

Untuk streaming, gunakan request yang sama dengan `"stream":true`.

## Catatan keamanan

Gateway ini tidak memvalidasi autentikasi client dan CORS mengizinkan semua origin. Jangan mengeksposnya langsung ke internet tanpa menambahkan autentikasi dan pembatasan akses di lapisan depan.

## File proyek

- `ai_gateway.py`: aplikasi FastAPI dan penerusan request.
- `zen_models.json`: cache daftar model.
- `opencode_request_body.json`: contoh body request.
- `test_gateway.py`: skrip uji manual; pengujian chat mengirim request ke backend upstream.