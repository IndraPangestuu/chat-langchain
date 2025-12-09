# 📚 Panduan Instalasi Chat LangChain (Bahasa Indonesia)

Panduan ini akan membantu Anda menginstal dan menjalankan aplikasi Chat LangChain dari awal hingga deployment ke Vercel. Panduan ditulis dengan bahasa yang sederhana dan langkah-langkah yang jelas.

---

## 📋 Daftar Isi

1. [Persiapan Awal](#1-persiapan-awal)
2. [Instalasi Backend Python](#2-instalasi-backend-python)
3. [Instalasi Frontend Next.js](#3-instalasi-frontend-nextjs)
4. [Setup Database Vector (Weaviate)](#4-setup-database-vector-weaviate)
5. [Menjalankan Proses Ingest Data](#5-menjalankan-proses-ingest-data)
6. [Menjalankan Aplikasi Lokal](#6-menjalankan-aplikasi-lokal)
7. [Deploy ke Vercel](#7-deploy-ke-vercel)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Persiapan Awal

### 1.1 Software yang Diperlukan

Pastikan komputer Anda sudah terinstal software berikut:

| Software | Versi Minimum | Cara Cek | Link Download |
|----------|---------------|----------|---------------|
| Python | 3.11+ | `python --version` | [python.org](https://python.org) |
| Node.js | 18+ | `node --version` | [nodejs.org](https://nodejs.org) |
| Yarn | 1.22+ | `yarn --version` | `npm install -g yarn` |
| Git | Terbaru | `git --version` | [git-scm.com](https://git-scm.com) |
| UV (Python package manager) | Terbaru | `uv --version` | Lihat langkah di bawah |

### 1.2 Instalasi UV (Python Package Manager)

UV adalah package manager Python yang cepat. Instal dengan perintah:

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Mac/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.3 Clone Repository

```bash
git clone https://github.com/langchain-ai/chat-langchain.git
cd chat-langchain
```

### 1.4 Daftar Akun yang Diperlukan

Anda perlu mendaftar di layanan berikut (semua gratis untuk tier dasar):

1. **Weaviate Cloud** - Database vector untuk menyimpan dokumen
   - Daftar di: https://console.weaviate.cloud/
   
2. **Supabase** - Database untuk Record Manager
   - Daftar di: https://supabase.com/dashboard
   
3. **API LLM** - Pilih salah satu:
   - OpenAI: https://platform.openai.com/
   - Atau gunakan API compatible seperti `https://api.algion.dev/v1`

---

## 2. Instalasi Backend Python

### 2.1 Buat Virtual Environment dan Install Dependencies

Buka terminal di folder `chat-langchain`, lalu jalankan:

```bash
# Buat virtual environment dengan UV
uv venv

# Aktifkan virtual environment
# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate

# Install semua dependencies dari pyproject.toml
uv sync
```

### 2.2 Buat File Environment Variables

Buat file baru bernama `.env` di folder utama (root) project:

```bash
# Windows:
copy NUL .env

# Mac/Linux:
touch .env
```

### 2.3 Isi File .env dengan Konfigurasi

Buka file `.env` dengan text editor dan isi dengan konfigurasi berikut:

```env
# ============================================
# KONFIGURASI API LLM
# ============================================

# Jika menggunakan OpenAI langsung:
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Jika menggunakan API compatible (seperti api.algion.dev):
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://api.algion.dev/v1

# ============================================
# KONFIGURASI WEAVIATE (Vector Database)
# ============================================
WEAVIATE_URL=https://your-cluster-url.weaviate.network
WEAVIATE_API_KEY=your-weaviate-api-key

# ============================================
# KONFIGURASI RECORD MANAGER (Supabase)
# ============================================
RECORD_MANAGER_DB_URL=postgresql://postgres:your-password@db.xxxxx.supabase.co:5432/postgres

# ============================================
# KONFIGURASI LANGSMITH (Opsional - untuk monitoring)
# ============================================
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=chat-langchain

# Untuk mengambil prompts dari LangSmith
LANGCHAIN_PROMPT_API_KEY=your-langsmith-api-key
LANGCHAIN_PROMPT_API_URL=https://api.smith.langchain.com
```

### 2.4 Konfigurasi untuk OpenAI-Compatible API

Jika Anda menggunakan API yang compatible dengan OpenAI (seperti `https://api.algion.dev/v1`), Anda perlu memodifikasi beberapa file.

#### 2.4.1 Modifikasi file `backend/utils.py`

Buka file `backend/utils.py` dan ubah fungsi `load_chat_model`:

```python
"""Shared utility functions used in the project."""

import os
import uuid
from typing import Any, Literal, Optional, Union

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name

    model_kwargs = {"temperature": 0, "stream_usage": True}
    
    # Konfigurasi untuk OpenAI-compatible API
    if provider == "openai":
        openai_api_base = os.environ.get("OPENAI_API_BASE")
        if openai_api_base:
            model_kwargs["openai_api_base"] = openai_api_base
    
    if provider == "google_genai":
        model_kwargs["convert_system_message_to_human"] = True
    
    return init_chat_model(model, model_provider=provider, **model_kwargs)


# ... (fungsi lainnya tetap sama)
```

#### 2.4.2 Modifikasi file `backend/retrieval.py`

Buka file `backend/retrieval.py` dan ubah fungsi `make_text_encoder`:

```python
import os
from contextlib import contextmanager
from typing import Iterator

import weaviate
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_weaviate import WeaviateVectorStore

from backend.configuration import BaseConfiguration
from backend.constants import WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model_name = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings
            
            # Konfigurasi untuk OpenAI-compatible API
            openai_api_base = os.environ.get("OPENAI_API_BASE")
            if openai_api_base:
                return OpenAIEmbeddings(
                    model=model_name,
                    openai_api_base=openai_api_base
                )
            return OpenAIEmbeddings(model=model_name)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


# ... (fungsi lainnya tetap sama)
```

#### 2.4.3 Modifikasi file `backend/embeddings.py`

Buka file `backend/embeddings.py` dan ubah:

```python
import os
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


def get_embeddings_model() -> Embeddings:
    """Get the embeddings model with support for custom API base."""
    openai_api_base = os.environ.get("OPENAI_API_BASE")
    
    if openai_api_base:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            chunk_size=200,
            openai_api_base=openai_api_base
        )
    
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)
```

### 2.5 Konfigurasi Model Default (Opsional)

Jika Anda ingin mengubah model default, edit file `backend/retrieval_graph/configuration.py`:

```python
@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """The configuration for the agent."""

    # Ubah model sesuai kebutuhan
    query_model: str = field(
        default="openai/gpt-4o-mini",  # atau model lain yang tersedia di API Anda
        metadata={
            "description": "The language model used for processing and refining queries."
        },
    )

    response_model: str = field(
        default="openai/gpt-4o-mini",  # atau model lain yang tersedia di API Anda
        metadata={
            "description": "The language model used for generating responses."
        },
    )
```

---

## 3. Instalasi Frontend Next.js

### 3.1 Masuk ke Folder Frontend

```bash
cd frontend
```

### 3.2 Install Node Modules

```bash
# Menggunakan Yarn (direkomendasikan)
yarn install

# Atau menggunakan npm
npm install
```

### 3.3 Buat File Environment Variables Frontend

Buat file `.env.local` di folder `frontend`:

```bash
# Windows:
copy NUL .env.local

# Mac/Linux:
touch .env.local
```

### 3.4 Isi File .env.local

```env
# ============================================
# KONFIGURASI API BACKEND
# ============================================

# URL API Backend (untuk development lokal)
NEXT_PUBLIC_API_URL=http://localhost:8000
API_BASE_URL=http://localhost:8000

# Assistant ID (sesuaikan dengan konfigurasi LangGraph)
NEXT_PUBLIC_ASSISTANT_ID=chat

# ============================================
# KONFIGURASI LANGSMITH (untuk API key di frontend)
# ============================================
LANGCHAIN_API_KEY=your-langsmith-api-key

# ============================================
# KONFIGURASI WEAVIATE (jika diperlukan di frontend)
# ============================================
WEAVIATE_HOST=your-cluster-url.weaviate.network
WEAVIATE_API_KEY=your-weaviate-api-key
WEAVIATE_INDEX_NAME=LangChain_General_Guides_And_Tutorials_OpenAI_text_embedding_3_small
```

---

## 4. Setup Database Vector (Weaviate)

### 4.1 Buat Akun Weaviate Cloud

1. Buka https://console.weaviate.cloud/
2. Klik "Sign Up" dan buat akun baru
3. Setelah login, klik "Create Cluster"

### 4.2 Konfigurasi Cluster

1. **Cluster Name**: Beri nama cluster Anda (contoh: `chat-langchain`)
2. **Cloud Provider**: Pilih provider (AWS, GCP, atau Azure)
3. **Region**: Pilih region terdekat dengan lokasi Anda
4. **Tier**: Pilih "Sandbox" untuk gratis (cukup untuk development)
5. Klik "Create"

### 4.3 Dapatkan Credentials

Setelah cluster dibuat (tunggu beberapa menit):

1. **Cluster URL**: Salin URL cluster dari dashboard
   - Contoh: `https://my-cluster-abc123.weaviate.network`
   - Simpan sebagai `WEAVIATE_URL`

2. **API Key**: Klik "API Keys" → Salin API key
   - Simpan sebagai `WEAVIATE_API_KEY`

### 4.4 Setup Record Manager (Supabase)

1. Buka https://supabase.com/dashboard
2. Klik "New Project"
3. Isi detail project:
   - **Name**: `chat-langchain-records`
   - **Database Password**: Buat password yang kuat (SIMPAN INI!)
   - **Region**: Pilih region terdekat
4. Klik "Create new project"

5. Setelah project dibuat, pergi ke **Settings** → **Database**
6. Scroll ke bagian "Connection string"
7. Pilih "URI" dan salin connection string
8. Ganti `[YOUR-PASSWORD]` dengan password yang Anda buat
   - Contoh: `postgresql://postgres:MyPassword123@db.xxxxx.supabase.co:5432/postgres`
   - Simpan sebagai `RECORD_MANAGER_DB_URL`

---

## 5. Menjalankan Proses Ingest Data

Proses ingest akan mengambil dokumentasi LangChain dan menyimpannya ke database vector.

### 5.1 Pastikan Environment Variables Sudah Diset

Pastikan file `.env` di folder root sudah berisi:
- `WEAVIATE_URL`
- `WEAVIATE_API_KEY`
- `RECORD_MANAGER_DB_URL`
- `OPENAI_API_KEY`
- `OPENAI_API_BASE` (jika menggunakan API compatible)

### 5.2 Jalankan Script Ingest

```bash
# Pastikan Anda di folder root project dan virtual environment aktif
cd chat-langchain

# Aktifkan virtual environment jika belum
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Jalankan ingest
python -m backend.ingest
```

### 5.3 Proses Ingest

Script akan melakukan:
1. Mengambil dokumentasi dari sitemap LangChain (Python, JS, dan docs.langchain.com)
2. Parsing dan ekstraksi konten
3. Memecah dokumen menjadi chunks (4000 karakter per chunk)
4. Generate embeddings menggunakan OpenAI
5. Menyimpan ke Weaviate

**Catatan**: Proses ini bisa memakan waktu 30-60 menit tergantung koneksi internet.

### 5.4 Verifikasi Ingest Berhasil

Setelah selesai, Anda akan melihat log seperti:
```
INFO:backend.ingest:Indexing stats: {'num_added': 5000, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}
INFO:backend.ingest:General Guides and Tutorials now has this many vectors: 5000
```

---

## 6. Menjalankan Aplikasi Lokal

### 6.1 Menjalankan Backend

Backend menggunakan LangGraph. Ada dua cara menjalankannya:

#### Opsi A: Menggunakan LangGraph CLI (Direkomendasikan)

```bash
# Install LangGraph CLI jika belum
pip install langgraph-cli

# Jalankan server (mode test - tanpa license)
langgraph test

# Atau jika punya license LangGraph Cloud:
langgraph up
```

Server akan berjalan di `http://localhost:8000`

#### Opsi B: Menjalankan Manual dengan Python

Buat file `run_backend.py` di folder root:

```python
import uvicorn
from langgraph.serve import create_app
from backend.retrieval_graph.graph import graph

app = create_app(graph)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Lalu jalankan:
```bash
pip install uvicorn
python run_backend.py
```

### 6.2 Menjalankan Frontend

Buka terminal baru (jangan tutup terminal backend):

```bash
cd frontend

# Development mode
yarn dev

# Atau dengan npm
npm run dev
```

Frontend akan berjalan di `http://localhost:3000`

### 6.3 Akses Aplikasi

1. Buka browser dan akses `http://localhost:3000`
2. Anda akan melihat interface chat
3. Coba ketik pertanyaan tentang LangChain

---

## 7. Deploy ke Vercel

### 7.1 Persiapan Deployment

#### 7.1.1 Fork Repository (Jika Belum)

1. Buka https://github.com/langchain-ai/chat-langchain
2. Klik "Fork" di pojok kanan atas
3. Pilih akun GitHub Anda

#### 7.1.2 Push Perubahan Anda

Jika Anda sudah memodifikasi kode:

```bash
git add .
git commit -m "Add custom API configuration"
git push origin main
```

### 7.2 Deploy Backend

Backend Chat LangChain memerlukan LangGraph Cloud untuk deployment production. Ada beberapa opsi:

#### Opsi A: LangGraph Cloud (Direkomendasikan)

1. Daftar di https://langchain-ai.github.io/langgraph/cloud/
2. Ikuti panduan deployment mereka
3. Setelah deploy, Anda akan mendapat URL API seperti:
   `https://your-app.langgraph.cloud`

#### Opsi B: Self-Hosted dengan Docker

Buat `Dockerfile` di folder root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install UV
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY backend ./backend

# Install dependencies
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "langgraph", "up", "--host", "0.0.0.0", "--port", "8000"]
```

Deploy ke platform seperti:
- Railway (https://railway.app)
- Render (https://render.com)
- Google Cloud Run
- AWS ECS

#### Opsi C: Vercel Serverless (Terbatas)

**Catatan**: Vercel Serverless memiliki batasan waktu eksekusi (10 detik untuk free tier), sehingga tidak ideal untuk LLM yang membutuhkan waktu lama.

### 7.3 Deploy Frontend ke Vercel

#### 7.3.1 Buat Akun Vercel

1. Buka https://vercel.com/signup
2. Daftar dengan akun GitHub Anda

#### 7.3.2 Import Project

1. Di dashboard Vercel, klik "Add New..." → "Project"
2. Pilih repository `chat-langchain` dari daftar
3. Klik "Import"

#### 7.3.3 Konfigurasi Project

Di halaman konfigurasi:

1. **Framework Preset**: Pilih "Next.js"
2. **Root Directory**: Ketik `frontend`
3. **Build Command**: `yarn build` (biasanya sudah otomatis)
4. **Output Directory**: `.next` (biasanya sudah otomatis)

#### 7.3.4 Set Environment Variables

Klik "Environment Variables" dan tambahkan:

| Key | Value | Environment |
|-----|-------|-------------|
| `API_BASE_URL` | URL backend Anda (contoh: `https://your-app.langgraph.cloud`) | Production, Preview, Development |
| `NEXT_PUBLIC_API_URL` | URL publik API (contoh: `https://your-frontend.vercel.app/api`) | Production, Preview, Development |
| `LANGCHAIN_API_KEY` | API key LangSmith Anda | Production, Preview, Development |
| `NEXT_PUBLIC_ASSISTANT_ID` | `chat` | Production, Preview, Development |

#### 7.3.5 Deploy

Klik "Deploy" dan tunggu proses selesai (biasanya 2-5 menit).

### 7.4 Konfigurasi vercel.json

Buat atau update file `frontend/vercel.json`:

```json
{
  "framework": "nextjs",
  "buildCommand": "yarn build",
  "outputDirectory": ".next",
  "installCommand": "yarn install",
  "regions": ["sin1"],
  "functions": {
    "app/api/**/*.ts": {
      "maxDuration": 60
    }
  },
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        { "key": "Access-Control-Allow-Credentials", "value": "true" },
        { "key": "Access-Control-Allow-Origin", "value": "*" },
        { "key": "Access-Control-Allow-Methods", "value": "GET,POST,PUT,DELETE,OPTIONS" },
        { "key": "Access-Control-Allow-Headers", "value": "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, x-api-key" }
      ]
    }
  ],
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "/api/:path*"
    }
  ]
}
```

### 7.5 Konfigurasi Domain Custom (Opsional)

1. Di dashboard Vercel, buka project Anda
2. Pergi ke "Settings" → "Domains"
3. Klik "Add"
4. Masukkan domain Anda (contoh: `chat.yourdomain.com`)
5. Ikuti instruksi untuk mengatur DNS:
   - Tambahkan CNAME record yang mengarah ke `cname.vercel-dns.com`
   - Atau A record ke IP Vercel

### 7.6 Setup GitHub Actions untuk Auto-Ingest

Buat file `.github/workflows/ingest.yml`:

```yaml
name: Update Index

on:
  schedule:
    # Jalankan setiap hari jam 00:00 UTC
    - cron: '0 0 * * *'
  workflow_dispatch:  # Memungkinkan trigger manual

jobs:
  ingest:
    runs-on: ubuntu-latest
    environment: Indexing
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install UV
        run: pip install uv
      
      - name: Install dependencies
        run: uv sync
      
      - name: Run ingest
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          OPENAI_API_BASE: ${{ secrets.OPENAI_API_BASE }}
          WEAVIATE_URL: ${{ secrets.WEAVIATE_URL }}
          WEAVIATE_API_KEY: ${{ secrets.WEAVIATE_API_KEY }}
          RECORD_MANAGER_DB_URL: ${{ secrets.RECORD_MANAGER_DB_URL }}
        run: uv run python -m backend.ingest
```

Kemudian di GitHub:
1. Pergi ke repository → Settings → Environments
2. Buat environment baru bernama "Indexing"
3. Tambahkan secrets yang diperlukan

---

## 8. Troubleshooting

### 8.1 Error Umum dan Solusinya

#### Error: "OPENAI_API_KEY not found"

**Penyebab**: Environment variable tidak terbaca

**Solusi**:
```bash
# Pastikan file .env ada dan berisi OPENAI_API_KEY
# Cek dengan:
echo $OPENAI_API_KEY  # Mac/Linux
echo %OPENAI_API_KEY%  # Windows CMD
$env:OPENAI_API_KEY  # Windows PowerShell
```

#### Error: "Connection refused" saat akses localhost:3000

**Penyebab**: Frontend tidak bisa terhubung ke backend

**Solusi**:
1. Pastikan backend berjalan di port 8000
2. Cek `NEXT_PUBLIC_API_URL` di `.env.local`
3. Pastikan tidak ada firewall yang memblokir

#### Error: "Weaviate connection failed"

**Penyebab**: Credentials Weaviate salah atau cluster belum aktif

**Solusi**:
1. Cek `WEAVIATE_URL` dan `WEAVIATE_API_KEY`
2. Pastikan cluster Weaviate sudah "Ready" di dashboard
3. Coba akses URL cluster di browser

#### Error: "Rate limit exceeded"

**Penyebab**: Terlalu banyak request ke API

**Solusi**:
1. Tunggu beberapa menit
2. Gunakan API key dengan limit lebih tinggi
3. Implementasi retry logic dengan backoff

#### Error: "Module not found" saat menjalankan Python

**Penyebab**: Virtual environment tidak aktif atau dependencies belum terinstal

**Solusi**:
```bash
# Aktifkan virtual environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate  # Windows

# Install ulang dependencies
uv sync
```

#### Error: "Build failed" di Vercel

**Penyebab**: Biasanya karena dependencies atau konfigurasi

**Solusi**:
1. Cek build logs di Vercel dashboard
2. Pastikan `Root Directory` diset ke `frontend`
3. Cek semua environment variables sudah diset
4. Coba build lokal dulu: `cd frontend && yarn build`

### 8.2 Tips Debugging

1. **Cek Logs Backend**:
   ```bash
   # Jika menggunakan langgraph
   langgraph test --verbose
   ```

2. **Cek Network di Browser**:
   - Buka Developer Tools (F12)
   - Tab "Network"
   - Lihat request yang gagal

3. **Test API Manual**:
   ```bash
   curl http://localhost:8000/health
   ```

4. **Verifikasi Weaviate**:
   ```python
   import weaviate
   client = weaviate.connect_to_weaviate_cloud(
       cluster_url="YOUR_URL",
       auth_credentials=weaviate.classes.init.Auth.api_key("YOUR_KEY")
   )
   print(client.is_ready())
   ```

### 8.3 Mendapatkan Bantuan

Jika masih mengalami masalah:

1. **GitHub Issues**: https://github.com/langchain-ai/chat-langchain/issues
2. **LangChain Discord**: https://discord.gg/langchain
3. **Stack Overflow**: Tag `langchain`

---

## 📝 Catatan Penting

1. **Keamanan**: Jangan pernah commit file `.env` ke repository. Pastikan `.env` ada di `.gitignore`.

2. **Biaya**: 
   - Weaviate Sandbox gratis tapi terbatas
   - OpenAI API berbayar per token
   - Vercel free tier cukup untuk development

3. **Performance**:
   - Proses ingest pertama kali memakan waktu lama
   - Gunakan caching jika memungkinkan
   - Monitor penggunaan API untuk menghindari biaya berlebih

4. **Update**:
   - Jalankan ingest secara berkala untuk update dokumentasi
   - Gunakan GitHub Actions untuk otomatisasi

---

Selamat! Anda telah berhasil menginstal dan men-deploy Chat LangChain. 🎉


---

## 📎 Lampiran

### A. Contoh File Konfigurasi Lengkap

#### A.1 File `.env` (Backend - Root Folder)

```env
# ============================================
# KONFIGURASI API LLM (OpenAI-Compatible)
# ============================================
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_API_BASE=https://api.algion.dev/v1

# ============================================
# KONFIGURASI WEAVIATE
# ============================================
WEAVIATE_URL=https://my-cluster-abc123.weaviate.network
WEAVIATE_API_KEY=weaviate-api-key-here

# ============================================
# KONFIGURASI RECORD MANAGER
# ============================================
RECORD_MANAGER_DB_URL=postgresql://postgres:MySecurePassword123@db.abcdefgh.supabase.co:5432/postgres

# ============================================
# KONFIGURASI LANGSMITH
# ============================================
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls-your-langsmith-key
LANGCHAIN_PROJECT=chat-langchain-dev
LANGCHAIN_PROMPT_API_KEY=ls-your-langsmith-key
LANGCHAIN_PROMPT_API_URL=https://api.smith.langchain.com
```

#### A.2 File `frontend/.env.local`

```env
# ============================================
# KONFIGURASI API
# ============================================
NEXT_PUBLIC_API_URL=http://localhost:8000
API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_ASSISTANT_ID=chat

# ============================================
# KONFIGURASI LANGSMITH
# ============================================
LANGCHAIN_API_KEY=ls-your-langsmith-key
```

#### A.3 File `frontend/.env.production` (untuk Vercel)

```env
# ============================================
# KONFIGURASI API PRODUCTION
# ============================================
NEXT_PUBLIC_API_URL=https://your-domain.vercel.app/api
API_BASE_URL=https://your-backend.langgraph.cloud
NEXT_PUBLIC_ASSISTANT_ID=chat

# ============================================
# KONFIGURASI LANGSMITH
# ============================================
LANGCHAIN_API_KEY=ls-your-langsmith-key
```

### B. Modifikasi Kode untuk API Compatible

#### B.1 File `backend/utils.py` (Versi Lengkap)

```python
"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
"""

import os
import uuid
from typing import Any, Literal, Optional, Union

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    
    Supports custom OpenAI-compatible API endpoints via OPENAI_API_BASE env var.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name

    model_kwargs = {"temperature": 0, "stream_usage": True}
    
    # Support for OpenAI-compatible APIs (like api.algion.dev)
    if provider == "openai":
        openai_api_base = os.environ.get("OPENAI_API_BASE")
        if openai_api_base:
            model_kwargs["openai_api_base"] = openai_api_base
            print(f"Using custom OpenAI API base: {openai_api_base}")
    
    if provider == "google_genai":
        model_kwargs["convert_system_message_to_human"] = True
    
    return init_chat_model(model, model_provider=provider, **model_kwargs)


def reduce_docs(
    existing: Optional[list[Document]],
    new: Union[
        list[Document],
        list[dict[str, Any]],
        list[str],
        str,
        Literal["delete"],
    ],
) -> list[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It also combines existing documents with the new one based on the document ID.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, or a single string.
    """
    if new == "delete":
        return []

    existing_list = list(existing) if existing else []
    if isinstance(new, str):
        return existing_list + [
            Document(page_content=new, metadata={"uuid": str(uuid.uuid4())})
        ]

    new_list = []
    if isinstance(new, list):
        existing_ids = set(doc.metadata.get("uuid") for doc in existing_list)
        for item in new:
            if isinstance(item, str):
                item_id = str(uuid.uuid4())
                new_list.append(Document(page_content=item, metadata={"uuid": item_id}))
                existing_ids.add(item_id)

            elif isinstance(item, dict):
                metadata = item.get("metadata", {})
                item_id = metadata.get("uuid", str(uuid.uuid4()))

                if item_id not in existing_ids:
                    new_list.append(
                        Document(**item, metadata={**metadata, "uuid": item_id})
                    )
                    existing_ids.add(item_id)

            elif isinstance(item, Document):
                item_id = item.metadata.get("uuid")
                if item_id is None:
                    item_id = str(uuid.uuid4())
                    new_item = item.copy(deep=True)
                    new_item.metadata["uuid"] = item_id
                else:
                    new_item = item

                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)

    return existing_list + new_list
```

#### B.2 File `backend/retrieval.py` (Versi Lengkap)

```python
"""Retrieval module for the chat application."""

import os
from contextlib import contextmanager
from typing import Iterator

import weaviate
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_weaviate import WeaviateVectorStore

from backend.configuration import BaseConfiguration
from backend.constants import WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder.
    
    Supports custom OpenAI-compatible API endpoints via OPENAI_API_BASE env var.
    """
    provider, model_name = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings
            
            # Support for OpenAI-compatible APIs
            openai_api_base = os.environ.get("OPENAI_API_BASE")
            if openai_api_base:
                print(f"Using custom OpenAI API base for embeddings: {openai_api_base}")
                return OpenAIEmbeddings(
                    model=model_name,
                    openai_api_base=openai_api_base
                )
            return OpenAIEmbeddings(model=model_name)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


@contextmanager
def make_weaviate_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Iterator[BaseRetriever]:
    """Create a Weaviate retriever."""
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.classes.init.Auth.api_key(
            os.environ.get("WEAVIATE_API_KEY", "not_provided")
        ),
        skip_init_checks=True,
    ) as weaviate_client:
        store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
            text_key="text",
            embedding=embedding_model,
            attributes=["source", "title"],
        )
        search_kwargs = {**configuration.search_kwargs, "return_uuids": True}
        yield store.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Iterator[BaseRetriever]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    match configuration.retriever_provider:
        case "weaviate":
            with make_weaviate_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
```

#### B.3 File `backend/embeddings.py` (Versi Lengkap)

```python
"""Embeddings module for the chat application."""

import os
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


def get_embeddings_model() -> Embeddings:
    """Get the embeddings model with support for custom API base.
    
    Supports custom OpenAI-compatible API endpoints via OPENAI_API_BASE env var.
    """
    openai_api_base = os.environ.get("OPENAI_API_BASE")
    
    if openai_api_base:
        print(f"Using custom OpenAI API base for embeddings: {openai_api_base}")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            chunk_size=200,
            openai_api_base=openai_api_base
        )
    
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)
```

### C. Checklist Deployment

Gunakan checklist ini untuk memastikan semua langkah sudah dilakukan:

#### Persiapan
- [ ] Python 3.11+ terinstal
- [ ] Node.js 18+ terinstal
- [ ] Yarn terinstal
- [ ] UV terinstal
- [ ] Git terinstal

#### Akun & Credentials
- [ ] Akun Weaviate Cloud dibuat
- [ ] Cluster Weaviate dibuat dan aktif
- [ ] WEAVIATE_URL disimpan
- [ ] WEAVIATE_API_KEY disimpan
- [ ] Akun Supabase dibuat
- [ ] Project Supabase dibuat
- [ ] RECORD_MANAGER_DB_URL disimpan
- [ ] API Key LLM (OpenAI/Algion) didapat
- [ ] (Opsional) Akun LangSmith dibuat

#### Backend
- [ ] Virtual environment dibuat
- [ ] Dependencies terinstal (`uv sync`)
- [ ] File `.env` dibuat dan diisi
- [ ] Modifikasi kode untuk API compatible (jika perlu)
- [ ] Backend bisa dijalankan lokal

#### Frontend
- [ ] Node modules terinstal (`yarn install`)
- [ ] File `.env.local` dibuat dan diisi
- [ ] Frontend bisa dijalankan lokal
- [ ] Frontend bisa terhubung ke backend

#### Ingest Data
- [ ] Script ingest berhasil dijalankan
- [ ] Data tersimpan di Weaviate (cek jumlah vectors)

#### Deployment
- [ ] Repository di-fork ke GitHub
- [ ] Akun Vercel dibuat
- [ ] Project di-import ke Vercel
- [ ] Environment variables diset di Vercel
- [ ] Build berhasil
- [ ] Aplikasi bisa diakses via URL Vercel
- [ ] (Opsional) Domain custom dikonfigurasi
- [ ] (Opsional) GitHub Actions untuk auto-ingest disetup

### D. Referensi Tambahan

- **Dokumentasi LangChain**: https://python.langchain.com/docs/
- **Dokumentasi LangGraph**: https://langchain-ai.github.io/langgraph/
- **Dokumentasi Weaviate**: https://weaviate.io/developers/weaviate
- **Dokumentasi Vercel**: https://vercel.com/docs
- **Dokumentasi Next.js**: https://nextjs.org/docs

---

*Panduan ini dibuat untuk membantu instalasi Chat LangChain dengan dukungan OpenAI-compatible API. Jika ada pertanyaan atau masalah, silakan buka issue di repository GitHub.*
