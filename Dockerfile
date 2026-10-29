FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /dip

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

EXPOSE 8000

# No access to LLM:
ENV DIP_OFFLINE=1

# required to save in the wiki editor, empty for no auth
ENV DIP_EDIT_TOKEN=

# data dirs
# TODO: mount as separate volumes
ENV DIP_DATA_DIR="/dip/data-embeddinggemma2"
ENV DIP_IMAGE_WIKI_DIR="/dip/data-embeddinggemma2/wiki/images"
ENV DIP_IMAGE_DIR="/dip/data-images"

CMD ["uvicorn", "dip.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--workers", "1"]

