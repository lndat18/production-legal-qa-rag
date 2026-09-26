#!/bin/sh
# Chạy đúng 1 lần bởi entrypoint chính chủ của image postgres:17-alpine, khi volume
# `postgres_data` được tạo lần đầu (deploy_spec.md mục 4). Tạo 2 database: `openwebui`
# (dữ liệu OpenWebUI) và `chatbot` (chatlog — chat_turns). $POSTGRES_USER đã có sẵn trong
# môi trường do docker-entrypoint.sh của image postgres export trước khi chạy script này.
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE openwebui OWNER "$POSTGRES_USER";
    CREATE DATABASE chatbot OWNER "$POSTGRES_USER";
EOSQL
